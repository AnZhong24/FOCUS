#!/usr/bin/env python3
"""Run throughput experiment suites and periodically check expectations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


MODE_CHOICES = ('base', 'delayed', 'sub_block')


@dataclass
class ExperimentSpec:
    index: int
    mode: str
    model: str
    batch_size: int
    dllm_block_length: int
    cache_block_seq_len: int
    sub_block_size: Optional[int]
    csv_path: Path
    log_path: Path
    err_path: Path
    command: List[str]


def _slugify(value: str) -> str:
    value = value.strip().replace('/', '__')
    value = re.sub(r'[^A-Za-z0-9._-]+', '_', value)
    return value.strip('_') or 'item'


def _now_tag() -> str:
    return time.strftime('%Y%m%d-%H%M%S', time.localtime())


def _parse_int_list(raw_values: Iterable[str]) -> List[int]:
    values: List[int] = []
    for raw in raw_values:
        for chunk in raw.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            values.append(int(chunk))
    return values


def _infer_block_length(model: str, explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    match = re.search(r'-b(\d+)\b', model)
    if match is not None:
        return int(match.group(1))
    lower = model.lower()
    if 'llada' in lower:
        return 32
    raise ValueError(
        f'Cannot infer dllm block length from model `{model}`. '
        'Pass --dllm-block-length explicitly.')


def _mode_cli_flags(mode: str, spec: ExperimentSpec) -> List[str]:
    if mode == 'base':
        return ['--dllm-confidence-threshold', '0.9']
    if mode == 'delayed':
        return [
            '--dllm-confidence-threshold', '0.9',
            '--dllm-enable-delayed-cache',
        ]
    if mode == 'sub_block':
        if spec.sub_block_size is None:
            raise ValueError('sub_block mode requires sub_block_size to be set.')
        return [
            '--dllm-confidence-threshold', '0.9',
            '--dllm-enable-delayed-cache',
            '--dllm-enable-sub-block-cache-reuse',
            '--dllm-sub-block-size', str(spec.sub_block_size),
        ]
    raise ValueError(f'Unsupported mode: {mode}')


def _base_command(args, spec: ExperimentSpec) -> List[str]:
    cmd = [
        args.python,
        'benchmark/profile_throughput.py',
        '--backend',
        'pytorch',
        '--eager-mode',
        '--skip-tokenize',
        '--skip-detokenize',
        '--num-prompts',
        str(args.num_prompts),
        '--concurrency',
        str(spec.batch_size),
        '--dllm-block-length',
        str(spec.dllm_block_length),
        '--dllm-denoising-steps',
        str(spec.dllm_block_length),
        '--cache-block-seq-len',
        str(spec.cache_block_seq_len),
        '--max-new-tokens',
        str(args.max_new_tokens),
        '--csv',
        str(spec.csv_path),
        '--seed',
        str(args.seed),
    ]
    if args.use_uvloop:
        cmd.append('--use-uvloop')
    if args.dataset_format != 'auto':
        cmd.extend(['--dataset-format', args.dataset_format])
    if args.hf_split != 'train':
        cmd.extend(['--hf-split', args.hf_split])
    if args.hf_streaming:
        cmd.append('--hf-streaming')
    if args.hf_data_file is not None:
        cmd.extend(['--hf-data-file', args.hf_data_file])
    if args.hf_revision is not None:
        cmd.extend(['--hf-revision', args.hf_revision])
    if args.max_scan_examples is not None:
        cmd.extend(['--max-scan-examples', str(args.max_scan_examples)])
    if args.repeat_block_detect:
        cmd.append('--repeat-block-detect')
        cmd.extend(['--repeat-block-window', str(spec.dllm_block_length)])
        cmd.extend(['--repeat-block-threshold', str(args.repeat_block_threshold)])
    for item in args.extra_arg:
        cmd.append(item)
    cmd.extend(_mode_cli_flags(spec.mode, spec))
    cmd.extend([args.dataset, spec.model])
    return cmd


def _build_experiments(args, run_dir: Path) -> List[ExperimentSpec]:
    raw_dir = run_dir / 'raw'
    logs_dir = run_dir / 'logs'
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    experiments: List[ExperimentSpec] = []
    idx = 0
    for model in args.model:
        dllm_block_length = _infer_block_length(model, args.dllm_block_length)
        cache_block_seq_len = args.cache_block_seq_len or dllm_block_length
        for batch_size in args.batch_size:
            for mode in args.mode:
                stem = f'{idx:03d}_{mode}_{_slugify(model)}_b{batch_size}'
                spec = ExperimentSpec(
                    index=idx,
                    mode=mode,
                    model=model,
                    batch_size=batch_size,
                    dllm_block_length=dllm_block_length,
                    cache_block_seq_len=cache_block_seq_len,
                    sub_block_size=args.sub_block_size,
                    csv_path=raw_dir / f'{stem}.csv',
                    log_path=logs_dir / f'{stem}.log',
                    err_path=logs_dir / f'{stem}.err',
                    command=[],
                )
                spec.command = _base_command(args, spec)
                experiments.append(spec)
                idx += 1
    return experiments


def _read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'CSV file is empty: {path}')
    return rows[-1]


def _to_number(value: str):
    lowered = value.strip().lower()
    if lowered in ('inf', '+inf'):
        return float('inf')
    if lowered == '-inf':
        return float('-inf')
    if lowered == 'nan':
        return float('nan')
    try:
        if any(ch in lowered for ch in '.e'):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_log_metrics(log_text: str) -> Dict[str, object]:
    patterns = {
        'input_tokens_reported': r'#Input tokens:\s+(\d+)',
        'prompts_reported': r'#Prompts:\s+(\d+)',
        'total_requests': r'Total requests\s+(\d+)',
        'successful_requests': r'Successful requests\s+(\d+)',
        'output_throughput_reported': r'Output throughput \(tok/s\)\s+([^\s]+)',
        'request_throughput_reported': r'Request throughput \(req/s\)\s+([^\s]+)',
        'input_throughput_reported': r'Input throughput \(tok/s\)\s+([^\s]+)',
        'benchmark_duration_reported': r'Benchmark duration\s+([^\s]+)',
    }
    parsed: Dict[str, object] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if match is None:
            continue
        parsed[key] = _to_number(match.group(1))
    return parsed


def _load_result_record(spec: ExperimentSpec, returncode: int, started_at: float, ended_at: float) -> Dict[str, object]:
    record: Dict[str, object] = {
        'index': spec.index,
        'mode': spec.mode,
        'model': spec.model,
        'batch_size': spec.batch_size,
        'dllm_block_length': spec.dllm_block_length,
        'cache_block_seq_len': spec.cache_block_seq_len,
        'sub_block_size': spec.sub_block_size,
        'returncode': returncode,
        'started_at': started_at,
        'ended_at': ended_at,
        'wall_time_sec': ended_at - started_at,
        'csv_path': str(spec.csv_path),
        'log_path': str(spec.log_path),
        'err_path': str(spec.err_path),
    }
    if spec.csv_path.exists():
        csv_row = _read_single_row_csv(spec.csv_path)
        for key, value in csv_row.items():
            record[key] = _to_number(value)
    if spec.log_path.exists():
        log_text = spec.log_path.read_text(encoding='utf-8', errors='replace')
        record.update(_parse_log_metrics(log_text))
    if spec.err_path.exists():
        err_text = spec.err_path.read_text(encoding='utf-8', errors='replace')
        if returncode != 0 and err_text.strip():
            record['stderr_tail'] = '\n'.join(err_text.strip().splitlines()[-20:])
    return record


def _write_json(path: Path, payload: object):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def _write_summary_csv(path: Path, records: List[Dict[str, object]]):
    if not records:
        return
    keys = sorted({key for record in records for key in record.keys()})
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _evaluate_expectations(records: List[Dict[str, object]], args) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def add_check(kind: str, name: str, status: str, **payload):
        checks.append({'kind': kind, 'name': name, 'status': status, **payload})

    for record in records:
        label = f'{record["mode"]}:{record["model"]}:b{record["batch_size"]}'
        if int(record.get('returncode', 1)) != 0:
            add_check('sanity', label, 'fail', reason='non_zero_returncode', value=record.get('returncode'))
            continue

        total_requests = int(record.get('total_requests', record.get('prompts_reported', 0) or 0))
        successful = int(record.get('successful_requests', record.get('completed', 0) or 0))
        output_tp = float(record.get('output_throughput', 0) or 0)
        duration = float(record.get('duration', record.get('wall_time_sec', 0)) or 0)

        if total_requests <= 0:
            add_check('sanity', label, 'fail', reason='no_requests', value=total_requests)
        elif successful <= 0:
            add_check('sanity', label, 'fail', reason='no_successful_requests', value=successful)
        elif successful != total_requests:
            add_check('sanity', label, 'warn', reason='partial_success', successful=successful, total=total_requests)
        else:
            add_check('sanity', label, 'pass', reason='all_requests_succeeded', successful=successful)

        if duration <= 0:
            add_check('sanity', f'{label}:duration', 'fail', reason='non_positive_duration', value=duration)
        elif output_tp < args.min_output_throughput:
            add_check('sanity', f'{label}:throughput', 'fail', reason='low_output_throughput', value=output_tp)
        else:
            add_check('sanity', f'{label}:throughput', 'pass', reason='output_throughput_ok', value=output_tp)

    grouped: Dict[tuple, Dict[str, Dict[str, object]]] = {}
    for record in records:
        key = (record['model'], record['batch_size'])
        grouped.setdefault(key, {})[record['mode']] = record

    for (model, batch_size), modes in grouped.items():
        if 'delayed' in modes and 'sub_block' in modes:
            delayed_tp = float(modes['delayed'].get('output_throughput', 0) or 0)
            sub_block_tp = float(modes['sub_block'].get('output_throughput', 0) or 0)
            ratio = (sub_block_tp / delayed_tp) if delayed_tp > 0 else float('inf')
            status = 'pass'
            if ratio < args.sub_block_min_ratio_vs_delayed:
                status = 'fail'
            elif ratio < args.sub_block_target_ratio_vs_delayed:
                status = 'warn'
            add_check('comparison',
                      f'sub_block_vs_delayed:{model}:b{batch_size}',
                      status,
                      ratio=ratio,
                      delayed_output_throughput=delayed_tp,
                      sub_block_output_throughput=sub_block_tp)

        if args.delayed_min_ratio_vs_base is not None and 'base' in modes and 'delayed' in modes:
            base_tp = float(modes['base'].get('output_throughput', 0) or 0)
            delayed_tp = float(modes['delayed'].get('output_throughput', 0) or 0)
            ratio = (delayed_tp / base_tp) if base_tp > 0 else float('inf')
            status = 'pass' if ratio >= args.delayed_min_ratio_vs_base else 'fail'
            add_check('comparison',
                      f'delayed_vs_base:{model}:b{batch_size}',
                      status,
                      ratio=ratio,
                      base_output_throughput=base_tp,
                      delayed_output_throughput=delayed_tp)

    counts = {'pass': 0, 'warn': 0, 'fail': 0}
    for check in checks:
        counts[check['status']] += 1
    return {'counts': counts, 'checks': checks}


def _print_check_summary(report: Dict[str, object]):
    counts = report['counts']
    print(f'[CHECK] pass={counts["pass"]} warn={counts["warn"]} fail={counts["fail"]}')
    for check in report['checks']:
        if check['status'] not in ('warn', 'fail'):
            continue
        print(f'[{check["status"].upper()}] {check["name"]}: {json.dumps(check, ensure_ascii=True)}')


def _run_experiment(spec: ExperimentSpec, env: Dict[str, str]) -> Dict[str, object]:
    print(f'\n[RUN] #{spec.index} mode={spec.mode} model={spec.model} batch={spec.batch_size}')
    print(f'[CMD] {" ".join(spec.command)}')
    started_at = time.time()
    with spec.log_path.open('w', encoding='utf-8') as log_f, spec.err_path.open('w', encoding='utf-8') as err_f:
        proc = subprocess.Popen(spec.command, cwd=Path(__file__).resolve().parents[1], env=env, stdout=log_f, stderr=err_f)
        returncode = proc.wait()
    ended_at = time.time()
    record = _load_result_record(spec, returncode, started_at, ended_at)
    if returncode == 0:
        print(f'[DONE] #{spec.index} wall_time={record["wall_time_sec"]:.2f}s '
              f'output_tp={record.get("output_throughput", "n/a")}')
    else:
        print(f'[FAIL] #{spec.index} returncode={returncode}')
    return record


def _save_manifest(path: Path, experiments: List[ExperimentSpec], records: List[Dict[str, object]]):
    completed = {int(record['index']) for record in records}
    payload = {
        'experiments': [asdict(spec) | {
            'csv_path': str(spec.csv_path),
            'log_path': str(spec.log_path),
            'err_path': str(spec.err_path),
        } for spec in experiments],
        'completed_indices': sorted(completed),
        'num_completed': len(completed),
        'num_total': len(experiments),
    }
    _write_json(path, payload)


def parse_args():
    parser = argparse.ArgumentParser(description='Run throughput experiment suites and collect structured results.')
    parser.add_argument('dataset', type=str, help='Dataset path or HuggingFace dataset id.')
    parser.add_argument('--model', action='append', required=True, help='Model repo id or local path. Repeatable.')
    parser.add_argument('--mode',
                        action='append',
                        default=None,
                        choices=MODE_CHOICES,
                        help='Experiment mode to run. Repeatable.')
    parser.add_argument('--batch-size',
                        action='append',
                        default=None,
                        help='Batch size(s), comma-separated or repeated.')
    parser.add_argument('--dllm-block-length',
                        type=int,
                        default=None,
                        help='Override DLLM block length for all models. If omitted, infer from model name.')
    parser.add_argument('--cache-block-seq-len',
                        type=int,
                        default=None,
                        help='Override cache block seq len. Defaults to dllm block length.')
    parser.add_argument('--sub-block-size',
                        type=int,
                        default=8,
                        help='Sub-block size for sub_block mode.')
    parser.add_argument('--num-prompts', type=int, default=5000, help='Number of prompts per experiment.')
    parser.add_argument('--max-new-tokens', type=int, default=2048, help='Max new tokens per request.')
    parser.add_argument('--dataset-format',
                        type=str,
                        default='auto',
                        choices=['auto', 'sharegpt', 'wildchat', 'math'],
                        help='Dataset format forwarded to profile_throughput.py.')
    parser.add_argument('--hf-split', type=str, default='train')
    parser.add_argument('--hf-streaming', action='store_true', default=False)
    parser.add_argument('--hf-data-file', type=str, default=None)
    parser.add_argument('--hf-revision', type=str, default=None)
    parser.add_argument('--max-scan-examples', type=int, default=None)
    parser.add_argument('--python',
                        type=str,
                        default='/home/zhonga/miniforge3/envs/FOCUS/bin/python',
                        help='Python executable used to launch benchmark/profile_throughput.py.')
    parser.add_argument('--results-dir', type=str, default='results', help='Base directory for experiment outputs.')
    parser.add_argument('--run-name', type=str, default=None, help='Optional explicit run directory name.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-uvloop', action='store_true', default=True)
    parser.add_argument('--no-use-uvloop', dest='use_uvloop', action='store_false')
    parser.add_argument('--repeat-block-detect', action='store_true', default=True)
    parser.add_argument('--no-repeat-block-detect', dest='repeat_block_detect', action='store_false')
    parser.add_argument('--repeat-block-threshold', type=int, default=3)
    parser.add_argument('--extra-arg', action='append', default=[], help='Extra raw arg forwarded to profile_throughput.py.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing them.')
    parser.add_argument('--disable-check-env', action='store_true', default=True)
    parser.add_argument('--enable-check-env', dest='disable_check_env', action='store_false')
    parser.add_argument('--hf-offline', action='store_true', default=False)
    parser.add_argument('--transformers-offline', action='store_true', default=True)
    parser.add_argument('--hf-modules-cache', type=str, default='/tmp/hf_modules')
    parser.add_argument('--check-every-experiments',
                        type=int,
                        default=1,
                        help='Run expectation checks after this many completed experiments.')
    parser.add_argument('--check-interval-seconds',
                        type=int,
                        default=300,
                        help='Run expectation checks when this much wall time has elapsed since the previous check.')
    parser.add_argument('--stop-on-fail', action='store_true', help='Stop the suite after any failed expectation.')
    parser.add_argument('--min-output-throughput',
                        type=float,
                        default=0.1,
                        help='Minimum acceptable output throughput for sanity checks.')
    parser.add_argument('--sub-block-min-ratio-vs-delayed',
                        type=float,
                        default=0.70,
                        help='Fail if sub_block output throughput falls below this ratio of delayed output throughput.')
    parser.add_argument('--sub-block-target-ratio-vs-delayed',
                        type=float,
                        default=1.0,
                        help='Warn if sub_block output throughput is below this ratio of delayed output throughput.')
    parser.add_argument('--delayed-min-ratio-vs-base',
                        type=float,
                        default=None,
                        help='Optional failure threshold for delayed/base output throughput ratio.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode is None:
        args.mode = list(MODE_CHOICES)
    args.mode = list(dict.fromkeys(args.mode))
    if args.batch_size is None:
        args.batch_size = [32, 64, 128, 256]
    else:
        args.batch_size = _parse_int_list(args.batch_size)

    run_name = args.run_name or f'throughput_suite_{_now_tag()}'
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    experiments = _build_experiments(args, run_dir)
    _write_json(run_dir / 'plan.json', [asdict(spec) | {
        'csv_path': str(spec.csv_path),
        'log_path': str(spec.log_path),
        'err_path': str(spec.err_path),
    } for spec in experiments])

    if args.dry_run:
        for spec in experiments:
            print(' '.join(spec.command))
        return

    env = os.environ.copy()
    if args.disable_check_env:
        env['LMDEPLOY_ENABLE_CHECK_ENV'] = '0'
    env['HF_HUB_OFFLINE'] = '1' if args.hf_offline else '0'
    env['TRANSFORMERS_OFFLINE'] = '1' if args.transformers_offline else '0'
    env['HF_MODULES_CACHE'] = args.hf_modules_cache
    Path(args.hf_modules_cache).mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    last_check_ts = time.time()
    summary_csv = run_dir / 'summary.csv'
    summary_json = run_dir / 'summary.json'
    checks_json = run_dir / 'checks.json'
    manifest_json = run_dir / 'manifest.json'

    for idx, spec in enumerate(experiments, start=1):
        record = _run_experiment(spec, env)
        records.append(record)
        _write_summary_csv(summary_csv, records)
        _write_json(summary_json, records)
        _save_manifest(manifest_json, experiments, records)

        elapsed_since_check = time.time() - last_check_ts
        should_check = (idx % args.check_every_experiments == 0) or (elapsed_since_check >= args.check_interval_seconds)
        if should_check:
            report = _evaluate_expectations(records, args)
            _write_json(checks_json, report)
            _print_check_summary(report)
            last_check_ts = time.time()
            if args.stop_on_fail and report['counts']['fail'] > 0:
                print('[STOP] expectation failure encountered, stopping early.')
                break

    final_report = _evaluate_expectations(records, args)
    _write_json(checks_json, final_report)
    _print_check_summary(final_report)
    print(f'[RESULTS] run_dir={run_dir}')


if __name__ == '__main__':
    main()
