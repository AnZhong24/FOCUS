# FOCUS

FOCUS is implemented on top of LMDeploy's PyTorch engine. It adds token-importance eviction for DLLM decoding (used by SDAR and LLaDA2) and builds on LMDeploy's delayed KV cache flow.

## Architecture overview

![FOCUS architecture overview](FOCUS.svg)

See [Figure5.pdf](Figure5.pdf) for the PDF version.

## Key modified files (FOCUS)

From inspecting this repo and the git diff, the main FOCUS-related changes are in:

- `lmdeploy/pytorch/kernels/cuda/focus.py`: Triton kernels for importance scoring, target selection, and state compaction.
- `lmdeploy/pytorch/models/sdar.py`: FOCUS eviction integrated into SDAR layers.
- `lmdeploy/pytorch/models/llada2.py`: FOCUS eviction integrated into LLaDA2 layers.
- `lmdeploy/pytorch/strategies/dllm/sequence.py`: FocusState tracking and per-step statistics.
- `lmdeploy/pytorch/strategies/dllm/model_inputs.py`: focus-specific inputs for DLLM batches.
- `lmdeploy/pytorch/model_inputs.py`: focus runtime view and host/device synchronization.
- `lmdeploy/pytorch/engine/inputs_maker.py`: builds focus masks and pinned buffers for delayed cache batches.
- `lmdeploy/pytorch/engine/model_agent.py`: propagates processed positions back to the scheduler.
- `tests/pytorch/kernel/test_focus_kernels.py` and `tests/pytorch/models/test_focus_pruning.py`: FOCUS test coverage.
- `benchmark/run_focus_throughput_evaluation.sh`, `benchmark/run_block_size_comparison.sh`, and `benchmark/run_sdar_delayed_cache_benchmark.sh`: benchmarking entry points.

## Install (CUDA)

FOCUS relies on Triton CUDA kernels and is intended for CUDA GPUs.
LMDeploy's default prebuilt wheels target CUDA 12 (since v0.3.0); RTX 50-series GPUs require CUDA 12.8 wheels.
CUDA 11+ is supported when building from source, but ensure your local CUDA toolkit matches your PyTorch/Triton stack.

1. Create and activate a Python environment.
2. Install runtime dependencies:

```bash
pip install -r requirements/runtime_cuda.txt
```

3. Install the repo. Use the PyTorch engine only (skip TurboMind build):

```bash
DISABLE_TURBOMIND=1 pip install -e .
```

## Benchmarking scripts (./benchmark)

All scripts write logs to `./results`. Run them from the repo root.

- Focus vs base throughput:

```bash
benchmark/run_focus_throughput_evaluation.sh <dataset_id> <model_id>
```

Example:

```bash
benchmark/run_focus_throughput_evaluation.sh anon8231489123/ShareGPT_Vicuna_unfiltered JetLM/SDAR-8B-Chat-b32
```

Notes:
- For `hendrycks-MATH` datasets, the script automatically sets `--dataset-format math`.
- Focus uses `--dllm-enable-delayed-cache` and `--dllm-enable-focus` with `--dllm-focus-alpha 1.5`.
- Base uses `--dllm-confidence-threshold 0.9` without focus.

- Block size comparison for SDAR:

```bash
benchmark/run_block_size_comparison.sh <dataset_id>
```

This runs SDAR models `JetLM/SDAR-8B-Chat-b16` and `JetLM/SDAR-8B-Chat-b64` for both Focus and Base settings.

- Delayed cache baseline for SDAR:

```bash
benchmark/run_sdar_delayed_cache_benchmark.sh <dataset_id>
```

This runs `JetLM/SDAR-8B-Chat-b32` with delayed cache enabled and FOCUS disabled.

Dataset notes:
- `dataset_id` can be a HuggingFace dataset ID or a local JSON/JSONL path supported by `benchmark/profile_throughput.py`.
- HuggingFace dataset IDs require the `datasets` package and network access.
