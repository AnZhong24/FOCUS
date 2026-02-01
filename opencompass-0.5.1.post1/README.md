# OpenCompass (benchmarking)

This directory contains a vendored OpenCompass snapshot used to benchmark
SDAR/LLaDA2.0-mini model quality with either the HuggingFace/Transformers backend or
the LMDeploy PyTorch engine.

## Upstream

- OpenCompass: https://github.com/open-compass/opencompass

## Environment

Assumes LMDeploy is already installed in your current environment.

From `opencompass-0.5.1.post1/`:

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
cd human-eval && pip install -e .
```

If you see `ModuleNotFoundError: tree_sitter_languages`, install it (e.g.
`mamba install tree_sitter_languages`).

## Model configs

OpenCompass can run with either the HuggingFace/Transformers backend or the
LMDeploy backend. Adjust paths and runtime knobs in:

- HuggingFace/Transformers:
  - [`opencompass/configs/models/sdar.py`](opencompass/configs/models/sdar.py)
  - [`opencompass/configs/models/llada2.py`](opencompass/configs/models/llada2.py)
- LMDeploy:
  - [`opencompass/configs/models/sdar_lmdeploy.py`](opencompass/configs/models/sdar_lmdeploy.py)
  - [`opencompass/configs/models/llada2_lmdeploy.py`](opencompass/configs/models/llada2_lmdeploy.py)

## Run

From `opencompass-0.5.1.post1/`:

- `PYTHONPATH=$PWD` is redundant when you run with only 1 GPU.
- `--max-num-workers` controls how many jobs run in parallel; set it to the
  number of GPUs you want to use. `--max-num-workers 2` assumes 2 GPUs. On a
  single GPU, omit it (default: 1) or set `--max-num-workers 1`.

```bash
# HuggingFace/Transformers backend
python run.py --models sdar --datasets gsm8k_gen math500_gen humaneval_gen sanitized_mbpp_gen IFEval_gen > my_log_hf.log 2>&1

PYTHONPATH=$PWD python run.py --models sdar --datasets gsm8k_gen math500_gen humaneval_gen sanitized_mbpp_gen IFEval_gen --max-num-workers 2 > my_log_hf.log 2>&1


# LMDeploy backend
python run.py --models sdar_lmdeploy --datasets gsm8k_gen math500_gen humaneval_gen sanitized_mbpp_gen IFEval_gen > my_log1.log 2>&1

PYTHONPATH=$PWD python run.py --models sdar_lmdeploy --datasets gsm8k_gen math500_gen humaneval_gen sanitized_mbpp_gen IFEval_gen --max-num-workers 2 > my_log_hf.log 2>&1
```

Swap `--models sdar` / `--models sdar_lmdeploy` for `--models llada2` /
`--models llada2_lmdeploy` to benchmark LLaDA2.0-mini.
