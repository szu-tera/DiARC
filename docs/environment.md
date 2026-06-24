# Environment Setup

This project is designed to run from a local clone with benchmark data in
`data/`, optional base checkpoints in `models/`, generated artifacts in
`outputs/`, and optional RE-ARC generator resources in `re_arc_gen/`.

## Recommended Python Environment

The released code was smoke-tested on Linux with NVIDIA L40 GPUs and this
software stack:

```text
python==3.12
torch==2.8.0
transformers==4.52.4
trl==0.9.6
peft==0.15.2
datasets==3.6.0
accelerate==1.7.0
bitsandbytes==0.48.1
unsloth==2025.10.3
numpy==1.26.4
scipy==1.16.2
scikit-learn==1.7.2
pillow==11.3.0
```

Create an environment and install dependencies:

```bash
conda create -n diarc python=3.12 -y
conda activate diarc
python -m pip install -U pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

If your CUDA or driver setup requires a specific PyTorch wheel, install that
wheel first from the official PyTorch index, then install the remaining
requirements.

## GPU Check

Check available GPUs:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

Run a small CUDA check on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
x = torch.ones((1024, 1024), device="cuda")
print(float(x.sum()))
PY
```

## Local Paths

The code uses repo-relative defaults:

```text
data/       benchmark files and generated preference JSONL files
models/     local base checkpoints
outputs/    trained adapters and evaluation outputs
re_arc_gen/ optional RE-ARC DSL/generator/verifier resources
```

Override them with environment variables:

```bash
export DIARC_DATA_DIR=/path/to/data
export DIARC_MODEL_DIR=/path/to/models
export DIARC_OUTPUT_DIR=/path/to/outputs
export DIARC_RE_ARC_GEN_DIR=/path/to/re_arc_gen
```

The training scripts run in offline mode by default and expect local model
checkpoints. You can either place checkpoints under `models/` or pass
`BASE_MODEL_PATH` / `--base-model-path`.

## Required External Assets

The repository includes small benchmark JSON files for ARC-AGI-1, ARC-AGI-2,
MiniARC, ConceptARC, 1D-ARC, and ARCcommunity.

It does not include:

- Third-party base model weights.
- Full RE-ARC generated corpora.
- Generated preference JSONL files.
- Paper experiment outputs.

For ARC-AGI-1 rule-level construction, `re_arc_gen/` should contain compatible
`dsl.py`, `generators.py`, and `verifiers.py` files. Task-specific editing also
requires a user-provided edited verifier or generator module.

## Smoke Tests

Run syntax and import checks:

```bash
python -m compileall -q src
python - <<'PY'
import importlib
for name in [
    "diarc.arc_loader",
    "diarc.negative_transforms",
    "diarc.build_external_preferences",
    "diarc.train_dpo_llama",
    "diarc.train_dpo_minitron",
    "diarc.train_dpo_qwen",
]:
    importlib.import_module(name)
    print("OK", name)
PY
```

Build a tiny preference dataset from included files:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diarc.build_external_preferences \
  --dataset miniarc \
  --dataset-root data/Mini-ARC \
  --output-dir /tmp/diarc_smoke/dpo_miniarc_grid_block \
  --transform-category grid_block \
  --top-k 2 \
  --max-tasks 2 \
  --no-augment \
  --ranker grid
```

For 1D-ARC, use the same builder with 1D-safe transforms:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diarc.build_external_preferences \
  --dataset 1d-arc \
  --dataset-root data/1D-ARC \
  --output-dir /tmp/diarc_smoke/dpo_1d_arc_random \
  --transform-category random_perturb \
  --top-k 2 \
  --max-tasks 2 \
  --no-augment \
  --ranker grid
```

## Training

Training expects `arc_dpo_data_all.jsonl` under
`$DIARC_DATA_DIR/<dataset_subdir>/`.

Llama-style checkpoints:

```bash
BASE_MODEL_PATH=models/Llama-3.2-3B-ReArc-merged \
CUDA_VISIBLE_DEVICES=0 python -m diarc.train_dpo_llama \
  --dataset-subdir dpo_conceptarc_morphology \
  --output-subdir llama3b-dpo-conceptarc-morphology
```

Minitron checkpoints:

```bash
BASE_MODEL_PATH=models/Mistral-NeMo-Minitron-8B-ARChitects-ReArc1200-bnb-4bit \
CUDA_VISIBLE_DEVICES=0 python -m diarc.train_dpo_minitron \
  --dataset-subdir dpo_conceptarc_morphology \
  --output-subdir minitron-dpo-conceptarc-morphology
```

Qwen checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diarc.train_dpo_qwen \
  --base-model-path models/qwen3_4b_grids15_sft139_bfloat16 \
  --dataset-subdir dpo_conceptarc_morphology \
  --output-subdir qwen3-dpo-conceptarc-morphology
```

## Released Adapters

The selected PEFT/LoRA adapters are hosted separately:

```text
https://huggingface.co/yyxdnmd/DiARC-adapters
```

Download the needed adapter directory from Hugging Face and combine it with the
matching local base checkpoint for evaluation or further training.
