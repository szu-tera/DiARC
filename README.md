<div align="center">

# DiARC: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models

[![Code](https://img.shields.io/badge/code-GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/szu-tera/DiARC)
[![Adapters](https://img.shields.io/badge/adapters-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/yyxdnmd/DiARC-adapters)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=for-the-badge)](LICENSE)

<p>
  <a href="#news"><b>News</b></a> |
  <a href="#overview"><b>Overview</b></a> |
  <a href="#getting-started"><b>Getting Started</b></a> |
  <a href="#preference-data-construction"><b>Data Construction</b></a> |
  <a href="#training"><b>Training</b></a> |
  <a href="#released-adapters"><b>Adapters</b></a> |
  <a href="#citation"><b>Citation</b></a>
</p>

</div>

## News

- **[2026/06]** We release the DiARC codebase and selected PEFT/LoRA adapters for three backbones across six ARC-style benchmarks.

## Overview

This repository contains the implementation for **DiARC**, a preference-learning
framework for improving ARC-style abstract reasoning in large language models.
DiARC constructs chosen/rejected output pairs for ARC-like tasks and trains LLMs
to distinguish correct outputs from plausible but rule-incorrect alternatives.

The code covers three parts of the experimental pipeline:

- Constructing preference data with output-level, DSL-level, and task-specific negative samples.
- Training LoRA adapters with Direct Preference Optimization (DPO).
- Evaluating ARC-specialized language models with ARC-oriented inference helpers.

The release includes the small ARC-style benchmark files used by the paper.
Large artifacts are intentionally not stored in this GitHub repository. Full
RE-ARC generated corpora, base model checkpoints, generated DPO JSONL files,
trained adapters, and raw experiment outputs should be placed under local
artifact directories when needed.

## Key Features

- **Preference learning for ARC-style reasoning.** DiARC uses chosen/rejected
  grid outputs rather than only fitting correct targets with SFT.
- **Multiple negative-construction levels.** The repository supports
  output-level visual transformations, DSL-level rule inversion, and
  task-specific rule editing.
- **Six ARC-style benchmarks.** Included loaders cover ARC-AGI-1, ARC-AGI-2,
  MiniARC, ConceptARC, 1D-ARC, and ARCcommunity.
- **Released adapters.** Selected DiARC LoRA adapters are hosted on Hugging Face
  and linked below.

## Project Structure

```text
DiARC/
├── src/diarc/
│   ├── arc_loader.py                      # ARC/ARC-like dataset loader
│   ├── negative_transforms.py             # Output-level negative transforms
│   ├── build_external_preferences.py      # Preference data for ARC-like benchmarks
│   ├── build_rearc_preferences.py         # Preference data from generated RE-ARC negatives
│   ├── build_dsl_motif_preferences.py     # DSL-level rule inversion preferences
│   ├── build_task_editing_preferences.py  # Task-specific edited-program preferences
│   ├── train_dpo_llama.py                 # DPO training for Llama-style checkpoints
│   ├── train_dpo_minitron.py              # DPO training for NeMo-Minitron checkpoints
│   ├── train_dpo_qwen.py                  # DPO training for Qwen checkpoints
│   └── evaluate_llama_arcagi1.py          # ARC-AGI-1 evaluation helper
├── data/
│   ├── ARC-AGI-1/
│   ├── ARC-AGI-2/
│   ├── Mini-ARC/
│   ├── ConceptARC/
│   ├── 1D-ARC/
│   └── arc-community/
├── scripts/                               # Example command-line wrappers
├── configs/                               # Example configuration files
├── docs/
│   ├── artifact_notes.md
│   └── environment.md
├── requirements.txt
└── README.md
```

## Getting Started

Clone the repository and install dependencies:

```bash
git clone https://github.com/szu-tera/DiARC.git
cd DiARC

conda create -n diarc python=3.12 -y
conda activate diarc
python -m pip install -U pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

The code was smoke-tested on Linux with NVIDIA L40 GPUs, CUDA-enabled PyTorch,
`torch==2.8.0`, `transformers==4.52.4`, `trl==0.9.6`, `peft==0.15.2`,
`bitsandbytes==0.48.1`, and `unsloth==2025.10.3`.

For a quick installation check:

```bash
python -m compileall -q src
python - <<'PY'
from diarc.arc_loader import ArcDataset
ds = ArcDataset.load_from_neoneye("data/Mini-ARC")
print(len(ds.keys), ds.keys[:2])
PY
```

See [docs/environment.md](docs/environment.md) for detailed environment,
GPU, artifact, and smoke-test instructions.

## Artifacts and Paths

The code uses repo-relative paths by default:

```text
data/       benchmark files and generated preference JSONL files
models/     local base model checkpoints
outputs/    trained adapters and evaluation outputs
re_arc_gen/ optional RE-ARC DSL/generator/verifier resources
```

You can override them with environment variables:

```bash
export DIARC_DATA_DIR=/path/to/data
export DIARC_MODEL_DIR=/path/to/models
export DIARC_OUTPUT_DIR=/path/to/outputs
export DIARC_RE_ARC_GEN_DIR=/path/to/re_arc_gen
```

Expected external assets are summarized in
[docs/artifact_notes.md](docs/artifact_notes.md).

## Preference Data Construction

DiARC uses three negative-construction families.

### Output-Level Visual Transformations

For ARC-like benchmark files:

```bash
python -m diarc.build_external_preferences \
  --dataset conceptarc \
  --dataset-root data/ConceptARC \
  --output-dir data/dpo_conceptarc_morphology \
  --transform-category morphology \
  --top-k 16 \
  --ranker auto
```

For 1D-ARC, the builder automatically uses 1D-safe transformations:

```bash
python -m diarc.build_external_preferences \
  --dataset 1d-arc \
  --dataset-root data/1D-ARC \
  --output-dir data/dpo_1d_arc_random \
  --transform-category random_perturb \
  --top-k 16 \
  --ranker grid
```

For generated RE-ARC samples:

```bash
python -m diarc.negative_transforms \
  --rearc-path data/re_arc \
  --output-dir data/dpo_arcagi1_output_morphology \
  --transform-category morphology
```

### DSL-Level Rule Inversion

```bash
python -m diarc.build_dsl_motif_preferences \
  --rearc-root re_arc_gen \
  --output data/dpo_arcagi1_dsl_motif/arc_dpo_data_all.jsonl \
  --rewrite all
```

### Task-Specific Rule Editing

Pass a module containing edited `verify_<task_id>` functions or edited
`generate_<task_id>` functions:

```bash
python -m diarc.build_task_editing_preferences \
  --rearc-root re_arc_gen \
  --edited-generator-module re_arc_gen/generators_task_specific_edits.py \
  --output data/dpo_arcagi1_task_editing/arc_dpo_data_all.jsonl
```

## Training

The training scripts expect `arc_dpo_data_all.jsonl` under
`$DIARC_DATA_DIR/<dataset_subdir>/`.

Llama-style checkpoints:

```bash
BASE_MODEL_PATH=models/Llama-3.2-3B-ReArc-merged \
CUDA_VISIBLE_DEVICES=0 python -m diarc.train_dpo_llama \
  --dataset-subdir dpo_conceptarc_morphology \
  --output-subdir llama3b-dpo-conceptarc-morphology
```

NeMo-Minitron checkpoints:

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

## Evaluation

The ARC-AGI-1 evaluation helper can be imported by small run scripts or called
from custom entry points. It supports base-model evaluation, DPO adapter
loading, test-time training, augmentation scoring, and DFS-style decoding
through environment variables.

Full evaluation requires local base checkpoints and, when applicable, matching
adapter directories.

## Released Adapters

The selected DiARC PEFT/LoRA adapters are hosted on Hugging Face:

```text
https://huggingface.co/yyxdnmd/DiARC-adapters
```

The adapter repository is organized by backbone and benchmark:

```text
yyxdnmd/DiARC-adapters
├── llama-3.2-3b/{arc-agi-1,arc-agi-2,miniarc,conceptarc,1d-arc,arccommunity}
├── minitron-8b/{arc-agi-1,arc-agi-2,miniarc,conceptarc,1d-arc,arccommunity}
└── qwen3-4b/{arc-agi-1,arc-agi-2,miniarc,conceptarc,1d-arc,arccommunity}
```

Each leaf directory contains a PEFT/LoRA adapter, adapter config, tokenizer
metadata when available, and a short README. The root `manifest.json` records
the selected negative-construction method and setting for each adapter.

## Notes

- This repository does not redistribute third-party base model weights or large
  generated training corpora.
- The included benchmark files are small ARC-style public benchmark files used
  by the paper.
- Some helper files are adapted from the ARChitects/Product-of-Experts ARC
  pipeline and retain their original license headers where applicable.

## Contact

Please open a GitHub issue for questions about the released code and artifacts.

## Citation

If you find this repository useful, please cite the paper once the official
BibTeX entry is available.
