# DiARC

This repository contains the anonymized implementation for **DiARC**, a
preference-learning framework for ARC-style abstract reasoning. The code covers
three pieces of the experimental pipeline:

1. Construct chosen/rejected ARC preference pairs.
2. Train LoRA adapters with DPO.
3. Evaluate ARC-specialized models.

The release includes the small ARC-style benchmark files used by the paper.
Large assets are intentionally not included: full RE-ARC generated corpora,
model checkpoints, LoRA adapters, generated DPO JSONL files, and raw experiment
outputs should be placed under local `data/`, `models/`, and `outputs/`
directories when needed.

## Layout

```text
src/diarc/
  arc_loader.py                    ARC/ARC-like dataset loader
  negative_transforms.py           Output-level negative transformations
  build_rearc_preferences.py       Preference data from pre-generated RE-ARC negatives
  build_external_preferences.py    Preference data from ARC-like benchmarks
  build_dsl_motif_preferences.py   DSL-level rule inversion preferences
  build_task_editing_preferences.py Task-specific edited-program preferences
  train_dpo_llama.py               DPO training for Llama-style checkpoints
  train_dpo_minitron.py            DPO training for NeMo-Minitron checkpoints
  train_dpo_qwen.py                DPO training for Qwen checkpoints
  evaluate_llama_arcagi1.py        ARC-AGI-1 direct/TTT evaluation helper
configs/
  dpo_config_example.yaml          Example DPO configuration
docs/
  artifact_notes.md                Expected external artifacts and paths
data/
  ARC-AGI-1/                       ARC-AGI-1 public JSON files
  ARC-AGI-2/                       ARC-AGI-2 public JSON files
  Mini-ARC/                        MiniARC benchmark files
  ConceptARC/                      ConceptARC benchmark files
  1D-ARC/                          1D-ARC benchmark files
  arc-community/                   ARCcommunity benchmark files
```

## Setup

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

The code defaults to offline local checkpoints. Override paths as needed:

```bash
export DIARC_DATA_DIR=/path/to/data
export DIARC_MODEL_DIR=/path/to/models
export DIARC_OUTPUT_DIR=/path/to/outputs
export DIARC_RE_ARC_GEN_DIR=/path/to/re_arc_gen
```

## Build Preference Data

DiARC uses three negative-construction families.

For output-level visual transformations on ARC-like benchmark files:

```bash
python3 -m diarc.build_external_preferences \
  --dataset conceptarc \
  --dataset-root data/ConceptARC \
  --output-dir data/dpo_conceptarc_morphology \
  --transform-category morphology \
  --top-k 16 \
  --ranker auto
```

For output-level visual transformations on generated RE-ARC samples:

```bash
python3 -m diarc.negative_transforms \
  --rearc-path data/re_arc \
  --output-dir data/dpo_arcagi1_output_morphology \
  --transform-category morphology
```

For DSL-level rule inversion on ARC-AGI-1 RE-ARC programs:

```bash
python3 -m diarc.build_dsl_motif_preferences \
  --rearc-root re_arc_gen \
  --output data/dpo_arcagi1_dsl_motif/arc_dpo_data_all.jsonl \
  --rewrite all
```

For task-specific edited ARC programs, pass a module containing edited
`verify_<task_id>` functions or edited `generate_<task_id>` functions:

```bash
python3 -m diarc.build_task_editing_preferences \
  --rearc-root re_arc_gen \
  --edited-generator-module re_arc_gen/generators_task_specific_edits.py \
  --output data/dpo_arcagi1_task_editing/arc_dpo_data_all.jsonl
```

## Train DPO

The training scripts expect `arc_dpo_data_all.jsonl` under
`$DIARC_DATA_DIR/<dataset_subdir>/`.

```bash
BASE_MODEL_PATH=models/Llama-3.2-3B-ReArc-merged \
python3 -m diarc.train_dpo_llama \
  --dataset-subdir dpo_conceptarc_morphology \
  --output-subdir llama3b-dpo-conceptarc-morphology
```

For NeMo-Minitron and Qwen checkpoints, use `diarc.train_dpo_minitron` and
`diarc.train_dpo_qwen` respectively.

## Evaluation

The ARC-AGI-1 evaluation helper can be imported by small run scripts or called
from custom entry points. It supports base, DPO adapter, TTT, augmentation
scoring, and DFS-style decoding through environment variables.

## Notes

- The repository does not redistribute third-party model weights or generated
  checkpoints.
- The released code uses repo-relative paths and environment variables; no
  machine-local paths are required.
- Some helper files are adapted from the ARChitects/Product-of-Experts ARC
  pipeline and retain their original license headers where applicable.
