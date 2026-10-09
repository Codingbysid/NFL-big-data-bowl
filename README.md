# The Cost of a Cut

Research pipeline for the NFL Big Data Bowl 2027. It asks whether a wide receiver's Combine cut-exit residual — yards gained along the outgoing direction in the half-second after a turn, adjusted for entry speed, turn angle, and drill — predicts rookie-season separation beyond conventional Combine tests.

The 2023 class is for development. 2024 is a chronological check. 2025 is evaluated once, after `freeze`.

## Data

Raw files stay in `nfl-big-data-bowl-2027/` (also linked from `data/raw/`). They are not modified and are gitignored. A Kaggle mount at `/kaggle/input/nfl-big-data-bowl-2027` is used when that path exists.

## Setup

```bash
python3.13 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest
```

## Commands

```bash
.venv/bin/python -m bdb2027 audit --data-root nfl-big-data-bowl-2027
.venv/bin/python -m bdb2027 extract --cohorts 2023 2024
.venv/bin/python -m bdb2027 review-gallery
.venv/bin/python -m bdb2027 protocol
.venv/bin/python -m bdb2027 develop
.venv/bin/python -m bdb2027 freeze
.venv/bin/python -m bdb2027 evaluate-heldout --cohort 2025
.venv/bin/python -m bdb2027 report
```

`evaluate-heldout` refuses to run if the configuration or code hash does not match `reports/freeze_manifest.json`. A smoke subset, which is not a reported result, is `python -m bdb2027 smoke`.

## Outputs

Audits and the protocol are in `reports/`. Metrics, player features, predictions, and figures are in `outputs/` after `report`. The notebook is `notebooks/submission.ipynb`. The draft writeup is `writeup/draft.md`.
