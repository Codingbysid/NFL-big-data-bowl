# NFL Big Data Bowl 2027

Two studies live in this repository. They do not share a population or an endpoint.

## Defensive-line get-off (current writeup)

Among 2023–24 defensive ends, tackles, and nose tackles, does a 1-to-5-yard forty transit add anything to rookie `player_get_off` beyond the official 10-yard split? The study did not establish that it does. 2025 is not used.

- Writeup: `writeup/getoff.md`
- Protocol: `reports/getoff_v1_protocol.md`
- Results: `reports/getoff_v1_results.md`
- Notebook: `notebooks/getoff_submission.ipynb` recomputes the measurement, the tuned comparison, the shared-penalty diagnostic, and the figures. `notebooks/submission.ipynb` is the older receiver study and does not support this writeup.
- Paths: `BDB_CODE_ROOT` (source and `config/`, read-only), `BDB_DATA_ROOT` (the CSV directory containing `players.csv`, read-only), and `BDB_WORK_ROOT` (writable outputs). On Kaggle, leave the work path unset and the study writes to `/kaggle/working/getoff-study`.
- Kaggle setup: attach the competition data and upload this repository as a second dataset. Set `BDB_CODE_ROOT` to that dataset. Install `duckdb` and `pyyaml` in the notebook environment if they are absent. That Kaggle execution has not been run.

## Closed receiver-cut study

The earlier study asks whether a wide receiver's Combine cut-exit residual predicts rookie-season separation. That candidate is closed. Its 2025 evaluation remains the historical result.

The 2023 class is for development. 2024 is a chronological check. 2025 is evaluated once, after `freeze`. The draft is `writeup/draft.md`. The notebook for that study only is `notebooks/submission.ipynb`.

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
