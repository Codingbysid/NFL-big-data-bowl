# Source-review resolution

Pinned baseline: `a9ea1b4d2c9870c43083553d3a622abda2f87d9a`. The archive hashes in `reports/archive_hashes_before.json` were recorded before these edits. The historical detector in `src/bdb2027/cuts.py` is unchanged. The bounded detector is `src/bdb2027/cuts_v2.py`.

## 1. NFL cuts could use frames after the throw

Evidence. `detect_cuts` builds kinematics on the full series, then keeps a cut when `t0 + 0.5` ends before the pass. It does not stop the outgoing heading window, and Savitzky–Golay uses neighboring frames. The regression test `test_historical_detector_changes_when_only_post_pass_heading_changes` reproduces the diagnostic: a pass at 1.6 seconds yields one accepted cut if the path continues at 90 degrees and none if it continues at 30 degrees. A second synthetic turn is accepted with the outgoing window ending at 1.7 seconds while the pass is at 1.6 and the exit ends at 1.5.

Correction. `detect_cuts_bounded` drops every coordinate after the earlier of the pass and the contact/catch boundary before smoothing or window search. The same two post-pass headings now produce the same result, and the crossing-window example is rejected. Each accepted row stores `latest_raw_timestamp`. On the 2023–24 OUT/IN reprocess, accepted NFL cuts had zero boundary violations (`outputs/measurement_v2/summary.json`).

Tests. `tests/test_measurement_v2.py`: post-pass invariance, earlier contact, missing pass, and the crossing window. Geometry tests still pass.

Remaining uncertainty. This does not, by itself, change the archived −0.0158-yard result. The primary model does not use NFL cut traits as predictors. The size of the change in real NFL exit distances is the difference between the old and bounded detectors, not a separation effect.

## 2. Historical translation was not restricted to common support

Evidence. In `pipeline.py::develop`, `translation_stats` is computed before `support_mask`, and the masks are discarded. `freeze` and `evaluate_heldout` likewise correlate traits without applying the frozen cells. The reported −0.31 (2024, 21 players) and +0.08 (2025, 20 players) are unrestricted comparisons. The later endpoint diagnostic, which did filter support, is a different calculation.

Correction. Those numbers keep their values and lose the “common support” description. The versioned study fits cells on 2023 only and applies them to both sides before any player aggregate (`support_v2.py`). Demand is fit on all eligible in-scope cuts, not only supported cells. Unsupported cuts are omitted, not set to zero. No new 2025 correlation was computed.

Tests. Evaluation rows do not change a frozen support table. An unsupported cut is dropped before the attempt mean rather than entered as zero.

Remaining uncertainty. The historical correlations are still descriptive of the players who had any kept cut, not of a shared speed-angle cell.

## 3. The demand penalty was large on the weight scale

Evidence. Attempt weights sum to 1, and the historical demand model uses Ridge `alpha=3`. The normalized objective `sum(w r^2)/sum(w) + lambda ||beta||^2` matches that alpha only when `alpha = lambda * sum(w)`. With weights summing to 1, lambda was 3. On this candidate’s Combine cuts, grouped exit error was 0.035 yards squared at lambda 0 and 0.063 at lambda 3 (`outputs/measurement_v2/summary.json`).

Correction. `demand_v2.py` selects lambda from `{0, 0.01, 0.1, 1, 3}` by player-grouped Combine exit error only. The chosen value on SPEED_OUT and DAGGER cuts was 0. Out-of-fold family means were about +0.01 yards, and the player-resample intervals covered zero (`calibration.csv`). The fastest and slowest speed bands still show means near +0.09 to +0.10 yards, on 10 and 5 players, without a full interval.

Tests. Rescaling weights and the corresponding Ridge alpha together leaves coefficients unchanged.

Remaining uncertainty. Lambda 0 is the exit-prediction choice for this two-family sample. It was not chosen to move a separation correlation, and it is not a claim that the historical all-family model should be refit.

## 4. The review sheet does not validate the frozen detector

Evidence, from `reports/review_sheet.csv` and `historical_review_reconciliation.json`. Four detected Combine marks and eight detected NFL marks have entry speed below 4 yards/second. The four individually reviewed Combine fails are SLOT_STRIKE or EXCLUDED. Among Combine rows in a primary family with speed at least 4, the sheet has 15 pass and 1 ambiguous. Nine NFL rows are labeled pass, and one of those has `detected=0`. There are no gold timestamps or code hashes on the sheet. Replaying the gallery seed joined 46 of 64 rows; three of those attempts contain more than one cut, while the old plot showed only the largest angle.

Correction. The old sheet and gallery stay as historical evidence. Nothing was deleted. The versioned sample stores trajectory id, cut id, code and config hashes, window bounds, and rejection reasons. Annotation plots omit the detector marker. The label file is empty. Scoring returns `pending_human` until a person fills intervals. That is not a 0 percent agreement result.

Tests. A blank template is pending, not a disagreement.

Remaining uncertainty. Final-detector precision is still unknown. An analyst who can see the marker did not fill the sheet.

## 5. The notebook could not reproduce from an empty work directory

Evidence. The previous full-run cell skipped `protocol`, did not pass `--data-root` or `PYTHONPATH` into the child, and then read `outputs/predictions.csv` from the repository.

Correction. `notebooks/submission.ipynb` now creates a fresh work directory, passes `--data-root` before the subcommand, sets `PYTHONPATH` for the child, runs protocol before develop, runs report, and joins the generated holdout predictions to `archive/original_evaluation/predictions.csv` with a `1e-8` tolerance. `Project` reads `BDB_DATA_ROOT`. Kaggle was not run.

Tests. `BDB_EXECUTE=1 jupyter nbconvert --execute notebooks/submission.ipynb` completed locally. The fresh holdout matched the archive within `1e-8`: maximum absolute differences were about `4e-15` for `pred_M0`, `pred_M1`, and `pred_M2`, and 0 for `y`. The archived delta is −0.015807633966148427 and the reproduced delta is −0.015807633966148288. Kaggle was not run. A 2025 `extract` command is refused unless it is `evaluate-heldout`; the notebook uses that command.
