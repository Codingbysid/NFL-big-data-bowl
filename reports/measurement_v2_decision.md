# Measurement decision: no-go

Candidate: Combine SPEED_OUT and DAGGER, NFL OUT and IN, 2023–24 only. Protocol: `reports/measurement_v2_protocol.md`, committed before this run. 2025 separation was not read. Stage C was not run.

## What was measured

The bounded detector kept an attempt only when it found exactly one pre-boundary cut.

| Quantity | Count |
| --- | ---: |
| Combine single cuts | 117 (69 players) |
| Combine attempts with two or more cuts, abstained | 38 |
| Combine attempts with no cut | 1 |
| NFL OUT/IN single cuts | 362 |
| NFL attempts with two or more cuts, abstained | 22 |
| NFL attempts with no in-boundary cut | 1,949 |
| NFL attempts missing snap or pass | 5 |
| Accepted NFL cuts using a post-boundary frame | 0 |
| 2023 support cells occupied on both sides | 2 (6–8 yd/s with 50–70° and 70–90°) |
| Players with at least two supported Combine cuts and one supported NFL cut | 15 |
| 2024 players with at least one supported cut on each side | 16 |

About 362 of 2,338 in-scope NFL routes (15 percent) produced one bounded cut. Support then kept 118 of those 362. The player intersection is 15, below the predeclared minimum of 20. Fourteen player-family pairs had two or more attempts of the same drill. That is not enough to estimate a same-condition repeatability threshold, and none is stated.

Exit-prediction cross-validation chose lambda 0. Lambda 3, the historical setting, had higher Combine exit error (0.063 versus 0.035 weighted yards squared). Out-of-fold SPEED_OUT and DAGGER mean residuals were about +0.01 yards, with intervals covering zero.

## Why the candidate stops

Two gates fail, and either one is enough.

Human agreement is pending. The validation gallery has 30 accepted Combine plots, 30 accepted NFL plots, and 10 rejected NFL plots (`outputs/measurement_v2/annotation_template.csv`). The validation split contained no Combine rejection, because only one Combine attempt in this scope failed to yield a cut. Labels are blank. The 90 percent rule cannot be scored. Filling it from the detector marker would not be an independent review.

The supported player sample is also below the rule written before this run: 15 players with both traits, against a minimum of 20. A performance model on cut-bearing routes was therefore not fit. No Combine-to-NFL residual correlation was computed for this candidate.

## What an evaluator can decide

Do not rank receivers by this residual, and do not treat SPEED_OUT/DAGGER exit distance as a validated NFL trait. The historical broad-route result is unchanged and still inconclusive. The next research step, if anyone continues, is a human labeling of the existing validation gallery against the separate detector key. Do not retune the detector on that gallery, and do not open 2025 separation until both the agreement gate and the player-count gate have been met on 2023–24.
