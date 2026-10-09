# Result summary

The cut-exit residual does not clear the bar set before the 2025 evaluation. On 19 held-out receivers the tracking model reduced player-mean absolute error by 0.016 yards relative to conventional Combine tests. The paired bootstrap interval is −0.033 to +0.002 and includes zero. The pre-set practical bar was 0.10 yards. The fitted coefficient is negative: a larger residual predicted slightly less separation, which is the opposite of the hypothesis.

## Holdout ladder (2025, 19 receivers, 3,552 routes)

| Model | Player-mean MAE (yards) | Route MAE (yards) | Rank correlation |
| --- | --- | --- | --- |
| M0 context | 0.187 | 1.420 | 0.816 |
| M1 conventional | 0.148 | 1.428 | 0.856 |
| M2 plus cut-exit residual | 0.132 | 1.425 | 0.884 |

M2 minus M1 = −0.016 yards. Bootstrap of frozen predictions, 2,000 player resamples: −0.033 to +0.002.

## Development checks, computed before the holdout

- 2023 out-of-fold: M2 minus M1 = +0.034 yards (tracking increased error) on 28 receivers.
- 2024 chronological check: M2 minus M1 = −0.002 yards on 21 receivers. Refit bootstrap −0.018 to +0.011.
- Reliability of the 2023 attempt residual: ICC 0.18 (0.04 to 0.29). Within-receiver SD 0.28 yards; between-receiver SD 0.18 yards.
- Combine trait versus forty, 2023: Spearman −0.08 (n = 24).
- Combine trait versus the NFL trait: Spearman −0.31 on 21 receivers in 2024, and +0.08 on 20 receivers in 2025.
- Frozen coefficient on the standardized trait: −0.038 yards of separation per standard deviation (training SD 0.20 yards).

## Profiles

Selected before 2025 outcomes, inside a ±0.15 second forty band around the training median of 4.46 seconds (16 of 19 modeled receivers):

- Jaylin Noel, residual +0.26 yards (3 attempts). Marginal tracking projection was 0.078 yards below the conventional projection. Observed mean separation 3.22; conventional route prediction 3.20; tracking prediction 3.14.
- Jaylin Lane, residual −0.40 yards (3 attempts). Marginal tracking projection was 0.055 yards above conventional, because the fitted sign treats a poor exit as a plus. Observed 3.18; conventional 3.10; tracking 3.16.

Post-hoc largest miss: Dont'e Thornton, observed 2.54 against predictions near 2.91–2.94. His residual rests on two attempts.

## What the numbers support

Keep conventional testing as the Combine baseline that actually moved error, from 0.187 to 0.148. Do not rank receivers by cut-exit residual. A coach can still record exit yards on breaks at 6–8 yards per second and 30–90 degrees, and can treat a change under about 0.3 yards as measurement noise. That drill record is not evidence that changing the drill will change separation.

## Reproduction

Development modeling took 86 seconds. Freezing the 2023–24 fit and evaluating 2025 took 46 seconds, including 11 seconds to read the 2025 tracking file. Peak resident memory on the extracts was about 0.6 GB. Commands are in `README.md`. Figure labels were cleaned after the holdout file was written; the models were not refit. Re-running `evaluate-heldout` will stop on the code-hash check until `freeze` is repeated.
