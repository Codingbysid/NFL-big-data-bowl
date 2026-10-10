# Get-off development result

The closed cut study is unchanged. This is a 2023–24 development analysis for defensive linemen. 2025 get-off was not joined to any Combine measurement.

## Cohort

Forty-two DE, DT, and NT prospects had a rookie regular-season play in 2023 or 2024. Forty had at least eight recorded get-off times. Three more lacked an official 10-yard split. None of the remaining players lacked a valid 1-to-5-yard transit. The primary set is 37 players, 18 from 2023 and 19 from 2024, and every model uses that set. Get-off is missing on many plays (for example 1,893 of 2,649 DE plays in 2023) and those plays are unused, not zeros. Sacks and scrambles were not removed.

All 94 defensive-line forty attempts had a stationary prefix and both crossings. The player feature is the median transit, about 0.76 seconds. Moving the origin 0.25 yards changes that time by a median of 0.032 seconds.

## Measurement

Forty-four players have two valid attempts. The second attempt is 0.007 seconds slower on average, with a paired SD of 0.035 seconds. The cross-attempt ICC is 0.59 (within-player SD 0.025 seconds; latent between-player SD 0.030 seconds). That reliability belongs to this sensor transit only. It says nothing about the official split, which is a different clock: the tracking time to 10 yards was already known to sit above the official split, and no offset was applied.

## Prediction

On 2024, after fitting on 2023:

| Model | Mean absolute error |
| --- | ---: |
| M0, weight, edge share, and snap depth | 0.058 seconds |
| M1, M0 plus official 10-yard split | 0.067 seconds |
| M2, M1 plus 1-to-5-yard transit | 0.061 seconds |

M2 minus M1 is −0.0068 seconds. The player bootstrap interval, conditional on the fitted 2024 predictions, is −0.0135 to +0.0005 seconds. It includes no improvement. Nested leave-one-player-out on the pooled 37 players ranks M0 best (0.065), then M1 (0.070), then M2 (0.072). The tracking term does not add information once both comparisons are shown.

That 0.0068-second gap is not the sensor’s own contribution. M1 was tuned at penalty 1 and M2 at penalty 3. A post-evaluation diagnostic, not a replacement result, refits M1 at penalty 3, the penalty already chosen for M2 on 2023. At that common penalty, 2024 error is 0.0594 seconds without the transit and 0.0607 seconds with it. Adding tracking slightly increases error. The original comparison of independently tuned models remains the historical development result.

The 2023 standardized transit coefficient is +0.004 seconds of get-off per training standard deviation of transit. A bootstrap that refits 2023 at the frozen penalty, without retuning it, runs from −0.034 to +0.040. The point estimate points in the hypothesized direction and is too small, and too uncertain, to use.

The raw official split has a positive rank correlation with median get-off: 0.35 in 2023 (18 players), 0.59 in 2024 (19), and 0.45 pooled. That description does not survive the chronological model: adding the split to context increased 2024 error. Leaving out any one 2024 player leaves the M2−M1 difference between about −0.008 and −0.005 seconds. Interior linemen show more of that small gap than edge players. Neither slice is a result to act on, and the nose-tackle slice is one player.

## Decision

The sensor transit is reasonably repeatable. It did not establish useful incremental prediction of rookie get-off beyond the official split. The official split’s own rank association with get-off is real as a description and did not improve the 2024 prediction once context was included. Neither result means Combine testing does not matter, and neither means that a player’s deployment caused his get-off. There is no scouting grade here. The 2025 class remains closed. No further feature was tried.
