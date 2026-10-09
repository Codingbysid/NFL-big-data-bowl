# The Cost of a Cut
## Does a Combine turn predict rookie separation?

> Working draft. The exported prediction results have been checked independently. Complete the source-code, measurement, and reproducibility checks in the companion review before submission.

Two receivers run similar forties. Can their movement through a Combine turn help a scout distinguish them? We tested the distance a receiver gains along the outgoing direction in the half-second after a turn, adjusted for entry speed, turn angle, and drill. The proposed relationship was simple: a better adjusted exit would predict more rookie-season separation.

This experiment did not establish that relationship. Adding the tracking trait reduced prediction error by 0.016 yards in the 2025 evaluation class, but the uncertainty interval included no improvement. The fitted association was negative, and the corresponding NFL movement measure showed little agreement with the Combine trait. Those findings limit the use of this particular measurement as a scouting grade. They do not settle whether other cut-related measurements are useful.

## The receivers and the outcome

The dataset contains 106 wide receivers across the 2023–25 classes. We studied regular-season routes in each player's rookie year. Eligible plays had a forward pass, a recorded separation label, no nullifying penalty, and a route in the specified analysis set. Screens and rare wheel/angle labels were excluded. Receivers needed at least 20 eligible routes for the primary evaluation.

| Draft class | Roster receivers | Eligible routes | Modeled receivers | Modeled routes |
| --- | ---: | ---: | ---: | ---: |
| 2023 | 38 | 6,280 | 28 | 6,279 |
| 2024 | 31 | 4,368 | 21 | 4,324 |
| 2025 | 37 | 3,605 | 19 | 3,552 |

*Table 1. The modeling sample after the route-count requirement. These are player-route observations, not independent prospects.*

The outcome is the supplied distance to the nearest defender when the pass is thrown. It is available on targeted and untargeted routes for completions, incompletions, and interceptions, and unavailable on sacks and scrambles. Missing outcomes were excluded rather than set to zero. Game tracking follows the prospect cohort, so we did not reconstruct separation from an incomplete set of defenders.

The sample describes receivers who played enough eligible routes. Of the 18 receivers excluded from the 2025 class by the route requirement, 13 were undrafted; only one of the 19 modeled receivers was undrafted. Results therefore cannot be extended directly to prospects who rarely played.

## Measuring the exit

For each detected turn, the reference time is the midpoint of heading change between the incoming and outgoing segments. Exit distance is the displacement over the following 0.5 seconds projected onto the outgoing direction. It measures movement along the route, not distance from a defender.

![A Combine speed-out trajectory and its half-second exit window.](../outputs/figures/fig1_exit_distance.png)

*Figure 1. An illustrative Combine speed-out from the supplied evidence. The turn reference and exit window define the measurement; the trajectory does not identify foot placement or joint mechanics.*

A training-fitted ridge model estimates expected exit distance from entry speed, turn angle, their interaction, and drill family. The observed-minus-expected difference is the cut-exit residual. The player trait is the median of qualifying attempt residuals; fewer than two attempts leaves it unavailable. The primary drill families were speed-out, slot-sail, dagger, post-corner, and qualifying slant attempts.

The protocol required a 30–110-degree turn, entry speed of at least four yards per second, and usable tracking around the reference time. The development extract contained 406 qualifying Combine cuts across 69 roster receivers before restriction to the performance-model sample.

The reported 2023 cross-drill consistency estimate was 0.18, with a bootstrap interval of 0.04–0.29. Most drill types were performed once, so this is not a test of repeating the same drill under the same conditions. The within-receiver residual standard deviation was 0.28 yards. It includes differences between drills and attempts and cannot establish a minimum detectable training improvement.

![Shared movement-demand cells in the 2023 development data.](../outputs/figures/fig2_support.png)

*Figure 2. The supplied 2023 support map used for the 2024 translation check. Cell labels give Combine/NFL cut counts. Supported cells cover entry speeds of 6–8 yards per second and turns of 30–90 degrees. Final training uses a separate 2023–24 support calculation.*

NFL cuts were measured only when the exit window finished before the throw and any recorded catch or contact. Approximately 10–11% of tracked forward passes yielded a qualifying cut. This is detector and window coverage, not an estimate of how often receivers change direction. An early throw relative to the break can leave insufficient time to measure the exit.

The two analyses use different populations: performance prediction covers the eligible route set, including go routes; the movement-translation check uses measurable cuts within shared demands. A null result on average route separation therefore does not rule out a more specific relationship on individual breaks.

## Testing additional information

The 2023 class supplied development and player-grouped out-of-fold predictions. The 2024 class provided a chronological development check. The recorded recipe was then fitted on 2023–24 and evaluated on 2025. The provided experiment log places protocol and model freezing before evaluation.

All three models include route family, coverage, cushion, down and distance, field position, formation, alignment, and motion. The conventional model adds body measurements, age, forty and ten-yard times, vertical, broad jump, three-cone, and shuttle. Missing conventional tests are imputed using training data with missingness indicators. The tracking model adds the cut-exit trait.

Training gives each receiver equal total weight across their routes. The primary score averages the absolute difference between each receiver's mean observed and mean predicted separation. These predictions condition on the routes and contexts actually observed; they are not unconditional pre-draft forecasts. NFL movement traits, receiving outcomes, and player names are not prospect predictors.

The models have independently selected ridge penalties. In the final fit, the conventional model uses 1.0 and the tracking model 0.3. Their comparison measures the difference between two trained pipelines. A player's prediction change also includes refitted conventional and context terms; it cannot be attributed entirely to the added trait.

## What the evaluation showed

For the 19 receivers in 2025, context alone produced player-mean absolute error of 0.187 yards. Conventional measurements reduced it to 0.148; adding tracking reduced it to 0.132.

The tracking-minus-conventional difference was −0.0158 yards: a 10.7% point reduction relative to conventional-model error. The supplied paired bootstrap interval was −0.0333 to +0.0022 yards, from 2,000 resamples of the 19 receivers. It includes no improvement and is conditional on the fitted pipelines. It does not incorporate every source of model-training or movement-measurement uncertainty.

![Performance across the three cohorts and uncertainty in the 2025 model comparison.](../outputs/figures/model_comparison_reviewed.png)

*Figure 3. Player-mean error reconstructed from the exported predictions. The right panel retains the original reported 2025 paired-bootstrap interval. Negative differences favor tracking. The three cohort results use their recorded development/evaluation procedures, not three independent final tests.*

The pattern across classes was inconsistent. Tracking increased error by 0.0344 yards in the 2023 out-of-fold results and reduced it by 0.0020 in the 2024 check. The analyst-chosen target was a 0.10-yard reduction in player-mean error. The 2025 estimate falls below that target, which is specific to this project rather than a general standard for useful scouting information.

Route-level errors tell a different story from player averages: context alone achieved 1.420 yards, conventional testing 1.428, and tracking 1.425 in 2025. Improvements in predicting player means did not translate into lower individual-route error than context alone.

The exported coefficient summary reports that one training standard deviation of the trait, about 0.20 yards, was associated with 0.038 yards less predicted separation. This conditional fitted association runs against the proposed direction; it does not imply that a receiver should aim for a slower exit.

The secondary Combine-to-NFL movement correlations were −0.31 for 21 receivers in 2024 and 0.08 for 20 in 2025. These reported estimates provide little support for consistent positive translation, although they describe a restricted movement sample.

Conventional testing also deserves a bounded conclusion. Its player-mean advantage over context alone was 0.006 yards in 2023, approximately zero in 2024, and 0.039 in 2025. On 2025 man-coverage routes, both Combine models performed worse than context alone. This experiment does not establish a universal advantage for either set of prospect measurements.

## What the player examples mean

The protocol selected Jaylin Noel and Jaylin Lane by the disagreement between their conventional and tracking projections within a forty-time band. Noel's observed mean separation was 3.224 yards. Conventional predictions averaged 3.195 on his routes, closer than tracking's 3.142. Lane averaged 3.180; tracking's 3.161 was closer than conventional's 3.099.

These contrasting examples show why a small average gain is insufficient for a scouting rule. Their prediction differences combine the trait term with other refitted coefficients. Dont'e Thornton, selected after evaluation as the largest tracking-model miss, averaged 2.541 yards against predictions near 2.91–2.94. The examples illustrate the models' limits rather than identify hidden stars or draft busts.

## A practical boundary for using the metric

A scout should not turn this residual into a receiver ranking on the present evidence. Its cross-drill consistency is limited, its NFL movement relationship is unstable, and the additional prediction result is uncertain across a small number of players.

A coach could use the geometric definition to design a standardized repeated-drill study: hold entry speed, turn angle, direction, and drill conditions comparable, collect repeated attempts, and estimate repeatability before setting a progress threshold. The present data do not support a rule that changes under 0.3 yards are noise, nor a claim that improving the number improves NFL separation.

This study evaluates one metric, one endpoint, and a selected rookie population. Nineteen evaluation receivers, limited comparable cuts, and the alignment of Combine year with draft class constrain the conclusion. The useful result is a boundary on interpretation: a measurable movement difference still needs reliable measurement, credible NFL translation, and additional decision value before becoming a scouting grade.
