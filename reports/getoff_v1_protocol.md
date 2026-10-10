# Get-off protocol, version 1

Date: 2026-10-10. Status: committed before any Combine-to-NFL association or model fit. This does not reopen the closed cut study and does not use the 2025 class.

## Hypothesis

Among defensive linemen, a longer sensor-derived early sprint transit time is associated with a longer rookie NFL get-off time, conditional on the official 10-yard split and the declared context. The official `ten_yd_split` is the conventional baseline. The added feature comes only from `combine_tracking.csv`.

A longer transit is a slower early movement after the player has already left the start. It is not reaction time, and it is not a replacement for the official split.

## Prior exposure

A pooled min, median, and max of `player_get_off` was computed once across all seasons while checking units, then discarded. The 2023–24 range was recomputed without 2025. No player-level join of get-off to a Combine time has been made. Sample size and outcome exposure are separate: 17 matched 2025 linemen fails the existing evaluation admission rule of 20. That headcount does not, by itself, make the 2025 cohort contaminated. This protocol does not lower that rule, and it does not open 2025.

## Population

DE, DT, and NT prospects. Rookie regular season means `games.season` equals `players.draft_year` and `season_type` is REG. Plays are not dropped because the result was a sack or a scramble. A missing `player_get_off` is unavailable, not a zero. The primary analysis keeps players with at least eight non-missing get-off plays. Every primary model uses that same player set after the tracking-quality rules below.

## Origin rule

This rule is fixed from forty geometry, before any get-off association.

The forty's long axis is the coordinate with the larger range. A usable attempt spans 35 to 50 yards on that axis. The running direction is the sign of the displacement from the origin to the last frame.

The origin is not the first timestamp and not a global clock offset. It is the median position of the opening stationary prefix: frames from the start of the attempt with speed below 2 yards/second. At least two such frames are required. Fewer than two makes the start ambiguous and the attempt missing.

Longitudinal distance is the signed distance from that origin along the running direction. `early_transit_1_5_s` is the interpolated time at 5 yards minus the interpolated time at 1 yard. Each crossing needs a sample at or before the yard line and a later sample at or beyond it. The two bracketing times may not be more than 0.15 seconds apart. A step implying speed above 13 yards/second invalidates the attempt. Duplicate timestamps keep the first. A transit outside 0.15 to 1.50 seconds is missing.

The player feature is the median of valid attempts. The fastest attempt is not chosen. No constant is subtracted to force tracking times onto the official split. The official split is not assigned to a particular tracked attempt.

The only predeclared sensitivity moves the origin 0.25 yards forward and 0.25 yards backward along the run and recomputes the player median. It is not a search over distances.

## Coverage stop

Stop before models if fewer than 80 percent of 2023–24 defensive-line forty attempts have a verified origin and both crossings, or if fewer than 30 players remain with an official split, a valid transit median, and at least eight get-off plays. Do not replace the feature with the official split and continue.

## NFL endpoint and context

The outcome is the player's median `player_get_off`, in seconds. It mixes movement, snap anticipation, stance, and the plays he actually faced. It is not pure acceleration. Sacks, pressure, and EPA are not endpoints.

Starting depth uses the snap frame only. Field x runs along the length of the field, 0 to 120. If the offense moves right, the line of scrimmage is `10 + yards_from_own_goal`; if left, `110 - yards_from_own_goal`, where own-goal yards are `yardline_number` when `yardline_side` is the possession team and `100 - yardline_number` otherwise. Defensive depth is `x - los` when the play goes right and `los - x` when it goes left. A snap depth outside 0 to 5 yards is missing for that play. The player value is the median of valid snaps. Later frames are not used to define this context. The ball and the other team are not required.

The adjustment set is frozen as combine weight, the share of rookie snaps lined up as EDGE rather than INTERIOR_LINE, and median starting depth. Missing context is imputed from the training players only, with a missingness indicator.

These are predictions conditional on observed rookie deployment, not pre-draft forecasts.

## Models

One row per player. Equal weight.

- M0: weight, edge share, median starting depth.
- M1: M0 plus official `ten_yd_split`.
- M2: M1 plus `early_transit_1_5_s`.

Ridge penalties are 0.3, 1, 3, and 10. Continuous predictors are standardized on the training players. The intercept is not penalized.

Primary comparison: choose each model's penalty by leave-one-player-out error on 2023, refit on all 2023 players, and score 2024. The paired bootstrap of 2024 absolute errors resamples players and does not refit; it is conditional on those fitted predictions.

Secondary comparison: nested leave-one-player-out on the pooled 2023–24 players, with the penalty chosen inside each training subset. Both results are reported. The more favorable one is not promoted.

No improvement threshold is declared. The 2025 admission rule of 20 players stays where it is and is not used to discard this development sample.
