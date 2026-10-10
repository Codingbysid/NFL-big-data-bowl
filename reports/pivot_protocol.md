# Pivot protocol: which movement question the files can support

Date: 2026-10-09. The SPEED_OUT/DAGGER candidate stays closed. Fifteen players had both a supported Combine cut and a supported NFL cut, below the predeclared minimum of 20. Its protocol, tests, and no-go report are unchanged. This document does not fit a prediction model and does not use separation, sacks, or EPA to choose a question.

## Bars, fixed before treating any candidate as supported

These are not lowered to let a question through.

- At least 30 distinct 2023–24 prospects with both a usable Combine measurement and a usable NFL movement measurement.
- An untouched evaluation class of at least 20 such prospects. A new feature on the wide receivers already scored in 2025 is not a fresh holdout.
- A median of at least 8 NFL movement observations per matched prospect.
- At least half of the matched development prospects have two comparable Combine trials. Otherwise repeatability is not claimable.
- The NFL measurement must be the prospect's own movement. Game tracking does not include every defender, the quarterback, or the ball.

## What 2025 outcomes have already been seen

The 2025 wide-receiver evaluation used `separation_at_pass_forward` for 19 receivers and 3,552 routes. The same class was used for the Combine-to-NFL cut correlation and for the retrospective common-penalty check. 2023 and 2024 receivers were the development sets for that study. No model has used 2025 sack, pressure, get-off, or non-receiver separation as an endpoint. A pooled min/median/max of `player_get_off` was computed once across all seasons while checking units; it was discarded. The range below uses 2023–24 only. No 2025 get-off value was joined to a Combine time.

## The three questions

### 1. Stationary acceleration

Combine drill: `FORTY_YARD_DASH`. The run is along x (median span 40.4 yards, median y span 0.4). In 768 of 793 attempts the fastest of the first three frames is under 2 yards/second, so the start is near rest. Most prospects have two timed attempts.

NFL situation: defensive linemen (DE, DT, NT) on rookie regular-season plays where `player_get_off` is present. Values in 2023–24 run from about 0.2 to 2.0 seconds (DE median 0.92, DT 1.03, NT 1.08), which is a timing, not a yardage. It does not require the ball or the blocker location.

A coach could decide whether a forty start agrees with timed get-off often enough to treat the forty as a no-contact proxy. The endpoint is get-off time, not sacks or expected points.

2023–24 matched linemen: 19 in 2023 and 20 in 2024 (39). Seventeen and 17 have two forty attempts. Median get-off plays: 78 and 148. Eighteen and 19 have at least 8 plays. 2025 matched linemen: 17, all with two forties and at least 8 get-off plays.

Measurement problem: time from the first tracking frame until x has increased 10 yards has median 2.05 seconds on 50 development linemen. The official `ten_yd_split` for those same players has median 1.72 seconds (Pearson 0.60). The tracking clock is not the official split. The primary Combine number is therefore the official 10-yard split. The two tracking attempts stay available for a later repeatability check, not as a silent replacement for the official time.

Wide receivers also have forty tracking and many rookie plays (24 and 26 matched in 2023 and 2024; median about 200 plays). They are not an untouched evaluation population.

### 2. Lateral efficiency

Combine drill that is actually lateral: `SHORT_SHUTTLE` or `THREE_CONE_DRILL`. Both are sparse and usually one attempt. Defensive-back skill drills (`BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL`, `W_DRILL`) are one attempt for almost every player.

NFL situation would have to be invented. Tracking does not locate the opponent or the ball, so closing speed and leverage are not observed. A new change-of-direction detector would repeat the closed cut project.

Matched cornerbacks and safeties with shuttle tracking and a rookie play: 7 in 2023, 6 in 2024, 6 in 2025. Two-attempt shuttles are 2, 2, and 0. This misses the 30-player bar and the repeat-trial bar.

### 3. Speed maintenance

Combine: speed over the last part of the forty, or the single `GO_ROUTE_RIGHT` (98 players, 105 attempts across three classes, about one rep). NFL: a sustained straight segment. That segment is not a supplied field. Building it is another detector. Defensive linemen do not have a 40-yard in-play sprint that matches the forty. Receiver go routes are common (median about 40 per matched receiver) but those receivers are the population already used for 2025 separation.

## Comparison

| Question | Combine observation | NFL observation | 2023–24 matched prospects | Two Combine trials | Median NFL reps | 2025 untouched prospects | Demand overlap | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Stationary start | Official 10-yard split; forty tracking for repeats | `player_get_off` for DE/DT/NT | 39 | 34 of 39 | 78 and 148 by class | 17 | Stance start only; no opponent in the forty | Development sample passes. Fresh test does not. |
| Lateral efficiency | Shuttle or cone, usually once | Not in the file without a new detector | 13 defensive backs with a shuttle | 4 of 13 | hundreds of plays, not lateral events | 6 | Opponent location missing | Not supported |
| Speed maintenance | One go-route rep, or the end of the forty | Straight segment not supplied | Receivers already studied | About one go rep | Go routes are common for receivers | None. 2025 receivers were scored | Poor for linemen; contaminated for receivers | Not supported |

## Decision

Question 1 for defensive linemen is the only candidate with a defined drill, a supplied NFL timing, two forty trials for most players, and 39 development prospects. It is not admitted as a 2025 confirmatory study: 17 matched evaluation prospects is below 20. No outcome model is fit.

Exact definitions, if a later development-only analysis is opened:

- Population: `nfl_position` in DE, DT, NT. Rookie regular season (`games.season = players.draft_year`, `season_type = REG`).
- Combine feature: `combine_results.ten_yd_split`, seconds. Missing splits are missing, not imputed from a mismatched tracking clock.
- Repeatability side check, not the grade: for each `FORTY_YARD_DASH` attempt, time from the first frame until x increases 10 yards. Report the bias against the official split before interpreting agreement between attempts.
- NFL movement: the player's median `player_get_off` on plays where it is present. Plays without it are missing, not slow get-offs.
- Not the endpoint: sacks, pressure, EPA, yards, or separation.
- Validation, still unfitted: 2023 fit, 2024 chronological check, player-equal weights, uncertainty over players. 2025 is not used. A later 2025 look would be a small retrospective class, not a fresh high-powered test.
- Stop rule: if the two forty tracking attempts disagree by about as much as linemen differ from each other, do not treat the official split as a stable individual grade.

The next research step is that development-only comparison of the official 10-yard split with median get-off. It is not another receiver cut model.
