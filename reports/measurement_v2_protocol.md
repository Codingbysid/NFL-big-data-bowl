# Measurement v2 protocol

Date: 2026-10-09. Status: post-evaluation. This file is not a preregistration of the original study. It is committed locally before the versioned measurement run. It does not change `archive/original_evaluation/` or the historical −0.0158-yard result.

## Candidate

Combine `SPEED_OUT` and `DAGGER`, compared with NFL `OUT` and `IN`. The choice follows the existing geometry review: these are recognizable stem-and-break routes. Other families stay excluded. Slot strikes were often marked after the plant, go routes often marked a release, and several rounded breaks were not one isolated turn. They are not a list to search until a correlation looks better.

The question is whether, after one recognizable break, exit distance relative to entry speed, turn angle, and drill shows the same ordering on comparable NFL breaks. The half-second projection and heading-midpoint reference stay as in the historical definition. Tracking does not identify a foot plant, body facing, or a physiological cause.

## One break rule

`single_bounded_break`. An attempt is kept only when the bounded detector returns exactly one cut. No cut is a rejection or an unavailable measurement. Two or more cuts is abstention, not an average and not zero ability.

## Information boundary

For NFL routes the boundary is the earlier of `pass_forward` and the declared contact or catch events (`first_contact`, `pass_arrived`, `pass_outcome_caught`, `tackle`, `pass_outcome_touchdown`). Coordinates after that time are dropped before smoothing, derivatives, window search, and the exit projection. The latest raw timestamp allowed into the measurement is stored on every accepted cut. A missing snap or pass rejects the route. Combine drills have no pass tag; the full drill recording is the observation.

## Support

Support cells are counted on 2023 in-scope cuts only, using the historical speed and angle edges. A cell is supported when it has at least 8 Combine cuts and 8 NFL cuts. That minimum is lower than the historical 15 because this scope is two families, and it was set here before any separation result. The same cells are applied to 2023 and 2024. The demand model is fit on all eligible in-scope training cuts, not only supported cells. Support filters both sides of a later comparison before player aggregation. Unsupported cuts are missing.

## Demand penalty

Objective: `sum(w * residual^2) / sum(w) + lambda * ||beta||^2`, with each player and then each attempt weighted equally. scikit-learn Ridge uses `alpha = lambda * sum(w)`. The grid is 0 (ordinary least squares), 0.01, 0.1, 1, and 3 (the historical setting). Lambda minimizes player-grouped cross-validated weighted squared error of Combine exit distance. Separation is not an input. Standardization and family indicators are fit inside each training fold. Out-of-fold family means are reported with a player-resample interval. They are not required to be zero.

## Review

Seed 20271009. Forty percent of in-scope players, chosen once, are the validation gallery. The rest are a detector-development pool that is not used to change these settings. From validation players, sample up to 30 accepted candidates per domain, split across the two families, plus up to 10 rejected trajectories per domain. Sampling probabilities and stratum sizes are stored. Annotation plots show the path and the information boundary and do not show the detector marker. Detector times live in a separate key file.

The engineering gate for this iteration is at least 90 percent intended-break agreement among reviewed accepted candidates in each domain. Ambiguous labels are not agreement. The denominator is the reviewed accepted candidates. A player-resample interval is reported if labels exist. Zero accepted measurements may use a frame after the boundary. This 90 percent figure is not an NFL standard and is not a precision or recall for all routes.

An empty label sheet means validation is pending. Labels are not filled in by the analyst who can see the detector. An automatic inspection is not a human gold label.

## Sample-size rule, before any separation result

A player-level translation or performance comparison requires at least 20 players with both a supported Combine trait and a supported NFL trait, including at least 15 such players in 2024, and at least two single supported attempts for a Combine trait. A later performance model, if the measurement gate passes, would use receivers with at least 8 supported in-scope routes. That route count is not chosen from a correlation. The historical 0.10-yard bar stays with the historical broad-route endpoint and is not reused here as a pass-fail rule.

## What this protocol does not do

It does not refit the archived 2025 model, does not read 2025 separation, and does not try another feature family if this candidate fails.
