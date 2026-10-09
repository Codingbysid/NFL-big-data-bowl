# Protocol

config_hash: dfc5bbb37f447ee85ca1eea3d26d2c0e7a746567c39394c2f78be23be0fb3505

This file was written before the outcome models were fit. The 2025 class may be counted for availability. Its separation values, distributions, and errors are not used until `evaluate-heldout`.

## Question

Among wide receivers with comparable conventional Combine measurements, does gaining more ground in the half-second after a turn, after adjusting for the speed and angle of that turn, predict more rookie-season separation?

Direction: a larger cut-exit residual is hypothesized to predict larger separation. The residual is in yards. It is not a defender-separation measure and it is not a force.

## Cohort and outcome

Roster: `nfl_position == WR`. Slot alignments stay in. Tight ends are not added.

Routes: rookie season only (`games.season == players.draft_year`), regular season only, penalty code N, a forward pass (`pass_result` in ['C', 'I', 'IN']), route in ['GO', 'HITCH', 'CROSS', 'IN', 'OUT', 'POST', 'CORNER', 'SLANT', 'FLAT']. Screens (['SCREEN']) and rare labels (['WHEEL', 'ANGLE']) are out. Missing separation stays missing and is not filled with zero.

The label is present on targeted and untargeted routes when a pass is thrown. The estimand is separation on those routes, not separation conditional on a target.

Primary sample: at least 20 eligible routes. Sensitivities use [10, 40] as evaluation filters, without refitting the ladder to chase them. A man-coverage check requires at least 5 man routes.

## Tracking trait

Combine families in the primary trait: ['SPEED_OUT', 'SLOT_SAIL', 'DAGGER', 'POST_CORNER', 'SLANT']. Names such as `SPEED_OUT_LEFT` and `SPEED_OUT_ROUTE_LEFT` are one family. Turn-back drills (['COMEBACK', 'CURL', 'SLOT_STRIKE']) and ['GO_ROUTE_RIGHT', 'RED_ZONE_FADE_RIGHT', 'GAUNTLET_DRILL', 'OVER_SHOULDER_ADJUST', 'CIRCUS_ROUTE_LEFT', 'FORTY_YARD_DASH', 'SHORT_SHUTTLE', 'THREE_CONE_DRILL'] are not in the primary trait.

A cut needs a stable incoming heading and a stable outgoing heading, a net angle from 30.0 to 110.0 degrees, incoming speed at least 4.0 yards per second, and 0.5 seconds of usable tracking on each side. The reference time is the halfway point of the heading change, not the slowest frame. Exit distance is yards traveled along the outgoing direction over the next 0.5 seconds.

The 4 yard-per-second entry gate was set after the outcome-blind gallery. Slower detections were release arcs and low-speed wiggles. The gallery notes are in `reports/review_sheet.csv`.

On NFL routes the same geometry is used, and the half-second window must end before the pass and before contact or a catch. Routes without snap and pass tags are not measured. A missing NFL trait is not zero.

Expected exit distance is a ridge on incoming speed, absolute angle, their interaction, and a shrunk drill-family offset (penalty 3.0). It is fit on training players only, with each training player weighted equally and each attempt inside a player weighted equally. The player trait is the median residual. Fewer than 2 attempts leaves the trait missing. Shrinkage is a sensitivity, not the primary predictor.

## Models

The ladder is ridge regression on routes, with each training player's weights summing to 1.

- M0: route, man/zone, cushion, down, distance, yard line, own-side flag, formation, alignment, motion.
- M1: M0 plus height, weight, age at the Combine tracking date (March 1 of the draft year if that date is missing), forty, ten-yard split, vertical, broad jump, three-cone, and short shuttle. Medians and missingness indicators are fit inside the training fold.
- M2: M1 plus the Combine cut-exit residual.

Player id and name are not features. NFL exit residuals, targets, catches, and yards are not features. NGS grades and draft slot are sensitivities. A forty-and-ten complete case is a sensitivity. Replacing the residual with entry speed is the ablation that checks whether the trait is just "ran into the turn fast."

The penalty is chosen from [0.3, 1.0, 3.0, 10.0, 30.0] by player-grouped error inside the training fold. The primary metric is the mean, across players, of the absolute gap between a player's mean actual separation and mean predicted separation. M2 minus M1 is the comparison. Negative means a smaller error. The analyst-chosen bar for a useful gain is 0.1 yards. That number is not an NFL standard.

## Chronology

2023 is the development class, with player-grouped out-of-fold predictions. 2024 is a chronological check. The recipe is then refit on 2023 and 2024 and applied once to 2025. A tree model, if it is run, is secondary and its settings are chosen on 2023 only.

Profiles for the writeup are the 2025 receivers inside 0.15 seconds of the 2023-24 median forty whose pre-draft tracking projection and conventional projection disagree the most in each direction. A large miss is shown as a miss. It is chosen after the outcomes and is labeled that way.

## What this will not claim

A positive point estimate without a stable measurement and a holdout interval is not a scouting rule. An association is not a training effect. Left-versus-right drill names are not limb asymmetry. There is no defender-tracking separation in these files.
