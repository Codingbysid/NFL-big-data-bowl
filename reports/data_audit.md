# Data audit

Data version `f57bbd690bd2b5d3`. Raw files were hashed and were not modified.

## Files

| file | bytes | sha256 |
| --- | --- | --- |
| players.csv | 47301 | 2cd7e0309e7aa405 |
| combine_results.csv | 41020 | e4773ef68167d508 |
| combine_tracking.csv | 83788174 | 541c69f6629f5f80 |
| player_career_successes.csv | 13801 | 0d4b8975debda4e2 |
| player_play.csv | 146879042 | 68926a268c0a49e5 |
| games.csv | 43174 | ca33543f6939af4e |
| game_tracking_2023.csv | 335463637 | 385e620b6b94e8cd |
| game_tracking_2024.csv | 717952893 | 06b7f34304414306 |
| game_tracking_2025.csv | 1022575093 | 419b6bc0388a9210 |

## Schema against the supplied field list

| file | missing | extra |
| --- | --- | --- |
| players.csv | — | — |
| combine_results.csv | — | — |
| combine_tracking.csv | — | — |
| games.csv | — | — |
| game_tracking_2024.csv | — | — |
| game_tracking_2025.csv | — | — |
| game_tracking_2023.csv | — | — |

Game tracking has an `o` orientation field. Combine tracking does not. Combine `dir` is movement direction. No column supports foot placement, hip angle, or limb asymmetry.

## Roster and games

| draft_year | nfl_position | n |
| --- | --- | --- |
| 2023 | C | 7 |
| 2023 | CB | 15 |
| 2023 | DE | 15 |
| 2023 | DT | 8 |
| 2023 | FS | 7 |
| 2023 | G | 17 |
| 2023 | NT | 3 |
| 2023 | OLB | 11 |
| 2023 | SS | 13 |
| 2023 | T | 13 |
| 2023 | TE | 13 |
| 2023 | WR | 38 |
| 2024 | C | 10 |
| 2024 | CB | 28 |
| 2024 | DB | 1 |
| 2024 | DE | 12 |
| 2024 | DT | 17 |
| 2024 | FS | 10 |
| 2024 | G | 15 |
| 2024 | ILB | 1 |
| 2024 | MLB | 1 |
| 2024 | NT | 1 |
| 2024 | OLB | 7 |
| 2024 | RB | 1 |
| 2024 | SS | 8 |
| 2024 | T | 21 |
| 2024 | TE | 13 |
| 2024 | WR | 31 |
| 2025 | C | 3 |
| 2025 | CB | 17 |
| 2025 | DB | 4 |
| 2025 | DE | 9 |
| 2025 | DT | 22 |
| 2025 | FB | 1 |
| 2025 | FS | 8 |
| 2025 | G | 19 |
| 2025 | ILB | 1 |
| 2025 | NT | 1 |
| 2025 | OLB | 12 |
| 2025 | SS | 7 |
| 2025 | T | 16 |
| 2025 | TE | 16 |
| 2025 | WR | 37 |

| season | season_type | n |
| --- | --- | --- |
| 2023 | POST | 13 |
| 2023 | PRE | 49 |
| 2023 | REG | 272 |
| 2024 | POST | 13 |
| 2024 | PRE | 49 |
| 2024 | REG | 272 |
| 2025 | POST | 13 |
| 2025 | PRE | 49 |
| 2025 | REG | 272 |

Key check: | players | players_ids | combine_dup_ids | players_missing_combine |
| --- | --- | --- | --- |
| 510 | 510 | 0 | 0 |

Penalty codes other than Y/N among wide-receiver plays: 0.

## Combine tracking

| entity_type | frames | ids |
| --- | --- | --- |
| PLAYER | 431094 | 510 |
| BALL | 32095 | 54 |

BALL rows exist and are excluded from player trajectories. Wide-receiver drill inventory:

| draft_year | drill_type | drill_name | players | frames |
| --- | --- | --- | --- | --- |
| 2023 | FORTY_YARD_DASH | FORTY_YARD_DASH | 32 | 3026 |
| 2024 | FORTY_YARD_DASH | FORTY_YARD_DASH | 28 | 2417 |
| 2025 | FORTY_YARD_DASH | FORTY_YARD_DASH | 33 | 3329 |
| 2023 | SHORT_SHUTTLE | SHORT_SHUTTLE | 12 | 778 |
| 2024 | SHORT_SHUTTLE | SHORT_SHUTTLE | 8 | 461 |
| 2025 | SHORT_SHUTTLE | SHORT_SHUTTLE | 9 | 588 |
| 2025 | SKILL_DRILLS_WR | CIRCUS_ROUTE_LEFT | 37 | 2551 |
| 2023 | SKILL_DRILLS_WR | COMEBACK_ROUTE_RIGHT | 35 | 1929 |
| 2024 | SKILL_DRILLS_WR | COMEBACK_ROUTE_RIGHT | 28 | 1714 |
| 2025 | SKILL_DRILLS_WR | COMEBACK_ROUTE_RIGHT | 34 | 2950 |
| 2023 | SKILL_DRILLS_WR | CURL_ROUTE_RIGHT | 36 | 2236 |
| 2024 | SKILL_DRILLS_WR | CURL_ROUTE_RIGHT | 30 | 1967 |
| 2025 | SKILL_DRILLS_WR | CURL_ROUTE_RIGHT | 36 | 1949 |
| 2023 | SKILL_DRILLS_WR | DAGGER_ROUTE_LEFT | 38 | 2846 |
| 2024 | SKILL_DRILLS_WR | DAGGER_ROUTE_LEFT | 31 | 2045 |
| 2025 | SKILL_DRILLS_WR | DAGGER_ROUTE_LEFT | 37 | 2570 |
| 2023 | SKILL_DRILLS_WR | GAUNTLET_DRILL | 38 | 6248 |
| 2024 | SKILL_DRILLS_WR | GAUNTLET_DRILL | 31 | 5396 |
| 2025 | SKILL_DRILLS_WR | GAUNTLET_DRILL | 37 | 6437 |
| 2023 | SKILL_DRILLS_WR | GO_ROUTE_RIGHT | 34 | 2960 |
| 2024 | SKILL_DRILLS_WR | GO_ROUTE_RIGHT | 29 | 2143 |
| 2025 | SKILL_DRILLS_WR | GO_ROUTE_RIGHT | 34 | 2480 |
| 2023 | SKILL_DRILLS_WR | OVER_SHOULDER_ADJUST | 36 | 2370 |
| 2024 | SKILL_DRILLS_WR | OVER_SHOULDER_ADJUST | 29 | 1848 |
| 2025 | SKILL_DRILLS_WR | OVER_SHOULDER_ADJUST | 34 | 2325 |
| 2023 | SKILL_DRILLS_WR | POST_CORNER_ROUTE_LEFT | 38 | 2137 |
| 2024 | SKILL_DRILLS_WR | POST_CORNER_ROUTE_LEFT | 31 | 2099 |
| 2023 | SKILL_DRILLS_WR | POST_CORNER_ROUTE_RIGHT | 35 | 2442 |
| 2024 | SKILL_DRILLS_WR | POST_CORNER_ROUTE_RIGHT | 29 | 2587 |
| 2025 | SKILL_DRILLS_WR | POST_CORNER_ROUTE_RIGHT | 31 | 2846 |
| 2023 | SKILL_DRILLS_WR | RED_ZONE_FADE_RIGHT | 35 | 1899 |
| 2024 | SKILL_DRILLS_WR | RED_ZONE_FADE_RIGHT | 29 | 1712 |
| 2025 | SKILL_DRILLS_WR | RED_ZONE_FADE_RIGHT | 33 | 1364 |
| 2023 | SKILL_DRILLS_WR | SLANT_ROUTE_LEFT | 38 | 1778 |
| 2024 | SKILL_DRILLS_WR | SLANT_ROUTE_LEFT | 31 | 1466 |
| 2025 | SKILL_DRILLS_WR | SLANT_ROUTE_LEFT | 37 | 1036 |
| 2023 | SKILL_DRILLS_WR | SLOT_SAIL_ROUTE_RIGHT | 32 | 2712 |
| 2024 | SKILL_DRILLS_WR | SLOT_SAIL_ROUTE_RIGHT | 30 | 2274 |
| 2025 | SKILL_DRILLS_WR | SLOT_SAIL_ROUTE_RIGHT | 35 | 2346 |
| 2023 | SKILL_DRILLS_WR | SLOT_STRIKE_ROUTE_LEFT | 36 | 2323 |
| 2024 | SKILL_DRILLS_WR | SLOT_STRIKE_ROUTE_LEFT | 28 | 1647 |
| 2025 | SKILL_DRILLS_WR | SLOT_STRIKE_ROUTE_LEFT | 36 | 2301 |
| 2024 | SKILL_DRILLS_WR | SPEED_OUT_LEFT | 31 | 1675 |
| 2023 | SKILL_DRILLS_WR | SPEED_OUT_ROUTE_LEFT | 38 | 1786 |
| 2025 | SKILL_DRILLS_WR | SPEED_OUT_ROUTE_LEFT | 37 | 2006 |
| 2023 | THREE_CONE_DRILL | THREE_CONE_DRILL | 5 | 385 |
| 2024 | THREE_CONE_DRILL | THREE_CONE_DRILL | 10 | 1000 |
| 2025 | THREE_CONE_DRILL | THREE_CONE_DRILL | 9 | 1019 |

## Game tracking

These files contain the prospect cohort, not all 22 players. Defender separation cannot be recomputed from them.

| frames | players | games | duplicate_timestamps | null_ids | season_file |
| --- | --- | --- | --- | --- | --- |
| 3665443 | 156 | 333 | 0 | 0 | 2023 |
| 7827944 | 318 | 334 | 0 | 0 | 2024 |
| 11160903 | 448 | 334 | 0 | 0 | 2025 |

## Separation label

Counts below include the 2025 class. Means do not.

| draft_year | pass_result | routes | separation_observed |
| --- | --- | --- | --- |
| 2023 | C | 13799 | 13799 |
| 2023 | I | 7391 | 7391 |
| 2023 | IN | 529 | 529 |
| 2023 | R | 1255 | 0 |
| 2023 | S | 1729 | 0 |
| 2024 | C | 6785 | 6785 |
| 2024 | I | 3769 | 3769 |
| 2024 | IN | 270 | 270 |
| 2024 | R | 777 | 0 |
| 2024 | S | 812 | 0 |
| 2025 | C | 3019 | 3019 |
| 2025 | I | 1846 | 1846 |
| 2025 | IN | 117 | 117 |
| 2025 | R | 289 | 0 |
| 2025 | S | 419 | 0 |

| draft_year | target | routes | separation_observed |
| --- | --- | --- | --- |
| 2023 | False | 17003 | 17003 |
| 2023 | True | 4716 | 4716 |
| 2024 | False | 8446 | 8446 |
| 2024 | True | 2378 | 2378 |
| 2025 | False | 3983 | 3983 |
| 2025 | True | 999 | 999 |

| population | n | mean | sd | p10 | p50 | p90 |
| --- | --- | --- | --- | --- | --- | --- |
| 2023-24 eligible routes only | 10648 | 2.979 | 2.004 | 0.908 | 2.523 | 5.707 |

On development routes the label is present for completions, incompletions, and interceptions, including untargeted routes, and absent for sacks and scrambles. The estimand is separation on rookie regular-season routes with a forward pass, not separation on targets only.

## Exclusion funnel

| draft_year | stage | routes | players | games | players_started |
| --- | --- | --- | --- | --- | --- |
| 2023 | wr_plays_in_file | 24703 | 38 | 760 | 38 |
| 2023 | wr_roster | 24703 | 38 | 760 | 38 |
| 2023 | regular_season | 20498 | 31 | 610 | 38 |
| 2023 | rookie_season | 7416 | 29 | 228 | 38 |
| 2023 | not_penalty | 7416 | 29 | 228 | 38 |
| 2023 | has_route | 7416 | 29 | 228 | 38 |
| 2023 | not_rare | 7405 | 29 | 228 | 38 |
| 2023 | not_excluded_family | 7152 | 29 | 228 | 38 |
| 2023 | forward_pass | 6280 | 29 | 228 | 38 |
| 2023 | has_separation | 6280 | 29 | 228 | 38 |
| 2024 | wr_plays_in_file | 12413 | 30 | 447 | 30 |
| 2024 | wr_roster | 12413 | 30 | 447 | 30 |
| 2024 | regular_season | 10164 | 28 | 350 | 30 |
| 2024 | rookie_season | 5201 | 27 | 191 | 30 |
| 2024 | not_penalty | 5201 | 27 | 191 | 30 |
| 2024 | has_route | 5201 | 27 | 191 | 30 |
| 2024 | not_rare | 5191 | 27 | 191 | 30 |
| 2024 | not_excluded_family | 5066 | 27 | 191 | 30 |
| 2024 | forward_pass | 4368 | 27 | 191 | 30 |
| 2024 | has_separation | 4368 | 27 | 191 | 30 |
| 2025 | wr_plays_in_file | 5690 | 37 | 243 | 37 |
| 2025 | wr_roster | 5690 | 37 | 243 | 37 |
| 2025 | regular_season | 4227 | 23 | 188 | 37 |
| 2025 | rookie_season | 4227 | 23 | 188 | 37 |
| 2025 | not_penalty | 4227 | 23 | 188 | 37 |
| 2025 | has_route | 4227 | 23 | 188 | 37 |
| 2025 | not_rare | 4222 | 23 | 188 | 37 |
| 2025 | not_excluded_family | 4126 | 23 | 188 | 37 |
| 2025 | forward_pass | 3605 | 23 | 185 | 37 |
| 2025 | has_separation | 3605 | 23 | 185 | 37 |
| all | wr_plays_in_file | 42806 | 105 | 932 | 105 |
| all | wr_roster | 42806 | 105 | 932 | 105 |
| all | regular_season | 34889 | 82 | 752 | 105 |
| all | rookie_season | 16844 | 79 | 607 | 105 |
| all | not_penalty | 16844 | 79 | 607 | 105 |
| all | has_route | 16844 | 79 | 607 | 105 |
| all | not_rare | 16818 | 79 | 607 | 105 |
| all | not_excluded_family | 16344 | 79 | 607 | 105 |
| all | forward_pass | 14253 | 79 | 604 | 105 |
| all | has_separation | 14253 | 79 | 604 | 105 |

## Where receivers lined up

| draft_year | lined_up_position | routes |
| --- | --- | --- |
| 2023 | FB | 9 |
| 2023 | RB | 28 |
| 2023 | SLOT_WR | 2530 |
| 2023 | TE | 172 |
| 2023 | WR | 4677 |
| 2024 | FB | 5 |
| 2024 | RB | 14 |
| 2024 | SLOT_WR | 2038 |
| 2024 | TE | 100 |
| 2024 | WR | 3044 |
| 2025 | FB | 2 |
| 2025 | RB | 13 |
| 2025 | SLOT_WR | 1491 |
| 2025 | TE | 101 |
| 2025 | WR | 2620 |

The roster rule is `nfl_position == WR`. A wide receiver who lines up in the slot, or occasionally at tight end, stays in the cohort. Tight ends are not added.

## Inclusion

Primary means at least 20 eligible rookie regular-season routes. Career production is not used.

| draft_year | group | players | forty_mean | forty_missing | height_mean | weight_mean | undrafted | three_cone_missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | primary | 28 | 4.482 | 4 | 72.42 | 194.107 | 4 | 20 |
| 2023 | excluded | 10 | 4.54 | 0 | 74.15 | 210.9 | 8 | 6 |
| 2024 | primary | 21 | 4.453 | 0 | 73.256 | 200.476 | 1 | 14 |
| 2024 | excluded | 10 | 4.445 | 2 | 71.3 | 189.2 | 3 | 8 |
| 2025 | primary | 19 | 4.427 | 1 | 72.178 | 192.895 | 1 | 12 |
| 2025 | excluded | 18 | 4.502 | 3 | 73.368 | 197.278 | 13 | 14 |

## Measurement, from the development extract
- 2023 NFL tracked forward passes: 6521. Share with a kept cut: 0.104. Snap tag present: 1.000. Pass tag present: 0.996. Duplicate timestamps: 0.
- 2024 NFL tracked forward passes: 4479. Share with a kept cut: 0.098. Snap tag present: 1.000. Pass tag present: 0.998. Duplicate timestamps: 0.

Primary-family Combine cuts after the 4.0 yd/s entry gate: 406 cuts, 69 receivers, median 5 attempts, minimum 3.
Share of those receivers with at least 2 attempts: 1.000.

| drill_family | cuts | angle | entry_speed | exit_yards |
| --- | --- | --- | --- | --- |
| DAGGER | 110 | 56.574 | 7.613 | 3.275 |
| POST_CORNER | 103 | 56.429 | 7.748 | 3.435 |
| SLANT | 13 | 36.473 | 4.479 | 2.971 |
| SLOT_SAIL | 97 | 68.116 | 7.406 | 3.216 |
| SPEED_OUT | 83 | 73.786 | 6.691 | 2.918 |
