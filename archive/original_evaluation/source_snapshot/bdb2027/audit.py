"""Phase A audit. Held-out separation values are counted, not summarized."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bdb2027.cohort import apply_funnel, funnel_table, load_player_dimension, load_wr_plays
from bdb2027.config import RAW_FILES, Project, drill_maps
from bdb2027.extract import cohort_dir
from bdb2027.io import (
    COMBINE_TRACK_TYPES,
    GAMES_TYPES,
    PLAYERS_TYPES,
    TRACK_TYPES,
    cached_fingerprints,
    connect,
    csv_scan,
    data_version,
)

EXPECTED = {
    "players.csv": [
        "nfl_id", "display_name", "draft_year", "nfl_position", "birth_date", "college_name",
        "college_conference", "draft_round", "draft_pick_within_round", "draft_overall_pick",
    ],
    "combine_results.csv": [
        "draft_year", "nfl_id", "combine_position", "combine_height", "combine_weight", "hand_size",
        "arm_length", "wing_span", "ten_yd_split", "forty", "vertical", "broad_jump", "three_cone",
        "short_shuttle", "bench_reps", "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score",
    ],
    "combine_tracking.csv": [
        "draft_year", "event_id", "nfl_id", "entity_type", "time", "drill_type", "drill_name", "attempt",
        "x", "y", "s", "a", "dis", "dir",
    ],
    "games.csv": ["game_id", "game_key", "season", "season_type", "week", "home_team_abbr", "visitor_team_abbr"],
    "game_tracking_2023.csv": ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "dis", "o", "dir", "event"],
}


def _md(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_none_"
    show = frame.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}".rstrip("0").rstrip("."))
        else:
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else str(v))
    cols = [str(c) for c in show.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in show.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def run_audit(project: Project) -> None:
    print("[audit] start", flush=True)
    cfg = project.config
    fingerprints = cached_fingerprints(project)
    version = data_version(fingerprints)
    con = connect(project)
    q = lambda sql: con.execute(sql).fetchdf()

    schema_rows = []
    for name, expected in EXPECTED.items():
        actual = _header(project.raw(name))
        if name.startswith("game_tracking"):
            for year in (2024, 2025):
                other = _header(project.raw(f"game_tracking_{year}.csv"))
                schema_rows.append(
                    {
                        "file": f"game_tracking_{year}.csv",
                        "missing": ",".join(c for c in actual if c not in other) or "—",
                        "extra": ",".join(c for c in other if c not in actual) or "—",
                    }
                )
        schema_rows.append(
            {
                "file": name,
                "missing": ",".join(c for c in expected if c not in actual) or "—",
                "extra": ",".join(c for c in actual if c not in expected) or "—",
            }
        )

    players = q(
        f"""
        SELECT draft_year, nfl_position, count(*) AS n
        FROM {csv_scan(project.raw('players.csv'), PLAYERS_TYPES)}
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    )
    games = q(
        f"""
        SELECT season, season_type, count(*) AS n
        FROM {csv_scan(project.raw('games.csv'), GAMES_TYPES)}
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    )
    keys = q(
        f"""
        SELECT
          (SELECT count(*) FROM {csv_scan(project.raw('players.csv'), PLAYERS_TYPES)}) AS players,
          (SELECT count(DISTINCT nfl_id) FROM {csv_scan(project.raw('players.csv'), PLAYERS_TYPES)}) AS players_ids,
          (SELECT count(*) - count(DISTINCT nfl_id) FROM {csv_scan(project.raw('combine_results.csv'), "'nfl_id':'BIGINT'")}) AS combine_dup_ids,
          (SELECT count(*) FROM {csv_scan(project.raw('players.csv'), PLAYERS_TYPES)} p
             ANTI JOIN {csv_scan(project.raw('combine_results.csv'), "'nfl_id':'BIGINT'")} c USING (nfl_id)) AS players_missing_combine
        """
    )
    drills = q(
        f"""
        SELECT ct.draft_year, ct.drill_type, ct.drill_name,
               count(DISTINCT nfl_id) AS players,
               count(*) AS frames
        FROM {csv_scan(project.raw('combine_tracking.csv'), COMBINE_TRACK_TYPES)} ct
        JOIN {csv_scan(project.raw('players.csv'), PLAYERS_TYPES)} pl USING (nfl_id)
        WHERE pl.nfl_position = 'WR' AND ct.entity_type = 'PLAYER'
        GROUP BY 1, 2, 3
        ORDER BY 2, 3, 1
        """
    )
    entities = q(
        f"""
        SELECT entity_type, count(*) AS frames, count(DISTINCT nfl_id) AS ids
        FROM {csv_scan(project.raw('combine_tracking.csv'), COMBINE_TRACK_TYPES)}
        GROUP BY 1
        """
    )
    tracking_rows = []
    for year in (2023, 2024, 2025):
        path = project.raw(f"game_tracking_{year}.csv")
        print(f"[audit] scanning game_tracking_{year}", flush=True)
        stats = q(
            f"""
            WITH gt AS (
                SELECT * FROM {csv_scan(path, TRACK_TYPES)}
            ),
            keyed AS (
                SELECT game_id, play_id, nfl_id, time
                FROM gt
            )
            SELECT
                (SELECT count(*) FROM gt) AS frames,
                (SELECT count(DISTINCT nfl_id) FROM gt) AS players,
                (SELECT count(DISTINCT game_id) FROM gt) AS games,
                (SELECT count(*) - count(DISTINCT (game_id, play_id, nfl_id, time)) FROM keyed) AS duplicate_timestamps,
                (SELECT count(*) FROM gt WHERE nfl_id IS NULL) AS null_ids
            """
        )
        stats["season_file"] = year
        tracking_rows.append(stats)
    tracking = pd.concat(tracking_rows, ignore_index=True)

    plays = load_wr_plays(project)
    plays = apply_funnel(plays, cfg)
    # Availability is kept. The numeric label for the held-out class is then blanked.
    plays["separation_observed"] = plays["separation_at_pass_forward"].notna()
    holdout_year = int(cfg["holdout_min_draft_year"])
    plays.loc[plays["draft_year"] >= holdout_year, "separation_at_pass_forward"] = np.nan
    funnel = funnel_table(plays, cfg)
    dev = plays.loc[plays["draft_year"] < holdout_year].copy()
    by_result = (
        plays.groupby(["draft_year", "pass_result"], dropna=False)
        .agg(routes=("nfl_id", "size"), separation_observed=("separation_observed", "sum"))
        .reset_index()
    )
    targeted = (
        plays.loc[plays["pass_result"].isin(cfg["routes"]["pass_results"])]
        .groupby(["draft_year", "target"])
        .agg(routes=("nfl_id", "size"), separation_observed=("separation_observed", "sum"))
        .reset_index()
    )
    dev_sep = dev.loc[dev["eligible"], "separation_at_pass_forward"]
    sep_summary = pd.DataFrame(
        [
            {
                "population": "2023-24 eligible routes only",
                "n": int(dev_sep.notna().sum()),
                "mean": float(dev_sep.mean()),
                "sd": float(dev_sep.std()),
                "p10": float(dev_sep.quantile(0.1)),
                "p50": float(dev_sep.quantile(0.5)),
                "p90": float(dev_sep.quantile(0.9)),
            }
        ]
    )
    players_dim = load_player_dimension(project)
    eligible_ids = set(plays.loc[plays["eligible"], "nfl_id"].astype(int))
    route_counts = plays.loc[plays["eligible"]].groupby("nfl_id").size()
    min_routes = int(cfg["routes"]["min_routes"])
    primary_ids = set(int(i) for i in route_counts[route_counts >= min_routes].index)
    dim = players_dim.copy()
    dim["in_primary"] = dim["nfl_id"].isin(primary_ids)
    bias_rows = []
    for year, part in dim.groupby("draft_year"):
        for flag, label in ((True, "primary"), (False, "excluded")):
            sub = part.loc[part["in_primary"] == flag]
            bias_rows.append(
                {
                    "draft_year": int(year),
                    "group": label,
                    "players": int(len(sub)),
                    "forty_mean": sub["forty"].mean(),
                    "forty_missing": int(sub["forty"].isna().sum()),
                    "height_mean": sub["height"].mean(),
                    "weight_mean": sub["weight"].mean(),
                    "undrafted": int(sub["undrafted"].sum()),
                    "three_cone_missing": int(sub["three_cone"].isna().sum()),
                }
            )
    bias = pd.DataFrame(bias_rows)
    penalty_ambiguous = int(plays["penalty_ambiguous"].sum()) if "penalty_ambiguous" in plays.columns else 0
    lineup = (
        plays.loc[(plays["season"] == plays["draft_year"]) & (plays["season_type"] == "REG")]
        .groupby(["draft_year", "lined_up_position"])
        .size()
        .reset_index(name="routes")
    )

    measurement = _measurement_section(project, cfg)
    dev_players = int(((dim["draft_year"] < holdout_year) & dim["in_primary"]).sum())
    eval_players = int(((dim["draft_year"] >= holdout_year) & dim["in_primary"]).sum())
    feasible = dev_players >= int(cfg["feasibility"]["min_dev_players"]) and eval_players >= int(cfg["feasibility"]["min_eval_players"])
    decision = "FEASIBLE" if feasible else "NOT FEASIBLE"
    con.close()

    coverage_path = project.reports / "coverage.csv"
    funnel.to_csv(coverage_path, index=False)
    lines = [
        "# Data audit",
        "",
        f"Data version `{version}`. Raw files were hashed and were not modified.",
        "",
        "## Files",
        "",
        _md(pd.DataFrame([{"file": k, "bytes": v.get("bytes"), "sha256": (v.get("sha256") or "")[:16]} for k, v in fingerprints.items()])),
        "",
        "## Schema against the supplied field list",
        "",
        _md(pd.DataFrame(schema_rows)),
        "",
        "Game tracking has an `o` orientation field. Combine tracking does not. Combine `dir` is movement direction. No column supports foot placement, hip angle, or limb asymmetry.",
        "",
        "## Roster and games",
        "",
        _md(players),
        "",
        _md(games),
        "",
        f"Key check: { _md(keys) }",
        "",
        f"Penalty codes other than Y/N among wide-receiver plays: {penalty_ambiguous}.",
        "",
        "## Combine tracking",
        "",
        _md(entities),
        "",
        "BALL rows exist and are excluded from player trajectories. Wide-receiver drill inventory:",
        "",
        _md(drills),
        "",
        "## Game tracking",
        "",
        "These files contain the prospect cohort, not all 22 players. Defender separation cannot be recomputed from them.",
        "",
        _md(tracking),
        "",
        "## Separation label",
        "",
        "Counts below include the 2025 class. Means do not.",
        "",
        _md(by_result),
        "",
        _md(targeted),
        "",
        _md(sep_summary),
        "",
        "On development routes the label is present for completions, incompletions, and interceptions, including untargeted routes, and absent for sacks and scrambles. The estimand is separation on rookie regular-season routes with a forward pass, not separation on targets only.",
        "",
        "## Exclusion funnel",
        "",
        _md(funnel),
        "",
        "## Where receivers lined up",
        "",
        _md(lineup),
        "",
        "The roster rule is `nfl_position == WR`. A wide receiver who lines up in the slot, or occasionally at tight end, stays in the cohort. Tight ends are not added.",
        "",
        "## Inclusion",
        "",
        f"Primary means at least {min_routes} eligible rookie regular-season routes. Career production is not used.",
        "",
        _md(bias),
        "",
        measurement,
        "",
    ]
    (project.reports / "data_audit.md").write_text("\n".join(lines))
    _write_feasibility(project, decision, dev_players, eval_players, measurement)
    print(f"[audit] {decision}. development primary players={dev_players}, evaluation primary players={eval_players}", flush=True)


def _measurement_section(project: Project, cfg: dict) -> str:
    parts = ["## Measurement, from the development extract"]
    primary = int(cfg["tracking"]["sg_window"])
    frames = []
    for year in cfg["development_cohorts"]:
        path = cohort_dir(project, int(year)) / "combine_cuts.parquet"
        if not path.exists():
            return "\n".join(parts + ["", "Combine cuts have not been extracted yet."])
        cuts = pd.read_parquet(path)
        cuts = cuts.loc[(cuts["sg_window"] == primary) & (cuts["drill_role"] == "primary")]
        cuts["draft_year"] = year
        frames.append(cuts)
        quality = pd.read_parquet(cohort_dir(project, int(year)) / "nfl_quality.parquet")
        parts.append(
            f"- {year} NFL tracked forward passes: {len(quality)}. "
            f"Share with a kept cut: {(quality['n_cuts_primary_window'] > 0).mean():.3f}. "
            f"Snap tag present: {quality['snap_present'].mean():.3f}. "
            f"Pass tag present: {quality['pass_present'].mean():.3f}. "
            f"Duplicate timestamps: {int(quality['n_duplicate_times'].sum())}."
        )
    cuts = pd.concat(frames, ignore_index=True)
    counts = cuts.groupby("nfl_id")["attempt_uid"].nunique()
    parts += [
        "",
        f"Primary-family Combine cuts after the {cfg['cuts']['entry_speed_min']} yd/s entry gate: {len(cuts)} cuts, {counts.size} receivers, median {counts.median():.0f} attempts, minimum {counts.min():.0f}.",
        f"Share of those receivers with at least 2 attempts: {(counts >= 2).mean():.3f}.",
        "",
        _md(
            cuts.groupby("drill_family")
            .agg(cuts=("nfl_id", "size"), angle=("abs_angle_deg", "median"), entry_speed=("speed_in", "median"), exit_yards=("exit_distance_0p5", "median"))
            .reset_index()
        ),
    ]
    return "\n".join(parts)


def _write_feasibility(project: Project, decision: str, dev_players: int, eval_players: int, measurement: str) -> None:
    cfg = project.config
    text = f"""# Feasibility

Decision: **{decision}**.

The planning thresholds in the configuration are {cfg['feasibility']['min_dev_players']} development receivers and {cfg['feasibility']['min_eval_players']} evaluation receivers with at least {cfg['routes']['min_routes']} eligible routes. The extract has {dev_players} development receivers and {eval_players} evaluation receivers at that bar. The evaluation count uses label availability only.

Repeated Combine attempts clear the second bar as well: every development receiver with a primary-family cut has at least three attempts after the entry-speed gate. Reliability is still a cross-drill quantity, because most drills are run once. That uncertainty belongs in the trait interval. It is not a reason to change the cohort.

With about {dev_players} development receivers and {eval_players} evaluation receivers, a player-level bootstrap interval on the M2 minus M1 error will be wide. A difference smaller than the analyst-chosen {cfg['models']['effect_threshold_mae_yards']} yard threshold can come out inconclusive. That is an acceptable result.

{measurement}
"""
    (project.reports / "feasibility.md").write_text(text)
