"""Cohort rules: rookie regular-season receiver routes and the exclusion funnel.

Eligibility is a function of roster position, game context, and label
availability. It does not depend on the tracking trait.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bdb2027.config import HOLDOUT_OUTCOME_COLUMNS, Project
from bdb2027.io import (
    COMBINE_RESULTS_TYPES,
    COMBINE_TRACK_TYPES,
    GAMES_TYPES,
    PLAYERS_TYPES,
    PLAY_TYPES,
    assert_outcomes_allowed,
    connect,
    csv_scan,
)

ROUTE_KEYS = ["game_id", "play_id", "nfl_id"]


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["true", "t", "1", "y", "yes"])


def prepare_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """Normalize types on a wide-receiver play table."""
    out = plays.copy()
    for col in ("game_id", "play_id", "nfl_id", "draft_year", "season", "week", "down"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in ("yards_to_go", "yardline_number", "cushion", "separation_at_pass_forward", "pass_length", "dropback_duration"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "target" in out.columns:
        out["target"] = _as_bool_series(out["target"])
    if "in_motion_at_ball_snap" in out.columns:
        motion = out["in_motion_at_ball_snap"]
        if motion.dtype != bool:
            text = motion.astype("string").str.strip().str.lower()
            out["motion"] = np.where(text.isna() | text.isin(["", "<na>", "nan", "none"]), "MISSING", np.where(text.isin(["true", "t", "1"]), "yes", "no"))
        else:
            out["motion"] = np.where(motion.isna(), "MISSING", np.where(motion, "yes", "no"))
    else:
        out["motion"] = "MISSING"
    out["coverage_man_zone"] = (
        out.get("team_coverage_man_zone", pd.Series(index=out.index, dtype="object"))
        .astype("string")
        .fillna("MISSING")
        .replace({"": "MISSING", "<NA>": "MISSING"})
    )
    out["offense_formation"] = out.get("offense_formation", pd.Series(index=out.index, dtype="object")).astype("string").fillna("MISSING")
    out["receiver_alignment"] = out.get("receiver_alignment", pd.Series(index=out.index, dtype="object")).astype("string").fillna("MISSING")
    out["route_ran"] = out.get("route_ran", pd.Series(index=out.index, dtype="object")).astype("string")
    penalty = out.get("play_nullified_by_penalty", pd.Series(index=out.index, dtype="object")).astype("string").str.upper()
    out["play_nullified_by_penalty"] = penalty
    out["penalty_ambiguous"] = penalty.notna() & ~penalty.isin(["Y", "N"])
    side = out.get("yardline_side", pd.Series(index=out.index, dtype="object")).astype("string")
    poss = out.get("possession_team", pd.Series(index=out.index, dtype="object")).astype("string")
    own = pd.Series(np.nan, index=out.index, dtype=float)
    known = side.notna() & poss.notna() & ~side.isin(["", "<NA>"]) & ~poss.isin(["", "<NA>"])
    own.loc[known] = (side.loc[known] == poss.loc[known]).astype(float)
    out["own_side"] = own
    out["down"] = out["down"].astype("string").replace({"<NA>": "MISSING"}).fillna("MISSING")
    return out


def route_masks(plays: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Sequential keep-masks. Each mask is applied on top of the previous ones."""
    routes = cfg["routes"]
    penalty = plays["play_nullified_by_penalty"].astype("string").str.upper()
    route = plays["route_ran"].astype("string")
    masks = {
        "wr_roster": plays["nfl_position"].astype("string") == cfg["position"] if "nfl_position" in plays.columns else pd.Series(True, index=plays.index),
        "regular_season": plays["season_type"].astype("string") == cfg["season_type"],
        "rookie_season": plays["season"].astype("Int64") == plays["draft_year"].astype("Int64"),
        "not_penalty": penalty.eq("N"),
        "has_route": route.notna() & ~route.isin(["", "<NA>", "nan"]),
        "not_rare": ~route.isin(routes["rare_drop"]),
        "not_excluded_family": ~route.isin(routes["excluded"]),
        "forward_pass": plays["pass_result"].astype("string").isin(routes["pass_results"]),
        "has_separation": plays["separation_at_pass_forward"].notna(),
    }
    return masks


def apply_funnel(plays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add exclusion flags. ``eligible`` is the primary analysis route."""
    out = plays.copy()
    keep = pd.Series(True, index=out.index)
    reason = pd.Series(pd.NA, index=out.index, dtype="string")
    for name, mask in route_masks(out, cfg).items():
        newly = keep & ~mask.fillna(False)
        reason = reason.mask(newly & reason.isna(), name)
        keep = keep & mask.fillna(False)
        out[f"pass_{name}"] = keep
    out["exclusion_reason"] = reason
    out["eligible"] = keep
    return out


def funnel_table(plays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Player, route, and game counts at each exclusion stage, by draft class."""
    flagged = apply_funnel(plays, cfg) if "eligible" not in plays.columns else plays
    stages = list(route_masks(plays, cfg))
    rows = []
    classes = sorted(flagged["draft_year"].dropna().unique())
    for draft_year in list(classes) + ["all"]:
        part = flagged if draft_year == "all" else flagged[flagged["draft_year"] == draft_year]
        current = pd.Series(True, index=part.index)
        base_players = int(part["nfl_id"].nunique())
        rows.append(_funnel_row(draft_year, "wr_plays_in_file", part, current, base_players))
        for stage in stages:
            current = part[f"pass_{stage}"] if f"pass_{stage}" in part.columns else current
            rows.append(_funnel_row(draft_year, stage, part, current, base_players))
    return pd.DataFrame(rows)


def _funnel_row(draft_year, stage, part, mask, base_players) -> dict:
    kept = part.loc[mask]
    return {
        "draft_year": draft_year,
        "stage": stage,
        "routes": int(mask.sum()) if len(part) else 0,
        "players": int(kept["nfl_id"].nunique()) if len(kept) else 0,
        "games": int(kept["game_id"].nunique()) if len(kept) else 0,
        "players_started": base_players,
    }


def load_wr_plays(project: Project, draft_years: list[int] | None = None) -> pd.DataFrame:
    """Wide-receiver player-play rows.

    Pass ``draft_years`` to keep held-out classes out of a development process.
    The returned table still includes preseason, postseason, and later-career
    rows so the exclusion funnel can count them.
    """
    con = connect(project)
    year_sql = ""
    if draft_years is not None:
        years = ", ".join(str(int(year)) for year in draft_years)
        year_sql = f" AND pl.draft_year IN ({years})"
    sql = f"""
        SELECT
            pp.game_id, pp.play_id, pp.nfl_id,
            pl.draft_year, pl.nfl_position,
            g.season, g.season_type, g.week,
            g.home_team_abbr, g.visitor_team_abbr,
            pp.team_abbr, pp.lined_up_position, pp.route_ran,
            pp.down, pp.yards_to_go, pp.yardline_side, pp.yardline_number, pp.possession_team,
            pp.play_direction, pp.offense_formation, pp.receiver_alignment,
            pp.team_coverage_man_zone, pp.team_coverage_type,
            CAST(pp.in_motion_at_ball_snap AS VARCHAR) AS in_motion_at_ball_snap,
            pp.play_nullified_by_penalty,
            pp.pass_result,
            CAST(pp.target AS VARCHAR) AS target,
            pp.cushion,
            pp.separation_at_pass_forward,
            pp.pass_length,
            pp.dropback_duration
        FROM {csv_scan(project.raw('player_play.csv'), PLAY_TYPES)} pp
        JOIN {csv_scan(project.raw('games.csv'), GAMES_TYPES)} g USING (game_id)
        JOIN {csv_scan(project.raw('players.csv'), PLAYERS_TYPES)} pl USING (nfl_id)
        WHERE pl.nfl_position = '{project.config['position']}'{year_sql}
    """
    frame = con.execute(sql).fetchdf()
    con.close()
    return prepare_plays(frame)


def load_player_dimension(project: Project) -> pd.DataFrame:
    """One row per wide receiver. No career-success columns and no names in the model frame.

    Names are returned in a separate column for the final profile only.
    """
    con = connect(project)
    sql = f"""
        SELECT
            pl.nfl_id, pl.display_name, pl.draft_year, pl.nfl_position, pl.birth_date,
            pl.college_name, pl.draft_round, pl.draft_overall_pick,
            cr.combine_position, cr.combine_height AS height, cr.combine_weight AS weight,
            cr.ten_yd_split, cr.forty, cr.vertical, cr.broad_jump, cr.three_cone, cr.short_shuttle,
            cr.ngs_athleticism_score, cr.ngs_college_production_score, cr.ngs_final_score,
            cd.combine_date
        FROM {csv_scan(project.raw('players.csv'), PLAYERS_TYPES)} pl
        LEFT JOIN {csv_scan(project.raw('combine_results.csv'), COMBINE_RESULTS_TYPES)} cr USING (nfl_id)
        LEFT JOIN (
            SELECT nfl_id, CAST(min(time) AS DATE) AS combine_date
            FROM {csv_scan(project.raw('combine_tracking.csv'), COMBINE_TRACK_TYPES)}
            WHERE entity_type = 'PLAYER'
            GROUP BY nfl_id
        ) cd USING (nfl_id)
        WHERE pl.nfl_position = '{project.config['position']}'
    """
    frame = con.execute(sql).fetchdf()
    con.close()
    frame["birth_date"] = pd.to_datetime(frame["birth_date"])
    frame["combine_date"] = pd.to_datetime(frame["combine_date"])
    month, day = project.config["conventional"]["age_fallback_md"].split("-")
    fallback = pd.to_datetime(
        frame["draft_year"].astype(int).astype(str) + f"-{int(month):02d}-{int(day):02d}"
    )
    frame["age_convention"] = np.where(frame["combine_date"].notna(), "combine_tracking_date", "march_1_draft_year")
    used = frame["combine_date"].fillna(fallback)
    frame["age"] = (used - frame["birth_date"]).dt.days / 365.25
    frame["undrafted"] = frame["draft_overall_pick"].isna().astype(float)
    frame["nfl_id"] = frame["nfl_id"].astype("int64")
    frame["draft_year"] = frame["draft_year"].astype("int64")
    return frame


def min_route_players(plays: pd.DataFrame, min_routes: int) -> set[int]:
    eligible = plays.loc[plays["eligible"]]
    counts = eligible.groupby("nfl_id").size()
    return set(int(i) for i in counts[counts >= min_routes].index)


def assert_no_cohort_overlap(train_ids, test_ids) -> None:
    overlap = set(map(int, train_ids)) & set(map(int, test_ids))
    if overlap:
        sample = sorted(overlap)[:8]
        raise ValueError(f"Train and evaluation cohorts overlap on {len(overlap)} players, e.g. {sample}")


def modeling_frame(plays: pd.DataFrame, allow_outcomes: bool, holdout_year: int) -> pd.DataFrame:
    """Eligible routes only. Held-out separation values are removed unless unlocked."""
    out = plays.loc[plays["eligible"]].copy()
    years = out["draft_year"].dropna().astype(int).unique()
    assert_outcomes_allowed(years, allow_outcomes, holdout_year)
    if not allow_outcomes:
        # Defense in depth: even if the assertion is bypassed by a caller that
        # filtered years first, blank any future class that slipped through.
        mask = out["draft_year"].astype("Int64") >= holdout_year
        for col in HOLDOUT_OUTCOME_COLUMNS:
            if col in out.columns:
                out.loc[mask, col] = np.nan
    return out
