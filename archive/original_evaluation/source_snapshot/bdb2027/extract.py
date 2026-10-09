"""Stream competition tracking into per-cohort Parquet and detect cuts.

Game files are filtered in DuckDB before they reach pandas. Held-out
separation values are not required for detection and are omitted unless
``allow_outcomes`` is set.
"""

from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bdb2027.cohort import apply_funnel, funnel_table, load_player_dimension, load_wr_plays
from bdb2027.config import Project, code_hash, config_hash, drill_maps, drill_side
from bdb2027.cuts import detect_cuts, detection_params
from bdb2027.io import (
    COMBINE_TRACK_TYPES,
    PLAYERS_TYPES,
    TRACK_TYPES,
    append_experiment,
    append_runtime,
    connect,
    csv_scan,
    source_token,
    stage_cache_valid,
    write_stage_meta,
    cached_fingerprints,
)

def _seconds(series: pd.Series) -> np.ndarray:
    values = pd.to_datetime(series).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return values / 1e9


def _stage(project: Project, name: str):
    class _Stage:
        def __enter__(self):
            self.t0 = time.perf_counter()
            print(f"[{name}] start", flush=True)
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.perf_counter() - self.t0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            append_runtime(project, name, elapsed, rss, "error" if exc_type else "ok")
            print(f"[{name}] {'error' if exc_type else 'ok'} {elapsed:.1f}s peak_rss={rss/1e9:.2f}GB", flush=True)
            return False

    return _Stage()


def cohort_dir(project: Project, cohort: int) -> Path:
    path = project.interim / f"cohort_{cohort}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def windows_of(cfg: dict) -> list[int]:
    primary = int(cfg["tracking"]["sg_window"])
    extra = [int(w) for w in cfg["tracking"]["sg_windows_sensitivity"]]
    ordered = [primary] + [w for w in extra if w != primary]
    return ordered


def export_combine_frames(project: Project, cohorts: list[int], destination: Path) -> None:
    con = connect(project)
    years = ", ".join(str(int(y)) for y in cohorts)
    sql = f"""
        COPY (
            SELECT ct.draft_year, ct.event_id, ct.nfl_id, ct.time, ct.drill_type, ct.drill_name,
                   ct.attempt, ct.x, ct.y, ct.s, ct.a, ct.dis, ct.dir
            FROM {csv_scan(project.raw('combine_tracking.csv'), COMBINE_TRACK_TYPES)} ct
            JOIN {csv_scan(project.raw('players.csv'), PLAYERS_TYPES)} pl USING (nfl_id)
            WHERE pl.nfl_position = '{project.config['position']}'
              AND ct.entity_type = 'PLAYER'
              AND ct.drill_type = 'SKILL_DRILLS_WR'
              AND ct.draft_year IN ({years})
        ) TO '{destination.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(sql)
    con.close()


def export_nfl_frames(project: Project, keys: pd.DataFrame, cohort: int, destination: Path) -> None:
    key_path = destination.parent / f"_keys_{cohort}.parquet"
    slim = keys[["game_id", "play_id", "nfl_id"]].drop_duplicates()
    slim.to_parquet(key_path, index=False)
    con = connect(project)
    tracking = project.raw(f"game_tracking_{cohort}.csv")
    sql = f"""
        COPY (
            SELECT gt.game_id, gt.play_id, gt.nfl_id, gt.time, gt.x, gt.y, gt.s, gt.a,
                   gt.dis, gt.o, gt.dir, gt.event
            FROM {csv_scan(tracking, TRACK_TYPES)} gt
            SEMI JOIN read_parquet('{key_path.as_posix()}') e
              ON gt.game_id = e.game_id AND gt.play_id = e.play_id AND gt.nfl_id = e.nfl_id
        ) TO '{destination.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(sql)
    con.close()


def _cut_rows(meta: dict, cuts, window: int) -> list[dict]:
    rows = []
    for cut in cuts:
        row = meta | cut.as_dict()
        row["sg_window"] = int(window)
        rows.append(row)
    return rows


def detect_combine(frames: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_map, role_map = drill_maps(cfg)
    cut_rows: list[dict] = []
    quality: list[dict] = []
    group_cols = ["nfl_id", "event_id", "attempt", "drill_name", "draft_year"]
    frames = frames.sort_values(group_cols + ["time"])
    n_groups = 0
    for key, group in frames.groupby(group_cols, sort=False):
        n_groups += 1
        nfl_id, event_id, attempt, drill_name, draft_year = key
        t = _seconds(group["time"])
        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        meta = {
            "source": "combine",
            "nfl_id": int(nfl_id),
            "draft_year": int(draft_year),
            "event_id": str(event_id),
            "attempt": int(attempt),
            "drill_name": str(drill_name),
            "drill_family": family_map.get(str(drill_name), "UNLISTED"),
            "drill_role": role_map.get(str(drill_name), "unlisted"),
            "drill_side": drill_side(str(drill_name)),
            "game_id": pd.NA,
            "play_id": pd.NA,
            "route_ran": pd.NA,
            "attempt_uid": f"{int(nfl_id)}|{event_id}|{int(attempt)}|{drill_name}",
        }
        info = None
        n_primary_window = 0
        for window in windows_of(cfg):
            found, info = detect_cuts(t, x, y, params=detection_params(cfg, window), require_pass_event=False)
            if window == int(cfg["tracking"]["sg_window"]):
                n_primary_window = len(found)
            cut_rows.extend(_cut_rows(meta, found, window))
        quality.append(
            meta
            | {
                "n_frames": info.n_frames if info else len(group),
                "n_duplicate_times": info.n_duplicate_times if info else 0,
                "n_jumps": info.n_jumps if info else 0,
                "n_heading_repairs": info.n_heading_repairs if info else 0,
                "duration_s": info.duration_s if info else np.nan,
                "n_cuts_primary_window": n_primary_window,
                "snap_present": False,
                "pass_present": False,
            }
        )
        if n_groups % 500 == 0:
            print(f"  combine attempts processed: {n_groups}", flush=True)
    return pd.DataFrame(cut_rows), pd.DataFrame(quality)


def detect_nfl(frames: pd.DataFrame, routes: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut_rows: list[dict] = []
    quality: list[dict] = []
    routes = routes.drop_duplicates(["game_id", "play_id", "nfl_id"]).copy()
    frames = frames.copy()
    for col in ("game_id", "play_id", "nfl_id"):
        routes[col] = routes[col].astype("int64")
        frames[col] = frames[col].astype("int64")
    lookup = routes.set_index(["game_id", "play_id", "nfl_id"], drop=False)
    frames = frames.sort_values(["game_id", "play_id", "nfl_id", "time"])
    n_groups = 0
    for key, group in frames.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        n_groups += 1
        game_id, play_id, nfl_id = (int(v) for v in key)
        route = lookup.loc[(game_id, play_id, nfl_id)]
        if isinstance(route, pd.DataFrame):
            route = route.iloc[0]
        t = _seconds(group["time"])
        events = group["event"].fillna("").astype(str).to_numpy()
        meta = {
            "source": "nfl",
            "nfl_id": nfl_id,
            "draft_year": int(route["draft_year"]),
            "event_id": pd.NA,
            "attempt": 1,
            "drill_name": f"NFL_{route['route_ran']}",
            "drill_family": str(route["route_ran"]),
            "drill_role": "primary",
            "drill_side": "play",
            "game_id": game_id,
            "play_id": play_id,
            "route_ran": str(route["route_ran"]),
            "attempt_uid": f"{nfl_id}|{game_id}|{play_id}",
        }
        info = None
        n_primary_window = 0
        for window in windows_of(cfg):
            found, info = detect_cuts(
                t,
                group["x"].to_numpy(dtype=float),
                group["y"].to_numpy(dtype=float),
                events=events,
                params=detection_params(cfg, window),
                require_pass_event=True,
            )
            if window == int(cfg["tracking"]["sg_window"]):
                n_primary_window = len(found)
            cut_rows.extend(_cut_rows(meta, found, window))
        quality.append(
            meta
            | {
                "n_frames": info.n_frames if info else len(group),
                "n_duplicate_times": info.n_duplicate_times if info else 0,
                "n_jumps": info.n_jumps if info else 0,
                "n_heading_repairs": info.n_heading_repairs if info else 0,
                "duration_s": info.duration_s if info else np.nan,
                "n_cuts_primary_window": n_primary_window,
                "snap_present": bool(info and np.isfinite(info.snap_t)),
                "pass_present": bool(info and np.isfinite(info.pass_t)),
            }
        )
        if n_groups % 2000 == 0:
            print(f"  nfl routes processed: {n_groups}", flush=True)
    return pd.DataFrame(cut_rows), pd.DataFrame(quality)


def frame_keys(plays: pd.DataFrame, cohort: int) -> pd.DataFrame:
    """Rookie regular-season forward passes. Screens stay in so exclusion is counted."""
    mask = (
        (plays["draft_year"] == cohort)
        & (plays["season"] == cohort)
        & (plays["season_type"] == "REG")
        & (plays["play_nullified_by_penalty"] == "N")
        & plays["pass_result"].isin(["C", "I", "IN"])
        & plays["route_ran"].notna()
    )
    return plays.loc[mask].copy()


def extract(project: Project, cohorts: list[int], allow_outcomes: bool = False) -> None:
    holdout = int(project.config["holdout_min_draft_year"])
    if any(int(c) >= holdout for c in cohorts) and not allow_outcomes:
        raise PermissionError(
            f"Extracting draft_year >= {holdout} requires evaluate-heldout. "
            "Schema and label-availability counts belong in the audit, not in a development extract."
        )
    fingerprints = cached_fingerprints(project)
    with _stage(project, "extract:" + ",".join(map(str, cohorts))):
        plays = apply_funnel(load_wr_plays(project, draft_years=cohorts), project.config)
        for cohort in cohorts:
            out = cohort_dir(project, int(cohort))
            token = source_token(
                fingerprints,
                ["players.csv", "combine_results.csv", "combine_tracking.csv", "player_play.csv", "games.csv", f"game_tracking_{cohort}.csv"],
            )
            token["allow_outcomes"] = "1" if allow_outcomes else "0"
            token["cohort"] = str(cohort)
            if stage_cache_valid(
                out,
                sources=token,
                config_hash=config_hash(project.config),
                code_hash=code_hash(),
                require_files=["plays.parquet", "combine_cuts.parquet", "nfl_cuts.parquet"],
            ):
                print(f"[extract] cohort {cohort} cache hit", flush=True)
                continue
            keys = frame_keys(plays, int(cohort))
            play_path = out / "plays.parquet"
            save_plays = keys.copy()
            if not allow_outcomes:
                # Development extracts never persist held-out labels. This cohort
                # is below the holdout year when allow_outcomes is false.
                pass
            save_plays.to_parquet(play_path, index=False)
            funnel_table(plays.loc[plays["draft_year"] == cohort], project.config).to_csv(out / "funnel.csv", index=False)
            combine_path = out / "combine_frames.parquet"
            export_combine_frames(project, [int(cohort)], combine_path)
            combine_frames = pd.read_parquet(combine_path)
            combine_cuts, combine_quality = detect_combine(combine_frames, project.config)
            combine_cuts.to_parquet(out / "combine_cuts.parquet", index=False)
            combine_quality.to_parquet(out / "combine_quality.parquet", index=False)
            nfl_path = out / "nfl_frames.parquet"
            export_nfl_frames(project, keys, int(cohort), nfl_path)
            nfl_frames = pd.read_parquet(nfl_path)
            covered = set(map(tuple, nfl_frames[["game_id", "play_id", "nfl_id"]].drop_duplicates().to_numpy()))
            key_tuples = list(map(tuple, keys[["game_id", "play_id", "nfl_id"]].to_numpy()))
            n_missing = sum(1 for item in key_tuples if item not in covered)
            (out / "tracking_gap.txt").write_text(f"forward_pass_routes={len(key_tuples)}\nroutes_without_tracking={n_missing}\n")
            if nfl_frames.empty:
                pd.DataFrame().to_parquet(out / "nfl_cuts.parquet", index=False)
                pd.DataFrame().to_parquet(out / "nfl_quality.parquet", index=False)
            else:
                nfl_cuts, nfl_quality = detect_nfl(nfl_frames, keys, project.config)
                nfl_cuts.to_parquet(out / "nfl_cuts.parquet", index=False)
                nfl_quality.to_parquet(out / "nfl_quality.parquet", index=False)
            write_stage_meta(
                out,
                sources=token,
                config_hash=config_hash(project.config),
                code_hash=code_hash(),
                extra={"cohort": int(cohort), "allow_outcomes": allow_outcomes},
            )
            print(
                f"[extract] cohort {cohort}: combine cuts {len(combine_cuts)} "
                f"nfl frames {len(nfl_frames)} nfl cuts {0 if nfl_frames.empty else len(nfl_cuts)}",
                flush=True,
            )
        players = load_player_dimension(project)
        players.to_parquet(project.interim / "wr_players.parquet", index=False)
        append_experiment(
            project,
            "extract",
            "cohorts",
            f"extracted {cohorts}; allow_outcomes={allow_outcomes}",
            "ok",
            True,
        )
