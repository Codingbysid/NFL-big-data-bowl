"""DuckDB access, fingerprints, caches, and the held-out outcome guard."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import duckdb
import pandas as pd

from bdb2027.config import HOLDOUT_OUTCOME_COLUMNS, RAW_FILES, Project

DUCKDB_MEMORY = "3GB"
DUCKDB_THREADS = 4


def connect(project: Project) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order=false")
    tmp = project.tmp.as_posix()
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def csv_scan(path: Path, dtype_sql: str = "", extra: str = "") -> str:
    """SQL expression that scans one competition CSV with NA sentinels."""
    opts = ["header=true", "nullstr='NA'"]
    if dtype_sql:
        opts.append(f"types={{{dtype_sql}}}")
    if extra:
        opts.append(extra)
    return f"read_csv('{path.as_posix()}', {', '.join(opts)})"


PLAYERS_TYPES = "'nfl_id':'BIGINT','draft_year':'INTEGER','draft_round':'INTEGER','draft_pick_within_round':'INTEGER','draft_overall_pick':'INTEGER','birth_date':'DATE'"
COMBINE_RESULTS_TYPES = (
    "'nfl_id':'BIGINT','draft_year':'INTEGER',"
    "'combine_height':'DOUBLE','combine_weight':'DOUBLE','hand_size':'DOUBLE',"
    "'arm_length':'DOUBLE','wing_span':'DOUBLE','ten_yd_split':'DOUBLE','forty':'DOUBLE',"
    "'vertical':'DOUBLE','broad_jump':'DOUBLE','three_cone':'DOUBLE','short_shuttle':'DOUBLE',"
    "'bench_reps':'DOUBLE','ngs_athleticism_score':'DOUBLE',"
    "'ngs_college_production_score':'DOUBLE','ngs_final_score':'DOUBLE'"
)
COMBINE_TRACK_TYPES = (
    "'event_id':'VARCHAR','nfl_id':'BIGINT','draft_year':'INTEGER','attempt':'INTEGER',"
    "'time':'TIMESTAMP','x':'DOUBLE','y':'DOUBLE','s':'DOUBLE','a':'DOUBLE','dis':'DOUBLE','dir':'DOUBLE'"
)
GAMES_TYPES = "'game_id':'BIGINT','season':'INTEGER','week':'INTEGER'"
PLAY_TYPES = "'game_id':'BIGINT','play_id':'BIGINT','nfl_id':'BIGINT','down':'INTEGER','yards_to_go':'DOUBLE','yardline_number':'DOUBLE'"
TRACK_TYPES = (
    "'game_id':'BIGINT','play_id':'BIGINT','nfl_id':'BIGINT','time':'TIMESTAMP',"
    "'x':'DOUBLE','y':'DOUBLE','s':'DOUBLE','a':'DOUBLE','dis':'DOUBLE','o':'DOUBLE','dir':'DOUBLE','event':'VARCHAR'"
)


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    st = path.stat()
    return {
        "name": path.name,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "mtime_ns": st.st_mtime_ns,
    }


def cached_fingerprints(project: Project, refresh: bool = False) -> dict:
    """Hash raw files once. Reuse the cache when size and mtime are unchanged."""
    cache_path = project.reports / "fingerprints.json"
    cached = {}
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text())
    out = {}
    for name in RAW_FILES:
        path = project.data_root / name
        if not path.exists():
            out[name] = {"name": name, "missing": True}
            continue
        st = path.stat()
        prev = cached.get(name) or {}
        if (
            not prev.get("missing")
            and prev.get("bytes") == st.st_size
            and prev.get("mtime_ns") == st.st_mtime_ns
            and prev.get("sha256")
        ):
            out[name] = prev
            continue
        out[name] = fingerprint(path)
    cache_path.write_text(json.dumps(out, indent=2))
    return out


def data_version(fingerprints: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(fingerprints):
        digest.update(name.encode())
        digest.update(str(fingerprints[name].get("sha256", "missing")).encode())
    return digest.hexdigest()[:16]


class HoldoutGuard(Exception):
    """Raised when held-out outcomes are requested before evaluation."""


def assert_outcomes_allowed(draft_years, allow_outcomes: bool, holdout_year: int = 2025) -> None:
    years = {int(y) for y in draft_years}
    if any(y >= holdout_year for y in years) and not allow_outcomes:
        raise HoldoutGuard(
            f"Outcome values for draft_year >= {holdout_year} are locked until "
            "evaluate-heldout. Availability counts are allowed; means, distributions, "
            "associations, and model errors are not."
        )


def strip_holdout_outcomes(frame: pd.DataFrame, draft_year_col: str, holdout_year: int, allow_outcomes: bool) -> pd.DataFrame:
    if allow_outcomes or draft_year_col not in frame.columns:
        return frame
    out = frame.copy()
    mask = out[draft_year_col].astype("Int64") >= holdout_year
    for col in HOLDOUT_OUTCOME_COLUMNS:
        if col in out.columns:
            out.loc[mask, col] = pd.NA
    return out


def stage_meta_path(directory: Path) -> Path:
    return directory / "_stage_meta.json"


def write_stage_meta(directory: Path, *, sources: dict, config_hash: str, code_hash: str, extra: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "sources": sources,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra": extra or {},
    }
    stage_meta_path(directory).write_text(json.dumps(payload, indent=2))


def stage_cache_valid(directory: Path, *, sources: dict, config_hash: str, code_hash: str, require_files: list[str] | None = None) -> bool:
    path = stage_meta_path(directory)
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    if meta.get("config_hash") != config_hash or meta.get("code_hash") != code_hash:
        return False
    if meta.get("sources") != sources:
        return False
    for name in require_files or []:
        if not (directory / name).exists():
            return False
    return True


def source_token(fingerprints: dict, names: list[str]) -> dict:
    return {name: fingerprints.get(name, {}).get("sha256") for name in names}


def append_runtime(project: Project, stage: str, seconds: float, peak_rss_bytes: int, detail: str = "") -> None:
    path = project.reports / "runtime.csv"
    header = not path.exists()
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},{stage},{seconds:.2f},{peak_rss_bytes},{detail}\n"
    with open(path, "a") as handle:
        if header:
            handle.write("timestamp,stage,seconds,peak_rss_bytes,detail\n")
        handle.write(line)


def append_experiment(project: Project, stage: str, name: str, detail: str, status: str, prespecified: bool) -> None:
    path = project.reports / "experiment_log.csv"
    header = not path.exists()
    # Quote detail so commas survive.
    safe = '"' + detail.replace('"', "'") + '"'
    line = (
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},{stage},{name},"
        f"{safe},{status},{str(bool(prespecified)).lower()}\n"
    )
    with open(path, "a") as handle:
        if header:
            handle.write("timestamp,stage,name,detail,status,prespecified\n")
        handle.write(line)


def assert_unique_keys(frame: pd.DataFrame, keys: list[str], what: str) -> None:
    if frame.empty:
        return
    duplicated = frame.duplicated(keys).sum()
    if int(duplicated):
        raise ValueError(f"{what} has {int(duplicated)} duplicated rows on {keys}")


def merge_frames_to_plays(frames: pd.DataFrame, plays: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Attach play attributes to frames without replicating the inverse join.

    The returned frame has one row per tracking frame. Callers that need
    play-level outcomes must aggregate back to ``keys`` first.
    """
    assert_unique_keys(plays, keys, "plays")
    merged = frames.merge(plays, on=keys, how="left", validate="many_to_one")
    if len(merged) != len(frames):
        raise ValueError(
            f"Frame/play merge changed row count from {len(frames)} to {len(merged)}. "
            "Tracking frames must not be fanned out."
        )
    return merged
