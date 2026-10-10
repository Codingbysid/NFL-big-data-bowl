"""Development study: forty early-transit time and rookie get-off.

2025 get-off values are refused. The historical receiver outcome list is
not the guard for this endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from bdb2027.config import Project, load_config

SPEC_PATH = "config/getoff_v1.yaml"


def load_spec(root: Path | None = None) -> dict:
    root = root or Project().root
    return load_config(Path(root) / SPEC_PATH)


def _con(root: Path) -> duckdb.DuckDBPyConnection:
    data = root / "nfl-big-data-bowl-2027"
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE VIEW players AS SELECT * FROM read_csv('{(data / "players.csv").as_posix()}', header=true, nullstr='NA',
          types={{'nfl_id':'BIGINT','draft_year':'INTEGER'}});
        CREATE VIEW games AS SELECT * FROM read_csv('{(data / "games.csv").as_posix()}', header=true, nullstr='NA',
          types={{'game_id':'BIGINT','season':'INTEGER'}});
        CREATE VIEW plays AS SELECT * FROM read_csv('{(data / "player_play.csv").as_posix()}', header=true, nullstr='NA',
          types={{'game_id':'BIGINT','play_id':'BIGINT','nfl_id':'BIGINT','player_get_off':'DOUBLE',
                  'yardline_number':'DOUBLE'}});
        CREATE VIEW results AS SELECT * FROM read_csv('{(data / "combine_results.csv").as_posix()}', header=true, nullstr='NA',
          types={{'nfl_id':'BIGINT','ten_yd_split':'DOUBLE','forty':'DOUBLE','combine_weight':'DOUBLE'}});
        CREATE VIEW comb AS SELECT * FROM read_csv('{(data / "combine_tracking.csv").as_posix()}', header=true, nullstr='NA',
          types={{'nfl_id':'BIGINT','draft_year':'INTEGER','attempt':'INTEGER','x':'DOUBLE','y':'DOUBLE','s':'DOUBLE','time':'TIMESTAMP'}});
        """
    )
    return con


def assert_development_years(years: list[int]) -> None:
    if any(int(year) >= 2025 for year in years):
        raise PermissionError("player_get_off for 2025 is closed. This study accepts 2023 and 2024 only.")


def measure_attempt(frame: pd.DataFrame, spec: dict, origin_shift: float = 0.0) -> dict:
    """Time from 1 yard to 5 yards along a verified forty start.

    A constant added to every timestamp does not change the result.
    Translating or reflecting the path, with the running direction
    recomputed, does not change the result. An incomplete crossing is missing.
    """
    cfg = spec["combine_feature"]
    out = {
        "valid": False,
        "reason": "",
        "early_transit_1_5_s": np.nan,
        "t_at_1": np.nan,
        "t_at_5": np.nan,
        "origin_x": np.nan,
        "origin_y": np.nan,
        "x_at_1": np.nan,
        "y_at_1": np.nan,
        "x_at_5": np.nan,
        "y_at_5": np.nan,
        "long_axis": "",
        "n_stationary": 0,
        "n_duplicate_times": 0,
        "gap_flag": False,
        "jump_flag": False,
        "ambiguous_start": False,
    }
    work = frame.sort_values("time").copy()
    work["time"] = pd.to_datetime(work["time"])
    before = len(work)
    work = work.drop_duplicates("time", keep="first")
    out["n_duplicate_times"] = int(before - len(work))
    if len(work) < 8:
        out["reason"] = "incomplete_attempt"
        return out
    t = (work["time"] - work["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    x = work["x"].to_numpy(dtype=float)
    y = work["y"].to_numpy(dtype=float)
    speed = work["s"].to_numpy(dtype=float)
    dt = np.diff(t)
    step = np.hypot(np.diff(x), np.diff(y))
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = step / np.maximum(dt, 1e-9)
    if np.any((dt > 0) & (implied > float(cfg["max_step_speed"]))):
        out["jump_flag"] = True
        out["reason"] = "coordinate_jump"
        return out
    x_span = float(np.nanmax(x) - np.nanmin(x))
    y_span = float(np.nanmax(y) - np.nanmin(y))
    long_axis = "x" if x_span >= y_span else "y"
    long = x if long_axis == "x" else y
    span = x_span if long_axis == "x" else y_span
    out["long_axis"] = long_axis
    if not (float(cfg["min_long_span_yards"]) <= span <= float(cfg["max_long_span_yards"])):
        out["reason"] = "implausible_span"
        return out
    slow = speed < float(cfg["stationary_speed_max"])
    n_stat = 0
    for flag in slow:
        if not flag:
            break
        n_stat += 1
    out["n_stationary"] = int(n_stat)
    if n_stat < int(cfg["min_stationary_frames"]):
        out["ambiguous_start"] = True
        out["reason"] = "ambiguous_start"
        return out
    origin_x = float(np.median(x[:n_stat]))
    origin_y = float(np.median(y[:n_stat]))
    origin_long = origin_x if long_axis == "x" else origin_y
    direction = np.sign(float(long[-1] - origin_long))
    if direction == 0:
        out["reason"] = "ambiguous_start"
        out["ambiguous_start"] = True
        return out
    origin_long = origin_long + direction * float(origin_shift)
    if long_axis == "x":
        origin_x = origin_long
    else:
        origin_y = origin_long
    out["origin_x"] = origin_x
    out["origin_y"] = origin_y
    distance = (long - origin_long) * direction
    t1, gap1, p1 = _crossing(t, distance, x, y, float(cfg["near_yard"]), float(cfg["max_gap_s"]))
    t5, gap5, p5 = _crossing(t, distance, x, y, float(cfg["far_yard"]), float(cfg["max_gap_s"]))
    out["gap_flag"] = bool(gap1 or gap5)
    if not np.isfinite(t1) or not np.isfinite(t5):
        out["reason"] = "incomplete_crossing" if not out["gap_flag"] else "gap_at_crossing"
        return out
    transit = float(t5 - t1)
    out["t_at_1"] = t1
    out["t_at_5"] = t5
    out["x_at_1"], out["y_at_1"] = p1
    out["x_at_5"], out["y_at_5"] = p5
    if not (float(cfg["transit_min_s"]) <= transit <= float(cfg["transit_max_s"])):
        out["reason"] = "transit_out_of_range"
        out["early_transit_1_5_s"] = transit
        return out
    out["early_transit_1_5_s"] = transit
    out["valid"] = True
    out["reason"] = "ok"
    return out


def _crossing(t, distance, x, y, target: float, max_gap: float):
    for i in range(1, len(distance)):
        before, after = distance[i - 1], distance[i]
        if not np.isfinite(before) or not np.isfinite(after):
            continue
        if before <= target <= after and after > before:
            if t[i] - t[i - 1] > max_gap:
                return float("nan"), True, (float("nan"), float("nan"))
            weight = 0.0 if after == before else (target - before) / (after - before)
            tx = float(x[i - 1] + weight * (x[i] - x[i - 1]))
            ty = float(y[i - 1] + weight * (y[i] - y[i - 1]))
            return float(t[i - 1] + weight * (t[i] - t[i - 1])), False, (tx, ty)
    return float("nan"), False, (float("nan"), float("nan"))


def extract_attempts(root: Path, spec: dict) -> pd.DataFrame:
    assert_development_years(spec["cohorts"])
    years = ",".join(str(int(y)) for y in spec["cohorts"])
    positions = ",".join(f"'{p}'" for p in spec["population"]["positions"])
    frame = _con(root).execute(
        f"""
        SELECT c.nfl_id, c.draft_year, p.nfl_position, c.event_id, c.attempt, c.time, c.x, c.y, c.s
        FROM comb c
        JOIN players p USING (nfl_id)
        WHERE c.drill_type = '{spec["combine_feature"]["drill_type"]}'
          AND c.draft_year IN ({years})
          AND p.nfl_position IN ({positions})
        ORDER BY c.nfl_id, c.event_id, c.attempt, c.time
        """
    ).df()
    rows = []
    for keys, part in frame.groupby(["nfl_id", "draft_year", "nfl_position", "event_id", "attempt"], sort=False):
        measured = measure_attempt(part, spec)
        shift = float(spec["combine_feature"]["origin_shift_sensitivity_yards"])
        forward = measure_attempt(part, spec, origin_shift=shift)
        backward = measure_attempt(part, spec, origin_shift=-shift)
        rows.append(
            {
                "nfl_id": int(keys[0]),
                "draft_year": int(keys[1]),
                "nfl_position": keys[2],
                "event_id": keys[3],
                "attempt": int(keys[4]),
                "attempt_uid": f"{keys[0]}|{keys[3]}|{keys[4]}",
                **measured,
                "transit_origin_plus": forward["early_transit_1_5_s"],
                "transit_origin_minus": backward["early_transit_1_5_s"],
            }
        )
    out = pd.DataFrame(rows)
    if out["attempt_uid"].duplicated().any():
        raise ValueError("Duplicate Combine attempt keys")
    return out


def _line_of_scrimmage(side, number, possession, direction) -> float:
    own = float(number) if side == possession else 100.0 - float(number)
    if str(direction).lower() == "right":
        return 10.0 + own
    if str(direction).lower() == "left":
        return 110.0 - own
    return float("nan")


def load_getoff(root: Path, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return play rows and an availability audit. 2025 is rejected up front."""
    assert_development_years(spec["cohorts"])
    years = ",".join(str(int(y)) for y in spec["cohorts"])
    positions = ",".join(f"'{p}'" for p in spec["population"]["positions"])
    plays = _con(root).execute(
        f"""
        SELECT pl.game_id, pl.play_id, pl.nfl_id, p.draft_year, p.nfl_position,
               pl.player_get_off, pl.pass_result, pl.lined_up_position,
               pl.yardline_number, pl.yardline_side, pl.possession_team, pl.play_direction,
               r.ten_yd_split, r.combine_weight
        FROM plays pl
        JOIN players p USING (nfl_id)
        JOIN games g USING (game_id)
        LEFT JOIN results r USING (nfl_id)
        WHERE g.season = p.draft_year
          AND g.season_type = '{spec["population"]["season_type"]}'
          AND p.draft_year IN ({years})
          AND p.nfl_position IN ({positions})
        """
    ).df()
    if plays.duplicated(["game_id", "play_id", "nfl_id"]).any():
        raise ValueError("Duplicate player-play keys")
    audit = (
        plays.assign(play_type=plays["pass_result"].fillna("none"))
        .groupby(["draft_year", "nfl_position", "play_type"], dropna=False)
        .agg(plays=("play_id", "size"), getoff_present=("player_get_off", "count"))
        .reset_index()
    )
    return plays, audit


def snap_depth(root: Path, plays: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Depth at the snap frame only. Later movement is not used."""
    keys = plays[["game_id", "play_id", "nfl_id", "draft_year", "play_direction", "yardline_number", "yardline_side", "possession_team"]].drop_duplicates()
    years = sorted(int(y) for y in keys["draft_year"].unique())
    frames = []
    for year in years:
        path = root / "nfl-big-data-bowl-2027" / f"game_tracking_{year}.csv"
        year_keys = keys.loc[keys["draft_year"].eq(year), ["game_id", "play_id", "nfl_id"]]
        if year_keys.empty:
            continue
        con = duckdb.connect()
        con.register("wanted", year_keys)
        snap = con.execute(
            f"""
            SELECT t.game_id, t.play_id, t.nfl_id, t.x, t.y
            FROM read_csv('{path.as_posix()}', header=true, nullstr='NA',
              types={{'game_id':'BIGINT','play_id':'BIGINT','nfl_id':'BIGINT','x':'DOUBLE','y':'DOUBLE'}}) t
            JOIN wanted w USING (game_id, play_id, nfl_id)
            WHERE t.event = 'ball_snap'
            """
        ).df()
        frames.append(snap)
    snap = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["game_id", "play_id", "nfl_id", "x", "y"])
    merged = keys.merge(snap, on=["game_id", "play_id", "nfl_id"], how="left")
    los = [
        _line_of_scrimmage(side, number, poss, direction)
        for side, number, poss, direction in zip(merged["yardline_side"], merged["yardline_number"], merged["possession_team"], merged["play_direction"])
    ]
    merged["los"] = los
    depth = np.full(len(merged), np.nan)
    right = merged["play_direction"].astype(str).str.lower().eq("right")
    left = merged["play_direction"].astype(str).str.lower().eq("left")
    depth[right.to_numpy()] = merged.loc[right, "x"].to_numpy() - merged.loc[right, "los"].to_numpy()
    depth[left.to_numpy()] = merged.loc[left, "los"].to_numpy() - merged.loc[left, "x"].to_numpy()
    cap = float(spec["context"]["max_plausible_depth_yards"])
    depth[(depth < 0) | (depth > cap)] = np.nan
    merged["start_depth"] = depth
    return merged


def build_players(attempts: pd.DataFrame, plays: pd.DataFrame, depth: pd.DataFrame, spec: dict) -> pd.DataFrame:
    minimum = int(spec["population"]["min_getoff_observations"])
    observed = plays.loc[plays["player_get_off"].notna()].copy()
    counts = observed.groupby("nfl_id").agg(
        n_getoff=("player_get_off", "size"),
        median_getoff=("player_get_off", "median"),
        draft_year=("draft_year", "first"),
        nfl_position=("nfl_position", "first"),
        ten_yd_split=("ten_yd_split", "first"),
        combine_weight=("combine_weight", "first"),
    )
    aligned = plays.loc[plays["lined_up_position"].isin([spec["context"]["edge_label"], spec["context"]["interior_label"]])]
    edge = aligned.groupby("nfl_id")["lined_up_position"].apply(lambda s: float(np.mean(s.eq(spec["context"]["edge_label"]))))
    counts["edge_play_share"] = edge
    counts["n_alignment_plays"] = aligned.groupby("nfl_id").size()
    depth_med = depth.groupby("nfl_id")["start_depth"].median()
    depth_n = depth.groupby("nfl_id")["start_depth"].apply(lambda s: int(np.isfinite(s).sum()))
    counts["median_start_depth"] = depth_med
    counts["n_valid_depth"] = depth_n
    transit = attempts.loc[attempts["valid"]].groupby("nfl_id")["early_transit_1_5_s"].median()
    n_valid = attempts.groupby("nfl_id")["valid"].sum()
    n_attempts = attempts.groupby("nfl_id").size()
    counts["early_transit_1_5_s"] = transit
    counts["n_valid_transit_attempts"] = n_valid
    counts["n_forty_attempts"] = n_attempts
    counts["transit_origin_plus"] = attempts.loc[attempts["transit_origin_plus"].notna()].groupby("nfl_id")["transit_origin_plus"].median()
    counts["transit_origin_minus"] = attempts.loc[attempts["transit_origin_minus"].notna()].groupby("nfl_id")["transit_origin_minus"].median()
    roster = plays.groupby("nfl_id").agg(draft_year=("draft_year", "first"), nfl_position=("nfl_position", "first"), n_plays=("play_id", "size"))
    ledger = roster.join(counts.drop(columns=["draft_year", "nfl_position"]), how="left")
    reasons = []
    keep = []
    for nfl_id, row in ledger.iterrows():
        reason = ""
        if not np.isfinite(row.get("n_getoff", np.nan)) or row["n_getoff"] < minimum:
            reason = "fewer_than_8_getoff"
        elif not np.isfinite(row.get("ten_yd_split", np.nan)):
            reason = "missing_official_split"
        elif not np.isfinite(row.get("early_transit_1_5_s", np.nan)):
            reason = "missing_transit"
        reasons.append(reason)
        keep.append(reason == "")
    ledger["exclude_reason"] = reasons
    ledger["in_primary"] = keep
    ledger.index.name = "nfl_id"
    return ledger.reset_index()


def repeatability(attempts: pd.DataFrame) -> dict:
    valid = attempts.loc[attempts["valid"]].sort_values(["nfl_id", "attempt"])
    pairs = []
    for _nfl, part in valid.groupby("nfl_id"):
        if len(part) < 2:
            continue
        first, second = part.iloc[0], part.iloc[1]
        pairs.append(float(second["early_transit_1_5_s"] - first["early_transit_1_5_s"]))
    pair = np.asarray(pairs, dtype=float)
    groups = [g["early_transit_1_5_s"].to_numpy(dtype=float) for _, g in valid.groupby("nfl_id") if len(g) >= 2]
    icc = _icc(groups)
    plus = attempts.loc[attempts["valid"] & attempts["transit_origin_plus"].notna()]
    return {
        "players_with_two_valid_attempts": int(len(pair)),
        "mean_second_minus_first": float(np.mean(pair)) if len(pair) else None,
        "sd_second_minus_first": float(np.std(pair, ddof=1)) if len(pair) > 1 else None,
        "icc": icc,
        "median_abs_change_when_origin_moves_0_25_forward": float((plus["transit_origin_plus"] - plus["early_transit_1_5_s"]).abs().median()) if len(plus) else None,
    }


def _icc(groups: list[np.ndarray]) -> dict:
    if len(groups) < 3:
        return {"n_players": len(groups), "icc": None}
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    grand = np.average(means, weights=ns)
    ssw = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    msw = ssw / (ns.sum() - len(groups))
    ssb = float(np.sum(ns * (means - grand) ** 2))
    msb = ssb / (len(groups) - 1)
    n0 = (ns.sum() - np.sum(ns**2) / ns.sum()) / (len(groups) - 1)
    denom = msb + (n0 - 1) * msw
    icc = (msb - msw) / denom if denom else np.nan
    return {
        "n_players": int(len(groups)),
        "n_attempts": int(ns.sum()),
        "icc": float(icc),
        "within_sd": float(np.sqrt(max(msw, 0.0))),
        "sd_of_player_means": float(np.std(means, ddof=1)),
        "latent_between_sd": float(np.sqrt((msb - msw) / n0)) if (msb - msw) > 0 else 0.0,
        "n0": float(n0),
    }


CONTEXT_COLUMNS = ["combine_weight", "edge_play_share", "median_start_depth"]


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    y = train["median_getoff"].to_numpy(dtype=float)
    x_train, stats = _matrix(train, columns, None)
    x_test, _ = _matrix(test, columns, stats)
    model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
    model.fit(x_train, y)
    return model.predict(x_test), model.coef_, float(model.intercept_)


def _matrix(frame: pd.DataFrame, columns: list[str], stats: dict | None):
    names: list[str] = []
    blocks: list[np.ndarray] = []
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        blocks.append(values)
        names.append(column)
        if column in CONTEXT_COLUMNS:
            blocks.append((~np.isfinite(values)).astype(float))
            names.append(f"{column}_missing")
    data = np.column_stack(blocks)
    if stats is None:
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where((~np.isfinite(std)) | (std < 1e-8), 1.0, std)
        for index, name in enumerate(names):
            if name.endswith("_missing"):
                mean[index] = 0.0
                std[index] = 1.0
        stats = {"mean": mean, "std": std, "names": names}
    filled = data.copy()
    for index in range(filled.shape[1]):
        missing = ~np.isfinite(filled[:, index])
        filled[missing, index] = stats["mean"][index]
        filled[:, index] = (filled[:, index] - stats["mean"][index]) / stats["std"][index]
    return filled, stats


def _mae(y, pred) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def _choose_alpha(frame: pd.DataFrame, columns: list[str], alphas: list[float]) -> float:
    scores = []
    for alpha in alphas:
        errors = []
        for idx in frame.index:
            train = frame.drop(index=idx)
            test = frame.loc[[idx]]
            pred, _, _ = _fit_predict(train, test, columns, alpha)
            errors.append(abs(float(pred[0]) - float(test["median_getoff"].iloc[0])))
        scores.append((float(np.mean(errors)), float(alpha)))
    return min(scores)[1]


def _nested_loo(frame: pd.DataFrame, columns: list[str], alphas: list[float]) -> np.ndarray:
    preds = np.zeros(len(frame))
    for position, idx in enumerate(frame.index):
        train = frame.drop(index=idx)
        alpha = _choose_alpha(train, columns, alphas)
        pred, _, _ = _fit_predict(train, frame.loc[[idx]], columns, alpha)
        preds[position] = float(pred[0])
    return preds


MODEL_COLUMNS = {
    "M0": ["combine_weight", "edge_play_share", "median_start_depth"],
    "M1": ["combine_weight", "edge_play_share", "median_start_depth", "ten_yd_split"],
    "M2": ["combine_weight", "edge_play_share", "median_start_depth", "ten_yd_split", "early_transit_1_5_s"],
}


def fit_models(players: pd.DataFrame, spec: dict) -> dict:
    primary = players.loc[players["in_primary"]].copy()
    train = primary.loc[primary["draft_year"].eq(2023)].reset_index(drop=True)
    test = primary.loc[primary["draft_year"].eq(2024)].reset_index(drop=True)
    alphas = [float(a) for a in spec["models"]["alphas"]]
    chosen = {}
    predictions = test[["nfl_id", "draft_year", "nfl_position", "median_getoff"]].copy()
    coefs = {}
    for level, columns in MODEL_COLUMNS.items():
        chosen[level] = _choose_alpha(train, columns, alphas)
        pred, coef, intercept = _fit_predict(train, test, columns, chosen[level])
        predictions[f"pred_{level}"] = pred
        coefs[level] = {"alpha": chosen[level], "intercept": intercept, "coef": [float(v) for v in coef], "columns": columns}
    y = predictions["median_getoff"].to_numpy()
    chronological = {}
    for level in MODEL_COLUMNS:
        chronological[level] = _mae(y, predictions[f"pred_{level}"])
    err1 = np.abs(y - predictions["pred_M1"].to_numpy())
    err2 = np.abs(y - predictions["pred_M2"].to_numpy())
    rng = np.random.default_rng(int(spec["models"]["seed"]))
    boots = []
    n = len(predictions)
    for _ in range(int(spec["models"]["bootstrap_players"])):
        take = rng.integers(0, n, n)
        boots.append(float(err2[take].mean() - err1[take].mean()))
    boots = np.asarray(boots)
    # Coefficient uncertainty refits 2023 at the frozen penalty. It does not retune alpha.
    transit_coefs = []
    frozen_alpha = chosen["M2"]
    columns = MODEL_COLUMNS["M2"]
    for _ in range(int(spec["models"]["coefficient_bootstrap"])):
        take = rng.integers(0, len(train), len(train))
        sample = train.iloc[take]
        if sample["nfl_id"].nunique() < 8:
            continue
        transit_coefs.append(_transit_coef(sample, columns, frozen_alpha))
    transit_coefs = np.asarray(transit_coefs, dtype=float)
    nested = {}
    for level, columns in MODEL_COLUMNS.items():
        nested_pred = _nested_loo(primary.reset_index(drop=True), columns, alphas)
        nested[level] = _mae(primary["median_getoff"], nested_pred)
    influence = []
    for idx in predictions.index:
        held = predictions.drop(index=idx)
        influence.append(
            {
                "nfl_id": int(predictions.loc[idx, "nfl_id"]),
                "delta_without_player": _mae(held["median_getoff"], held["pred_M2"]) - _mae(held["median_getoff"], held["pred_M1"]),
            }
        )
    by_position = {}
    for position, part in predictions.groupby("nfl_position"):
        by_position[position] = {
            "n": int(len(part)),
            "mae_m1": _mae(part["median_getoff"], part["pred_M1"]),
            "mae_m2": _mae(part["median_getoff"], part["pred_M2"]),
        }
    spearman = {
        "pooled": _spear(primary["ten_yd_split"], primary["median_getoff"]),
        "y2023": _spear(train["ten_yd_split"], train["median_getoff"]),
        "y2024": _spear(test["ten_yd_split"], test["median_getoff"]),
        "transit_2024": _spear(test["early_transit_1_5_s"], test["median_getoff"]),
    }
    return {
        "n_2023": int(len(train)),
        "n_2024": int(len(test)),
        "alphas": chosen,
        "chronological_mae": chronological,
        "delta_m2_minus_m1": float(err2.mean() - err1.mean()),
        "delta_bootstrap": {
            "p2_5": float(np.percentile(boots, 2.5)),
            "p97_5": float(np.percentile(boots, 97.5)),
            "n_boot": int(len(boots)),
            "label": "conditional on the 2023-fitted 2024 predictions; players resampled; models not refit",
        },
        "transit_coefficient": {
            "point": float(np.mean(transit_coefs)) if len(transit_coefs) else None,
            "p2_5": float(np.percentile(transit_coefs, 2.5)) if len(transit_coefs) else None,
            "p97_5": float(np.percentile(transit_coefs, 97.5)) if len(transit_coefs) else None,
            "label": "2023 players resampled and M2 refit at the frozen penalty; alpha not retuned",
        },
        "nested_loo_mae": nested,
        "influence": influence,
        "by_position": by_position,
        "spearman_official_split": spearman,
        "predictions": predictions,
        "coefs": coefs,
        "standardized_coefficients_2023": dict(zip(_matrix(train, MODEL_COLUMNS["M2"], None)[1]["names"], coefs["M2"]["coef"])),
    }


def _transit_coef(train: pd.DataFrame, columns: list[str], alpha: float) -> float:
    y = train["median_getoff"].to_numpy(dtype=float)
    x, stats = _matrix(train, columns, None)
    model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
    model.fit(x, y)
    return float(model.coef_[stats["names"].index("early_transit_1_5_s")])


def common_penalty_diagnostic(players: pd.DataFrame, spec: dict | None = None) -> dict:
    """Refit the split model at the tracking model's penalty.

    This is a post-evaluation diagnostic. It does not replace the
    independently tuned M1-versus-M2 comparison. The shared penalty is 3,
    which was selected on 2023 for M2 only. 2025 is not used.
    """
    spec = spec or load_spec()
    primary = players.loc[players["in_primary"]].copy()
    train = primary.loc[primary["draft_year"].eq(2023)].reset_index(drop=True)
    test = primary.loc[primary["draft_year"].eq(2024)].reset_index(drop=True)
    if len(test) != 19 or len(train) != 18:
        raise ValueError(f"Expected 18/19 development players, found {len(train)}/{len(test)}")
    y = test["median_getoff"].to_numpy(dtype=float)
    shared = 3.0
    pred_m1, _, _ = _fit_predict(train, test, MODEL_COLUMNS["M1"], shared)
    pred_m2, _, _ = _fit_predict(train, test, MODEL_COLUMNS["M2"], shared)
    mae_m1 = _mae(y, pred_m1)
    mae_m2 = _mae(y, pred_m2)
    return {
        "label": "post-evaluation diagnostic; not a replacement for the independently tuned comparison",
        "shared_penalty": shared,
        "penalty_source": "2023 leave-one-player-out choice for M2",
        "n_2024": int(len(test)),
        "mae_m1_alpha_3": mae_m1,
        "mae_m2_alpha_3": mae_m2,
        "tracking_minus_split": mae_m2 - mae_m1,
        "mae": {"M1_alpha_3.0": mae_m1, "M2_alpha_3.0": mae_m2},
    }


def _spear(left, right) -> dict:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 5:
        return {"n": int(len(frame)), "spearman": None}
    return {"n": int(len(frame)), "spearman": float(frame["left"].corr(frame["right"], method="spearman"))}


def run(root: Path | None = None) -> dict:
    project = Project()
    root = root or project.root
    spec = load_spec(root)
    out = project.outputs / "getoff_v1"
    out.mkdir(parents=True, exist_ok=True)
    attempts = extract_attempts(root, spec)
    attempts.to_csv(out / "attempts.csv", index=False)
    plays, audit = load_getoff(root, spec)
    audit.to_csv(out / "getoff_availability.csv", index=False)
    depth = snap_depth(root, plays, spec)
    players = build_players(attempts, plays, depth, spec)
    players.to_csv(out / "player_ledger.csv", index=False)
    quality = {
        "attempts": int(len(attempts)),
        "valid_attempts": int(attempts["valid"].sum()),
        "valid_fraction": float(attempts["valid"].mean()) if len(attempts) else 0.0,
        "ambiguous_start": int(attempts["ambiguous_start"].sum()),
        "incomplete_crossing": int(attempts["reason"].eq("incomplete_crossing").sum()),
        "players_with_8_getoff": int((players["n_getoff"].fillna(0) >= 8).sum()),
        "primary_players": int(players["in_primary"].sum()),
        "repeatability": repeatability(attempts),
    }
    (out / "measurement_quality.json").write_text(json.dumps(quality, indent=2))
    stop = (
        quality["valid_fraction"] < float(spec["coverage_stop"]["min_fraction_attempts_with_origin_and_crossings"])
        or quality["primary_players"] < int(spec["coverage_stop"]["min_matched_development_players"])
    )
    summary = {"stopped_before_models": bool(stop), "quality": quality}
    if not stop:
        fitted = fit_models(players, spec)
        fitted["predictions"].to_csv(out / "predictions_2024.csv", index=False)
        summary.update({k: v for k, v in fitted.items() if k != "predictions"})
        diagnostic = common_penalty_diagnostic(players, spec)
        (out / "common_penalty_diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
        summary["common_penalty_diagnostic"] = diagnostic
        _figures(out / "presentation", attempts, players, fitted, diagnostic)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    print(json.dumps({k: summary[k] for k in summary if k not in {"influence", "coefs"}}, indent=2, default=_json))
    return summary


def _json(value):
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return str(value)


def _figures(out: Path, attempts: pd.DataFrame, players: pd.DataFrame, fitted: dict, diagnostic: dict | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out.mkdir(parents=True, exist_ok=True)
    valid = attempts.loc[attempts["valid"]].sort_values(["nfl_id", "attempt"])
    chosen = valid.groupby("nfl_id", as_index=False).head(1).head(4)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    ax = axes[0]
    for row in chosen.itertuples(index=False):
        ax.plot(
            [row.origin_x, row.x_at_1, row.x_at_5],
            [row.origin_y, row.y_at_1, row.y_at_5],
            color="#9E9E9E",
            lw=1.0,
            zorder=1,
        )
        ax.scatter([row.origin_x], [row.origin_y], color="#000000", s=28, zorder=3)
        ax.scatter([row.x_at_1], [row.y_at_1], color="#E69F00", s=28, zorder=3)
        ax.scatter([row.x_at_5], [row.y_at_5], color="#009E73", s=28, zorder=3)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#000000", markersize=7, label="Origin"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#E69F00", markersize=7, label="1-yard crossing"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#009E73", markersize=7, label="5-yard crossing"),
            Line2D([0], [0], color="#9E9E9E", lw=1.2, label="Line joins landmarks, not the path"),
        ],
        frameon=False,
        fontsize=8,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (yards)")
    ax.set_ylabel("y (yards)")
    ax.set_title("Sampled landmarks")
    ax = axes[1]
    both = valid.groupby("nfl_id").filter(lambda g: len(g) >= 2)
    first, second = [], []
    if len(both):
        wide = both.groupby("nfl_id")["early_transit_1_5_s"].apply(list)
        first = [v[0] for v in wide]
        second = [v[1] for v in wide]
        ax.scatter(first, second, color="#0072B2", s=28)
        lo = min(first + second) - 0.02
        hi = max(first + second) + 0.02
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.plot([lo, hi], [lo, hi], color="#666666", lw=0.8)
    ax.set_xlabel("First valid attempt (seconds)")
    ax.set_ylabel("Second valid attempt (seconds)")
    ax.set_title("Repeated forties, same scale")
    fig.tight_layout()
    fig.savefig(out / "fig_repeats.png", dpi=140)
    plt.close(fig)

    primary = players.loc[players["in_primary"]].copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for year, marker in ((2023, "o"), (2024, "s")):
        part = primary.loc[primary["draft_year"].eq(year)]
        ax.scatter(part["early_transit_1_5_s"], part["median_getoff"], marker=marker, s=36, label=str(year))
    ax.legend(title="Draft class", frameon=False)
    ax.set_xlabel("Median 1-to-5-yard transit (seconds)")
    ax.set_ylabel("Median rookie get-off (seconds)")
    ax.set_title(f"Description only, n={len(primary)}. Not the incremental test.")
    fig.tight_layout()
    fig.savefig(out / "fig_translation.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    ax = axes[0]
    labels = ["Context", "Plus official split", "Plus transit"]
    values = [fitted["chronological_mae"][k] for k in ("M0", "M1", "M2")]
    ax.bar(labels, values, color=["#666666", "#0072B2", "#009E73"])
    ax.set_ylabel("2024 mean absolute error (seconds)")
    ax.set_title("Independently tuned models")
    ax.tick_params(axis="x", labelrotation=15)
    ax = axes[1]
    delta_ms = 1000 * fitted["delta_m2_minus_m1"]
    low_ms = 1000 * fitted["delta_bootstrap"]["p2_5"]
    high_ms = 1000 * fitted["delta_bootstrap"]["p97_5"]
    ax.errorbar(
        [0],
        [delta_ms],
        yerr=[[delta_ms - low_ms], [high_ms - delta_ms]],
        fmt="o",
        color="#009E73",
        capsize=4,
        label="Tuned M2−M1, conditional bootstrap",
    )
    if diagnostic is not None:
        diag_ms = 1000 * diagnostic["tracking_minus_split"]
        ax.scatter([1], [diag_ms], color="#D55E00", s=36, label="Same penalty 3, retrospective")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([0, 1], ["Tuned models", "Shared penalty"])
    ax.set_ylabel("Tracking minus split (milliseconds)")
    ax.set_title(f"Interval {low_ms:.1f} to {high_ms:+.1f} ms")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_prediction.png", dpi=140)
    plt.close(fig)
    chosen.to_csv(out / "trajectory_examples.csv", index=False)


if __name__ == "__main__":
    run()
