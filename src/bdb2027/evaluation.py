"""Player-equal metrics, paired bootstrap, and sanity checks.

Frames and repeated routes are not treated as independent prospects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_player_weights(nfl_ids) -> np.ndarray:
    """Each player receives total weight 1, split evenly across that player's rows."""
    ids = pd.Series(list(nfl_ids))
    counts = ids.value_counts()
    if (counts <= 0).any():
        raise ValueError("Player row counts must be positive")
    weights = (1.0 / ids.map(counts)).to_numpy(dtype=float)
    return weights


def player_table(nfl_id, y, pred) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "nfl_id": np.asarray(nfl_id),
            "y": np.asarray(y, dtype=float),
            "p": np.asarray(pred, dtype=float),
        }
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["nfl_id", "y", "p", "n", "route_mae"])
    frame["ae"] = (frame["y"] - frame["p"]).abs()
    grouped = frame.groupby("nfl_id", as_index=False).agg(
        y=("y", "mean"),
        p=("p", "mean"),
        n=("y", "size"),
        route_mae=("ae", "mean"),
    )
    return grouped


def summarize(nfl_id, y, pred) -> dict:
    grouped = player_table(nfl_id, y, pred)
    if grouped.empty:
        return {"n_players": 0, "n_routes": 0, "mae": np.nan, "rmse": np.nan, "spearman": np.nan, "calibration_slope": np.nan, "route_mae": np.nan, "route_mae_player_equal": np.nan}
    error = grouped["y"] - grouped["p"]
    route_n = int(grouped["n"].sum())
    slope = np.nan
    if grouped["p"].std(ddof=0) > 1e-8:
        slope = float(np.polyfit(grouped["p"].to_numpy(), grouped["y"].to_numpy(), 1)[0])
    spearman = grouped["y"].corr(grouped["p"], method="spearman")
    return {
        "n_players": int(len(grouped)),
        "n_routes": route_n,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(spearman) if pd.notna(spearman) else np.nan,
        "calibration_slope": slope,
        "route_mae": np.nan,
        "route_mae_player_equal": float(grouped["route_mae"].mean()),
    }


def summarize_frames(nfl_id, y, pred) -> dict:
    """Like summarize, but route-level MAE is the unweighted mean over routes."""
    out = summarize(nfl_id, y, pred)
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(pred, dtype=float)})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    out["route_mae"] = float(np.mean(np.abs(frame["y"] - frame["p"]))) if len(frame) else np.nan
    return out


def paired_bootstrap_delta(nfl_id, y, pred_a, pred_b, n_boot: int = 2000, seed: int = 2027) -> dict:
    """Bootstrap players, not routes. Delta is MAE(b) - MAE(a); negative favors b."""
    table_a = player_table(nfl_id, y, pred_a).rename(columns={"p": "pa", "route_mae": "route_mae_a"})
    table_b = player_table(nfl_id, y, pred_b)[["nfl_id", "p"]].rename(columns={"p": "pb"})
    merged = table_a.merge(table_b, on="nfl_id", how="inner")
    if merged.empty:
        return {"n_players": 0, "delta_mae": np.nan, "p2_5": np.nan, "p97_5": np.nan, "n_boot": n_boot}
    yv = merged["y"].to_numpy()
    pa = merged["pa"].to_numpy()
    pb = merged["pb"].to_numpy()
    point = float(np.mean(np.abs(yv - pb)) - np.mean(np.abs(yv - pa)))
    rng = np.random.default_rng(seed)
    n = len(merged)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, n, size=n)
        deltas[i] = np.mean(np.abs(yv[take] - pb[take])) - np.mean(np.abs(yv[take] - pa[take]))
    return {
        "n_players": int(n),
        "delta_mae": point,
        "p2_5": float(np.percentile(deltas, 2.5)),
        "p97_5": float(np.percentile(deltas, 97.5)),
        "n_boot": int(n_boot),
        "label": "paired player bootstrap of frozen predictions; conditional on the fitted pipeline",
    }


def leave_one_player_influence(nfl_id, y, pred_a, pred_b) -> pd.DataFrame:
    """How the MAE gap changes when each player is removed from the metric."""
    table_a = player_table(nfl_id, y, pred_a).rename(columns={"p": "pa"})
    table_b = player_table(nfl_id, y, pred_b)[["nfl_id", "p"]].rename(columns={"p": "pb"})
    merged = table_a.merge(table_b, on="nfl_id")
    yv = merged["y"].to_numpy()
    pa = merged["pa"].to_numpy()
    pb = merged["pb"].to_numpy()
    full = float(np.mean(np.abs(yv - pb)) - np.mean(np.abs(yv - pa)))
    rows = []
    for i, nfl in enumerate(merged["nfl_id"]):
        mask = np.ones(len(merged), dtype=bool)
        mask[i] = False
        delta = float(np.mean(np.abs(yv[mask] - pb[mask])) - np.mean(np.abs(yv[mask] - pa[mask])))
        rows.append({"nfl_id": int(nfl), "delta_without_player": delta, "influence": full - delta})
    return pd.DataFrame(rows).sort_values("influence", key=np.abs, ascending=False)
