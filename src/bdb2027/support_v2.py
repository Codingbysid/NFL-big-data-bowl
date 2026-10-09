"""Common-support membership for the versioned measurement study.

Cells are counted on a declared training fold. Later rows cannot change those
cells. Cuts outside the cells are missing measurements, not zero ability.
"""

from __future__ import annotations

import pandas as pd

from bdb2027.features import support_mask


def fit_support(combine: pd.DataFrame, nfl: pd.DataFrame, cfg: dict) -> list[dict]:
    _c, _n, cells = support_mask(combine, nfl, cfg)
    return cells


def supported_keys(cells: list[dict]) -> set[tuple[str, str]]:
    return {(row["speed_bin"], row["angle_bin"]) for row in cells if row.get("supported")}


def assign_bins(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = frame.copy()
    out["speed_bin"] = pd.cut(out["speed_in"], cfg["support"]["speed_edges"], include_lowest=True).astype(str)
    out["angle_bin"] = pd.cut(out["abs_angle_deg"], cfg["support"]["angle_edges"], include_lowest=True).astype(str)
    return out


def apply_support(frame: pd.DataFrame, cells: list[dict], cfg: dict) -> pd.DataFrame:
    out = assign_bins(frame, cfg)
    keys = supported_keys(cells)
    out["in_support"] = [(s, a) in keys for s, a in zip(out["speed_bin"], out["angle_bin"])]
    out["support_status"] = np_where(out["in_support"], "included", "excluded_unsupported_cell")
    return out


def np_where(mask, yes, no):
    return [yes if flag else no for flag in mask]


def supported_attempt_means(cuts: pd.DataFrame) -> pd.DataFrame:
    """Average only supported cuts. Players with none are omitted."""
    kept = cuts.loc[cuts["in_support"]].copy()
    if kept.empty:
        return kept
    return (
        kept.groupby(["nfl_id", "attempt_uid"], as_index=False)
        .agg(residual=("residual", "mean"), n_cuts=("residual", "size"), drill_family=("drill_family", "first"))
    )
