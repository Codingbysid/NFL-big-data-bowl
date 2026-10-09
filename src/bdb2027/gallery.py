"""Outcome-blind plots for checking the cut detector.

Plots are titled with a hash. Names, draft status, and separation are not drawn.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from bdb2027.config import Project
from bdb2027.extract import cohort_dir
from bdb2027.plots import plot_montage, plot_trajectory


def anon(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(f"bdb2027-review|{raw}".encode()).hexdigest()[:10]


def _seconds(series: pd.Series) -> np.ndarray:
    values = pd.to_datetime(series).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return values / 1e9


def _take(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.iloc[0:0]
    # One row per player, then sample players.
    one = frame.groupby("nfl_id", as_index=False).head(1)
    if len(one) > n:
        index = rng.choice(len(one), size=n, replace=False)
        one = one.iloc[np.sort(index)]
    return one


def _sample_combine(quality: pd.DataFrame, cuts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    primary = ["SPEED_OUT", "DAGGER", "SLOT_SAIL", "POST_CORNER", "SLANT"]
    rows = []
    detected = cuts.loc[cuts["sg_window"] == 5].copy()
    for family in primary:
        hit = detected.loc[detected["drill_family"] == family]
        rows.append(_take(hit, 4, rng))
        missed = quality.loc[(quality["drill_family"] == family) & (quality["n_cuts_primary_window"] == 0)]
        rows.append(_take(missed, 2, rng))
    rows.append(_take(quality.loc[(quality["drill_role"] == "secondary_turnback") & (quality["n_cuts_primary_window"] > 0)], 3, rng))
    rows.append(_take(quality.loc[(quality["drill_role"] == "excluded") & (quality["n_cuts_primary_window"] > 0)], 3, rng))
    rows.append(_take(quality.loc[(quality["drill_role"] == "excluded") & (quality["n_cuts_primary_window"] == 0)], 2, rng))
    out = pd.concat([r for r in rows if len(r)], ignore_index=True)
    return out.drop_duplicates(["nfl_id", "event_id", "attempt", "drill_name"])


def _sample_nfl(quality: pd.DataFrame, cuts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    detected = cuts.loc[cuts["sg_window"] == 5]
    for route in ["OUT", "IN", "SLANT", "HITCH", "CORNER", "POST", "CROSS", "GO"]:
        rows.append(_take(detected.loc[detected["route_ran"] == route], 3, rng))
    rows.append(_take(quality.loc[(quality["route_ran"] == "OUT") & (quality["n_cuts_primary_window"] == 0)], 4, rng))
    rows.append(_take(quality.loc[quality["route_ran"] == "SCREEN"], 2, rng))
    out = pd.concat([r for r in rows if len(r)], ignore_index=True)
    return out.drop_duplicates(["game_id", "play_id", "nfl_id"])


def _cut_for(cuts: pd.DataFrame, mask: pd.Series) -> dict | None:
    part = cuts.loc[mask & (cuts["sg_window"] == 5)]
    if part.empty:
        return None
    best = part.sort_values("abs_angle_deg", ascending=False).iloc[0]
    return best.to_dict()


def write_gallery(project: Project) -> Path:
    rng = np.random.default_rng(2027)
    out = project.reports / "review_gallery"
    out.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    image_paths = []

    combine_q = pd.read_parquet(cohort_dir(project, 2023) / "combine_quality.parquet")
    combine_c = pd.read_parquet(cohort_dir(project, 2023) / "combine_cuts.parquet")
    combine_f = pd.read_parquet(cohort_dir(project, 2023) / "combine_frames.parquet")
    chosen = _sample_combine(combine_q, combine_c, rng)
    for row in chosen.itertuples(index=False):
        frames = combine_f[
            (combine_f["nfl_id"] == row.nfl_id)
            & (combine_f["event_id"] == row.event_id)
            & (combine_f["attempt"] == row.attempt)
            & (combine_f["drill_name"] == row.drill_name)
        ]
        token = anon("combine", row.nfl_id, row.event_id, row.attempt, row.drill_name)
        cut = _cut_for(
            combine_c,
            (combine_c["nfl_id"] == row.nfl_id)
            & (combine_c["event_id"] == row.event_id)
            & (combine_c["attempt"] == row.attempt),
        )
        title = f"{token}  {row.drill_family}  {'cut detected' if cut else 'no cut'}"
        path = out / f"combine_{token}.png"
        plot_trajectory(_seconds(frames["time"]), frames["x"].to_numpy(), frames["y"].to_numpy(), path, title, cut)
        image_paths.append(path)
        sheet_rows.append(
            {
                "anon_id": token,
                "source": "combine",
                "family": row.drill_family,
                "drill_or_route": row.drill_name,
                "detected": int(cut is not None),
                "abs_angle": None if cut is None else round(float(cut["abs_angle_deg"]), 1),
                "exit_distance": None if cut is None else round(float(cut["exit_distance_0p5"]), 2),
                "speed_in": None if cut is None else round(float(cut["speed_in"]), 2),
                "verdict": "",
                "note": "",
            }
        )

    nfl_q = pd.read_parquet(cohort_dir(project, 2023) / "nfl_quality.parquet")
    nfl_c = pd.read_parquet(cohort_dir(project, 2023) / "nfl_cuts.parquet")
    nfl_f = pd.read_parquet(cohort_dir(project, 2023) / "nfl_frames.parquet")
    chosen_n = _sample_nfl(nfl_q, nfl_c, rng)
    for row in chosen_n.itertuples(index=False):
        frames = nfl_f[(nfl_f["game_id"] == row.game_id) & (nfl_f["play_id"] == row.play_id) & (nfl_f["nfl_id"] == row.nfl_id)]
        token = anon("nfl", row.game_id, row.play_id, row.nfl_id)
        cut = _cut_for(
            nfl_c,
            (nfl_c["game_id"] == row.game_id) & (nfl_c["play_id"] == row.play_id) & (nfl_c["nfl_id"] == row.nfl_id),
        )
        title = f"{token}  {row.route_ran}  {'cut detected' if cut else 'no cut'}"
        path = out / f"nfl_{token}.png"
        plot_trajectory(_seconds(frames["time"]), frames["x"].to_numpy(), frames["y"].to_numpy(), path, title, cut)
        image_paths.append(path)
        sheet_rows.append(
            {
                "anon_id": token,
                "source": "nfl",
                "family": row.route_ran,
                "drill_or_route": row.route_ran,
                "detected": int(cut is not None),
                "abs_angle": None if cut is None else round(float(cut["abs_angle_deg"]), 1),
                "exit_distance": None if cut is None else round(float(cut["exit_distance_0p5"]), 2),
                "speed_in": None if cut is None else round(float(cut["speed_in"]), 2),
                "verdict": "",
                "note": "",
            }
        )

    sheet = pd.DataFrame(sheet_rows)
    sheet_path = project.reports / "review_sheet.csv"
    sheet.to_csv(sheet_path, index=False)
    plot_montage(image_paths[:24], project.reports / "review_montage_combine.png", cols=4)
    plot_montage(image_paths[24:], project.reports / "review_montage_nfl.png", cols=4)
    print(f"[gallery] {len(sheet)} plots, {int((sheet.source=='combine').sum())} combine and {int((sheet.source=='nfl').sum())} nfl", flush=True)
    return sheet_path
