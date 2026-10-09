"""2023–24 measurement study for one stem-and-break scope.

This module does not read 2025 separation and does not replace the archived
detector. Human review labels are not invented.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bdb2027.config import Project, code_hash, config_hash, load_config
from bdb2027.cuts import detection_params
from bdb2027.cuts_v2 import detect_cuts_bounded
from bdb2027.demand_v2 import PENALTY_GRID, select_lambda, weighted_mse
from bdb2027.demand_v2 import WeightedDemandModel
from bdb2027.extract import cohort_dir
from bdb2027.features import player_attempt_cut_weights
from bdb2027.plots import plot_trajectory
from bdb2027.support_v2 import apply_support, fit_support

V2_CONFIG = "config/measurement_v2.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seconds(series: pd.Series) -> np.ndarray:
    values = pd.to_datetime(series).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return values / 1e9


def _cut_key(trajectory_id: str, t0: float) -> str:
    return hashlib.sha256(f"measurement_v2|{trajectory_id}|{t0:.4f}".encode()).hexdigest()[:16]


def _load_scope(project: Project, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    combine_rows = []
    nfl_rows = []
    for year in spec["cohorts"]:
        cq = pd.read_parquet(cohort_dir(project, year) / "combine_quality.parquet")
        cf = pd.read_parquet(cohort_dir(project, year) / "combine_frames.parquet")
        keep_c = cq["drill_family"].isin(spec["scope"]["combine_families"])
        cq = cq.loc[keep_c, ["nfl_id", "event_id", "attempt", "drill_name", "drill_family", "attempt_uid"]]
        cf = cf.merge(cq, on=["nfl_id", "event_id", "attempt", "drill_name"], how="inner")
        combine_rows.append(cf)
        nq = pd.read_parquet(cohort_dir(project, year) / "nfl_quality.parquet")
        nf = pd.read_parquet(cohort_dir(project, year) / "nfl_frames.parquet")
        keep_n = nq["route_ran"].isin(spec["scope"]["nfl_routes"])
        nq = nq.loc[keep_n, ["game_id", "play_id", "nfl_id", "route_ran", "attempt_uid", "draft_year"]]
        nf = nf.merge(nq, on=["game_id", "play_id", "nfl_id"], how="inner")
        nfl_rows.append(nf)
    return pd.concat(combine_rows, ignore_index=True), pd.concat(nfl_rows, ignore_index=True)


def _measure_group(frame: pd.DataFrame, params: dict, *, domain: str, fingerprints: dict) -> tuple[list[dict], list[dict]]:
    frame = frame.sort_values("time")
    t = _seconds(frame["time"])
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    events = frame["event"].fillna("").to_numpy() if "event" in frame.columns else None
    family = frame["drill_family"].iloc[0] if domain == "combine" else frame["route_ran"].iloc[0]
    trajectory_id = str(frame["attempt_uid"].iloc[0])
    accepted, _info, reasons = detect_cuts_bounded(
        t, x, y, events=events, params=params, require_pass_event=(domain == "nfl")
    )
    base = {
        "domain": domain,
        "trajectory_id": trajectory_id,
        "nfl_id": int(frame["nfl_id"].iloc[0]),
        "draft_year": int(frame["draft_year"].iloc[0]),
        "family": str(family),
        "drill_family": str(family),
        "code_hash": fingerprints["code_hash"],
        "config_hash": fingerprints["config_hash"],
        "detector": "measurement_v2_bounded",
    }
    rows = []
    for cut in accepted:
        row = {**base, **cut}
        row["cut_id"] = _cut_key(trajectory_id, float(cut["t0"]))
        row["selected"] = True
        row["rejection_reason"] = ""
        rows.append(row)
    rejections = []
    for reason in reasons:
        rejections.append({**base, **reason, "selected": False, "cut_id": ""})
    if len(accepted) > 1:
        for row in rows:
            row["selected"] = False
            row["rejection_reason"] = "abstain_multiple_cuts"
    return rows, rejections


def measure(project: Project, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = detection_params(load_config(project.root / "config" / "research.yaml"))
    fingerprints = {"code_hash": code_hash(), "config_hash": config_hash(spec)}
    combine, nfl = _load_scope(project, spec)
    accepted, rejected = [], []
    for _uid, frame in combine.groupby("attempt_uid", sort=False):
        rows, reasons = _measure_group(frame, params, domain="combine", fingerprints=fingerprints)
        accepted.extend(rows)
        rejected.extend(reasons)
    for _uid, frame in nfl.groupby("attempt_uid", sort=False):
        rows, reasons = _measure_group(frame, params, domain="nfl", fingerprints=fingerprints)
        accepted.extend(rows)
        rejected.extend(reasons)
    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def _player_split(ids: list[int], fraction: float, seed: int) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    ordered = np.array(sorted(ids))
    rng.shuffle(ordered)
    n_val = max(1, int(round(len(ordered) * fraction)))
    validation = set(int(i) for i in ordered[:n_val])
    return {int(i): ("validation" if int(i) in validation else "detector_development") for i in ordered}


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.iloc[0:0].copy()
    take = min(n, len(frame))
    index = rng.choice(len(frame), size=take, replace=False)
    out = frame.iloc[np.sort(index)].copy()
    out["sampling_probability"] = take / len(frame)
    out["stratum_n"] = len(frame)
    return out


def review_sample(cuts: pd.DataFrame, rejected: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player-disjoint validation sample. Detector settings are not retuned here."""
    ids = sorted(set(cuts["nfl_id"]).union(rejected["nfl_id"]))
    roles = _player_split(ids, float(spec["review"]["validation_fraction"]), int(spec["review"]["seed"]))
    cuts = cuts.copy()
    rejected = rejected.copy()
    cuts["split"] = cuts["nfl_id"].map(roles)
    rejected["split"] = rejected["nfl_id"].map(roles)
    rng = np.random.default_rng(int(spec["review"]["seed"]))
    rows = []
    single = cuts.loc[cuts["rejection_reason"].fillna("").eq("") & cuts["selected"].astype(bool)]
    for domain in ("combine", "nfl"):
        part = single.loc[(single["domain"] == domain) & (single["split"] == "validation")]
        families = sorted(part["family"].unique())
        per = max(1, int(np.ceil(spec["review"]["accepted_per_domain"] / max(len(families), 1))))
        for family in families:
            taken = _sample(part.loc[part["family"] == family], per, rng)
            taken["stratum"] = f"{domain}|{family}|accepted"
            rows.append(taken)
        misses = rejected.loc[(rejected["domain"] == domain) & (rejected["split"] == "validation")]
        misses = misses.drop_duplicates("trajectory_id")
        taken = _sample(misses, int(spec["review"]["rejected_per_domain"]), rng)
        if len(taken):
            taken["stratum"] = f"{domain}|rejected"
            rows.append(taken)
    sample = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return sample, pd.DataFrame({"nfl_id": list(roles), "split": [roles[i] for i in roles]})


def write_annotation_pack(sample: pd.DataFrame, out: Path, frames_lookup: dict) -> None:
    """Blind gallery and an empty label sheet. Detector times stay in a separate key."""
    gallery = out / "annotation_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    labels = []
    keys = []
    for row in sample.itertuples(index=False):
        token = hashlib.sha256(f"v2-review|{row.trajectory_id}".encode()).hexdigest()[:10]
        labels.append(
            {
                "review_id": token,
                "domain": row.domain,
                "family": row.family,
                "stratum": row.stratum,
                "sampling_probability": getattr(row, "sampling_probability", np.nan),
                "stratum_n": getattr(row, "stratum_n", np.nan),
                "intended_break_exists": "",
                "intended_interval_start_s": "",
                "intended_interval_end_s": "",
                "ambiguous": "",
                "reviewer_id": "",
                "reviewer_role": "",
                "annotated_before_seeing_detector": "",
                "notes": "",
            }
        )
        keys.append(
            {
                "review_id": token,
                "trajectory_id": row.trajectory_id,
                "cut_id": getattr(row, "cut_id", ""),
                "detector_t0": getattr(row, "t0", np.nan),
                "t_in0": getattr(row, "t_in0", np.nan),
                "t_out1": getattr(row, "t_out1", np.nan),
                "latest_raw_timestamp": getattr(row, "latest_raw_timestamp", np.nan),
                "bound_t": getattr(row, "bound_t", np.nan),
                "selected": getattr(row, "selected", False),
            }
        )
        frames = frames_lookup.get(row.trajectory_id)
        if frames is not None:
            t = _seconds(frames["time"])
            bound = getattr(row, "bound_t", np.nan)
            plot_trajectory(
                t,
                frames["x"].to_numpy(dtype=float),
                frames["y"].to_numpy(dtype=float),
                gallery / f"{token}.png",
                f"{token}  {row.family}",
                cut=None,
                bound_t=None if pd.isna(bound) else float(bound),
            )
    pd.DataFrame(labels).to_csv(out / "annotation_template.csv", index=False)
    pd.DataFrame(keys).to_csv(out / "annotation_detector_key.csv", index=False)


def score_annotations(template: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Score human labels. An empty template is pending, not agreement."""
    if template.empty or template["intended_break_exists"].astype(str).str.strip().eq("").all():
        return {"status": "pending_human", "agreement": None, "n_reviewed": 0}
    merged = template.merge(key, on="review_id", how="left")
    established = merged.loc[merged["ambiguous"].astype(str).str.lower().ne("yes")].copy()
    # Agreement: a labeled interval contains the detector t0, or both say no break.
    def _agree(row) -> bool:
        exists = str(row["intended_break_exists"]).strip().lower()
        if exists in {"no", "absent"} and not bool(row["selected"]):
            return True
        if exists != "yes" or not np.isfinite(row.get("detector_t0", np.nan)):
            return False
        start = pd.to_numeric(row["intended_interval_start_s"], errors="coerce")
        end = pd.to_numeric(row["intended_interval_end_s"], errors="coerce")
        return bool(np.isfinite(start) and np.isfinite(end) and start <= row["detector_t0"] <= end)

    established["agree"] = established.apply(_agree, axis=1)
    accepted = established.loc[established["selected"].astype(bool)]
    by_domain = {}
    for domain, part in accepted.groupby("domain"):
        by_domain[domain] = {"agree": int(part["agree"].sum()), "n": int(len(part))}
    return {"status": "scored", "by_domain": by_domain, "n_reviewed": int(len(established))}


def _family_calibration(residuals: pd.DataFrame) -> list[dict]:
    rows = []
    rng = np.random.default_rng(20271009)
    for family, part in residuals.groupby("drill_family"):
        player_means = part.groupby("nfl_id")["residual"].mean()
        means = []
        for _ in range(400):
            take = rng.choice(player_means.index.to_numpy(), size=len(player_means), replace=True)
            means.append(float(player_means.loc[take].mean()))
        arr = np.asarray(means, dtype=float)
        rows.append(
            {
                "slice": f"family:{family}",
                "n_cuts": int(len(part)),
                "n_players": int(len(player_means)),
                "mean_residual": float(part.groupby("nfl_id")["residual"].mean().mean()),
                "p2_5": float(np.nanpercentile(arr, 2.5)),
                "p97_5": float(np.nanpercentile(arr, 97.5)),
            }
        )
    return rows


def reconcile_historical_gallery(project: Project, out: Path) -> dict:
    """Join the old sheet to trajectories and list every frozen-config cut.

    The old plot showed only the largest-angle cut. Rows are not deleted.
    """
    from bdb2027.gallery import _sample_combine, _sample_nfl, anon

    sheet = pd.read_csv(project.reports / "review_sheet.csv")
    rng = np.random.default_rng(2027)
    records = []
    for year_ok, kind, sampler in (
        (2023, "combine", _sample_combine),
        (2023, "nfl", _sample_nfl),
    ):
        quality = pd.read_parquet(cohort_dir(project, 2023) / f"{kind}_quality.parquet")
        cuts = pd.read_parquet(cohort_dir(project, 2023) / f"{kind}_cuts.parquet")
        chosen = sampler(quality, cuts, rng)
        for row in chosen.itertuples(index=False):
            if kind == "combine":
                token = anon("combine", row.nfl_id, row.event_id, row.attempt, row.drill_name)
                mask = (cuts["nfl_id"] == row.nfl_id) & (cuts["event_id"] == row.event_id) & (cuts["attempt"] == row.attempt) & (cuts["sg_window"] == 5)
            else:
                token = anon("nfl", row.game_id, row.play_id, row.nfl_id)
                mask = (cuts["game_id"] == row.game_id) & (cuts["play_id"] == row.play_id) & (cuts["nfl_id"] == row.nfl_id) & (cuts["sg_window"] == 5)
            part = cuts.loc[mask]
            records.append(
                {
                    "anon_id": token,
                    "source": kind,
                    "n_cuts_in_attempt": int(len(part)),
                    "shown_was_largest_angle_only": True,
                    "cut_speeds": ";".join(f"{v:.2f}" for v in part["speed_in"].tolist()),
                }
            )
    joined = sheet.merge(pd.DataFrame(records), on=["anon_id", "source"], how="left")
    detected = joined.loc[joined["detected"].eq(1)]
    primary = {"SPEED_OUT", "DAGGER", "SLOT_SAIL", "POST_CORNER", "SLANT"}
    gated = detected.loc[detected["source"].eq("combine") & detected["family"].isin(primary) & (detected["speed_in"] >= 4)]
    payload = {
        "sheet_rows": int(len(sheet)),
        "joined_rows": int(joined["n_cuts_in_attempt"].notna().sum()),
        "detected_below_4_combine": int(((detected["source"] == "combine") & (detected["speed_in"] < 4)).sum()),
        "detected_below_4_nfl": int(((detected["source"] == "nfl") & (detected["speed_in"] < 4)).sum()),
        "combine_fail_families": sorted(joined.loc[(joined["source"] == "combine") & (joined["verdict"] == "fail"), "family"].dropna().unique().tolist()),
        "gated_combine_verdicts": gated["verdict"].value_counts().to_dict(),
        "nfl_pass_rows": int(((joined["source"] == "nfl") & (joined["verdict"] == "pass")).sum()),
        "nfl_pass_rows_with_detected_0": int(((joined["source"] == "nfl") & (joined["verdict"] == "pass") & (joined["detected"] == 0)).sum()),
        "attempts_with_multiple_cuts_in_joined_gallery": int((joined["n_cuts_in_attempt"] > 1).sum()),
    }
    (out / "historical_review_reconciliation.json").write_text(json.dumps(payload, indent=2))
    joined.to_csv(out / "historical_review_joined.csv", index=False)
    return payload


def run(project: Project | None = None) -> dict:
    project = project or Project()
    spec = load_config(project.root / V2_CONFIG)
    out = project.outputs / "measurement_v2"
    out.mkdir(parents=True, exist_ok=True)
    print("[v2] measuring 2023-24 scope", flush=True)
    reconcile_historical_gallery(project, out)
    cuts, rejected = measure(project, spec)
    if cuts.empty:
        raise RuntimeError("Bounded detector accepted no in-scope cuts")
    # One-break rule: multiple cuts on an attempt are abstentions.
    multi = set(cuts.loc[cuts["rejection_reason"].eq("abstain_multiple_cuts"), "trajectory_id"])
    usable = cuts.loc[~cuts["trajectory_id"].isin(multi)].copy()
    combine = usable.loc[usable["domain"].eq("combine")].copy().reset_index(drop=True)
    nfl = usable.loc[usable["domain"].eq("nfl")].copy()
    print(f"[v2] accepted single cuts combine {len(combine)} nfl {len(nfl)}", flush=True)
    selection = select_lambda(combine, PENALTY_GRID, n_splits=5)
    model = WeightedDemandModel(alpha=float(selection["chosen_lambda"]), reference_family="SPEED_OUT")
    model.fit(combine, player_attempt_cut_weights(combine))
    combine = combine.copy()
    combine["expected_exit"] = model.predict(combine)
    combine["residual"] = combine["exit_distance_0p5"] - combine["expected_exit"]
    combine["oof_pred"] = selection["oof_prediction"]
    combine["oof_residual"] = combine["exit_distance_0p5"] - combine["oof_pred"]
    train_c = combine.loc[combine["draft_year"].eq(2023)]
    train_n = nfl.loc[nfl["draft_year"].eq(2023)]
    support_cfg = {
        "support": {
            "speed_edges": load_config()["support"]["speed_edges"],
            "angle_edges": load_config()["support"]["angle_edges"],
            "min_cell_combine": int(spec["support"]["min_cell_combine"]),
            "min_cell_nfl": int(spec["support"]["min_cell_nfl"]),
        }
    }
    cells = fit_support(train_c, train_n, support_cfg)
    combine = apply_support(combine, cells, support_cfg)
    nfl_scored = nfl.copy()
    if len(nfl_scored):
        nfl_model = WeightedDemandModel(alpha=float(selection["chosen_lambda"]), reference_family="OUT")
        # NFL demand uses the same lambda but its own 2023 fit. Family names differ.
        nfl_2023 = nfl_scored.loc[nfl_scored["draft_year"].eq(2023)]
        if nfl_2023["nfl_id"].nunique() >= 2 and len(nfl_2023) >= 8:
            nfl_model.reference_family = "OUT"
            nfl_model.fit(nfl_2023, player_attempt_cut_weights(nfl_2023))
            nfl_scored["expected_exit"] = nfl_model.predict(nfl_scored)
            nfl_scored["residual"] = nfl_scored["exit_distance_0p5"] - nfl_scored["expected_exit"]
        nfl_scored = apply_support(nfl_scored, cells, support_cfg)
    cuts.to_parquet(out / "accepted_cuts.parquet", index=False)
    rejected.to_parquet(out / "rejections.parquet", index=False)
    combine.to_parquet(out / "combine_single_cuts.parquet", index=False)
    nfl_scored.to_parquet(out / "nfl_single_cuts.parquet", index=False)
    pd.DataFrame(cells).to_csv(out / "support_cells_2023.csv", index=False)
    membership = pd.concat(
        [
            combine[["cut_id", "trajectory_id", "domain", "in_support", "support_status", "speed_bin", "angle_bin"]],
            nfl_scored[["cut_id", "trajectory_id", "domain", "in_support", "support_status", "speed_bin", "angle_bin"]]
            if len(nfl_scored) and "cut_id" in nfl_scored.columns
            else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    membership.to_csv(out / "support_membership.csv", index=False)
    calibration = _family_calibration(combine.dropna(subset=["oof_residual"]).assign(residual=lambda d: d["oof_residual"]))
    speed = combine.dropna(subset=["oof_residual"]).copy()
    speed["speed_band"] = pd.cut(speed["speed_in"], [4, 6, 8, 13], include_lowest=True)
    for band, part in speed.groupby("speed_band", observed=False):
        if len(part) == 0:
            continue
        calibration.append(
            {
                "slice": f"speed:{band}",
                "n_cuts": int(len(part)),
                "n_players": int(part["nfl_id"].nunique()),
                "mean_residual": float(part.groupby("nfl_id")["oof_residual"].mean().mean()),
                "p2_5": None,
                "p97_5": None,
            }
        )
    pd.DataFrame(calibration).to_csv(out / "calibration.csv", index=False)
    boundary_violations = 0
    if len(nfl_scored) and "latest_raw_timestamp" in nfl_scored.columns:
        boundary_violations = int(
            (
                (nfl_scored["latest_raw_timestamp"] > nfl_scored["bound_t"] + 1e-6)
                | (nfl_scored["t_out1"] > nfl_scored["bound_t"] + 1e-6)
            ).sum()
        )
    supported_c = combine.loc[combine["in_support"]]
    supported_n = nfl_scored.loc[nfl_scored["in_support"]] if len(nfl_scored) and "in_support" in nfl_scored.columns else nfl_scored.iloc[0:0]
    both = set(supported_c.groupby("nfl_id").size().loc[lambda s: s >= 2].index).intersection(set(supported_n["nfl_id"]))
    both_2024 = set(supported_c.loc[supported_c["draft_year"].eq(2024), "nfl_id"]).intersection(
        set(supported_n.loc[supported_n["draft_year"].eq(2024), "nfl_id"]) if len(supported_n) else set()
    )
    sample, roles = review_sample(pd.concat([combine, nfl_scored], ignore_index=True), rejected, spec)
    sample.to_csv(out / "review_sample.csv", index=False)
    roles.to_csv(out / "player_split.csv", index=False)
    # Blind plots use stored frames. Build a small lookup for the sample only.
    lookup = {}
    if len(sample):
        combine_frames, nfl_frames = _load_scope(project, spec)
        for frame in (combine_frames, nfl_frames):
            wanted = set(sample["trajectory_id"])
            part = frame.loc[frame["attempt_uid"].isin(wanted)]
            for uid, group in part.groupby("attempt_uid"):
                lookup[str(uid)] = group
    write_annotation_pack(sample, out, lookup)
    score = score_annotations(pd.read_csv(out / "annotation_template.csv"), pd.read_csv(out / "annotation_detector_key.csv"))
    (out / "annotation_score.json").write_text(json.dumps(score, indent=2))
    same = (
        combine.groupby(["nfl_id", "drill_family"])["trajectory_id"].nunique()
        if len(combine)
        else pd.Series(dtype=int)
    )
    summary = {
        "chosen_lambda": selection["chosen_lambda"],
        "penalty_grid": selection["grid"],
        "historical_lambda": 3.0,
        "demand_fit_on": spec["demand"]["fit_on"],
        "support_training_year": 2023,
        "n_support_cells": int(sum(1 for row in cells if row["supported"])),
        "combine_single_cuts": int(len(combine)),
        "nfl_single_cuts": int(len(nfl_scored)),
        "combine_players": int(combine["nfl_id"].nunique()),
        "nfl_players": int(nfl_scored["nfl_id"].nunique()) if len(nfl_scored) else 0,
        "players_with_both_supported": int(len(both)),
        "players_with_both_supported_2024": int(len(both_2024)),
        "boundary_violations_on_accepted_nfl": boundary_violations,
        "same_family_groups_with_2plus": int((same >= 2).sum()) if len(same) else 0,
        "review_status": score["status"],
        "review_rows": int(len(sample)),
        "abstained_multiple_cut_attempts": int(len(multi)),
        "oof_wmse_at_chosen_lambda": float(
            weighted_mse(
                combine["exit_distance_0p5"].to_numpy(),
                combine["oof_pred"].to_numpy(),
                player_attempt_cut_weights(combine),
            )
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _figures(out, summary, calibration)
    print("[v2] summary", json.dumps({k: summary[k] for k in ("chosen_lambda", "players_with_both_supported", "review_status", "boundary_violations_on_accepted_nfl")}), flush=True)
    return summary


def _figures(out: Path, summary: dict, calibration: list[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labels = [row["slice"] for row in calibration if row["slice"].startswith("family") or row["slice"].startswith("speed")]
    means = [row["mean_residual"] for row in calibration if row["slice"] in labels]
    ax.axhline(0, color="black", lw=0.8)
    ax.bar(range(len(labels)), means, color="#0072B2")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Out-of-fold residual (yards)")
    ax.set_title("Combine exit residuals after the chosen penalty")
    fig.tight_layout()
    fig.savefig(out / "fig_calibration.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(
        ["Combine cuts", "NFL cuts", "Players with both", "2024 players with both"],
        [
            summary["combine_single_cuts"],
            summary["nfl_single_cuts"],
            summary["players_with_both_supported"],
            summary["players_with_both_supported_2024"],
        ],
        color=["#0072B2", "#009E73", "#D55E00", "#E69F00"],
    )
    ax.set_ylabel("Count")
    ax.set_title("Candidate coverage after the one-break and support rules")
    fig.tight_layout()
    fig.savefig(out / "fig_coverage.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
