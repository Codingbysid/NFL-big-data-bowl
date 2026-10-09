"""Post-evaluation repairs. The original 2025 freeze is not replaced.

New numbers written here are diagnostics or retrospective checks. They are
not a second holdout test.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bdb2027.config import Project, code_hash, presentation_fingerprint, scientific_fingerprint
from bdb2027.evaluation import summarize_frames
from bdb2027.features import (
    add_residuals,
    apply_shrinkage,
    collapse_attempts,
    fit_trait_bundle,
    reliability_icc,
    trait_table,
)
from bdb2027.io import append_experiment
from bdb2027.models import choose_alpha, fit_predict_split
from bdb2027.pipeline import _eligible_nfl_cuts, _metrics, _restrict, load_cuts, load_routes, oof_measurement

ORIGINAL_CODE_HASH = "a61da719cfad7712dc182e84bdd3d345479615745b40e344c4f635787e5dbdaf"
ORIGINAL_CONFIG_HASH = "dfc5bbb37f447ee85ca1eea3d26d2c0e7a746567c39394c2f78be23be0fb3505"
ORIGINAL_FROZEN_AT = "2026-10-08T08:54:45Z"


def archive_original(project: Project) -> Path:
    """Copy the evaluated artifacts. Do not rewrite the historical manifest."""
    dest = project.root / "archive" / "original_evaluation"
    dest.mkdir(parents=True, exist_ok=True)
    note = dest / "SOURCE_RECOVERY.md"
    if (dest / "models.joblib").exists() and note.exists():
        return dest
    pairs = [
        (project.reports / "freeze_manifest.json", dest / "freeze_manifest.json"),
        (project.interim / "frozen" / "manifest.json", dest / "frozen_manifest_copy.json"),
        (project.interim / "frozen" / "models.joblib", dest / "models.joblib"),
        (project.interim / "development" / "metrics.json", dest / "development_metrics.json"),
        (project.interim / "development" / "predictions.parquet", dest / "development_predictions.parquet"),
        (project.interim / "heldout" / "metrics.json", dest / "heldout_metrics.json"),
        (project.interim / "heldout" / "predictions.parquet", dest / "heldout_predictions.parquet"),
        (project.interim / "heldout" / "combine_traits.parquet", dest / "holdout_combine_traits.parquet"),
        (project.interim / "heldout" / "nfl_traits.parquet", dest / "holdout_nfl_traits.parquet"),
        (project.interim / "heldout" / "marginal.parquet", dest / "marginal.parquet"),
        (project.interim / "heldout" / "player_predictions.parquet", dest / "player_predictions.parquet"),
        (project.outputs / "predictions.csv", dest / "predictions.csv"),
        (project.reports / "protocol.md", dest / "protocol.md"),
        (project.reports / "experiment_log.csv", dest / "experiment_log_at_archive.csv"),
        (project.root / "config" / "research.yaml", dest / "research.yaml"),
    ]
    for src, target in pairs:
        if src.exists():
            shutil.copy2(src, target)
    # Source snapshot matching the evaluated hash: current tree, with the one
    # post-freeze plotting edit reverted inside the archive only.
    snap = dest / "source_snapshot" / "bdb2027"
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(project.root / "src" / "bdb2027", snap, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    plots = snap / "plots.py"
    plots.write_text(
        plots.read_text().replace(
            """            if annot[i][j]:
                ink = "black" if grid[i, j] >= 0.5 else "white"
                ax.text(j, i, annot[i][j], ha="center", va="center", color=ink, fontsize=8)""",
            """            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center", color="white", fontsize=8)""",
        )
    )
    note.write_text(
        f"""# Original evaluation archive

Git had no commits, so the evaluated source was not in history. The only package edit after `{ORIGINAL_FROZEN_AT}` was a plotting-ink change in `plots.py`. Reverting that line inside this snapshot reproduces code hash `{ORIGINAL_CODE_HASH}`.

The manifest, model object, and prediction files in this directory are copies of the original freeze. They are not a new freeze.

- frozen_at: {ORIGINAL_FROZEN_AT}
- config_hash: {ORIGINAL_CONFIG_HASH}
- code_hash: {ORIGINAL_CODE_HASH}
- training players: 49
- training routes: 10603
- evaluation players: 19
- evaluation routes: 3552

A later plotting or text edit must not be described as a refit. Repeating `freeze` after viewing 2025 does not create a new untouched holdout.
"""
    )
    return dest


def _support_lookup(manifest: dict) -> set[tuple[str, str]]:
    return {
        (row["speed_bin"], row["angle_bin"])
        for row in manifest.get("support_cells", [])
        if row.get("supported")
    }


def _cell_labels(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = frame.copy()
    out["speed_bin"] = pd.cut(out["speed_in"], cfg["support"]["speed_edges"], include_lowest=True).astype(str)
    out["angle_bin"] = pd.cut(out["abs_angle_deg"], cfg["support"]["angle_edges"], include_lowest=True).astype(str)
    return out


def export_measurements(project: Project) -> dict:
    """Compact cut and player tables from the original extraction and frozen demand model."""
    out = project.outputs / "post_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    cfg = project.config
    bundle = joblib.load(project.interim / "frozen" / "models.joblib")
    manifest = json.loads((project.reports / "freeze_manifest.json").read_text())
    demand = bundle["demand"]
    supported = _support_lookup(manifest)
    parts = []
    for year in (2023, 2024, 2025):
        cuts = load_cuts(project, [year], "combine", 5)
        if cuts.empty:
            continue
        primary = cuts.loc[cuts["drill_role"].eq("primary")].copy()
        scored = add_residuals(primary, demand)
        scored = _cell_labels(scored, cfg)
        scored["in_final_support"] = [
            (s, a) in supported for s, a in zip(scored["speed_bin"], scored["angle_bin"])
        ]
        scored["residual_fit"] = "final_2023_24_demand"
        scored["residual_is_out_of_sample"] = year == 2025
        scored["cohort"] = year
        parts.append(scored)
    combine = pd.concat(parts, ignore_index=True)
    combine["cuts_in_attempt"] = combine.groupby("attempt_uid")["attempt_uid"].transform("size")
    combine.to_parquet(out / "combine_cuts_window5.parquet", index=False)

    nfl_parts = []
    for year in (2023, 2024, 2025):
        nfl = _eligible_nfl_cuts(load_cuts(project, [year], "nfl", 5), cfg)
        if nfl.empty:
            continue
        scored = add_residuals(nfl, bundle["nfl_demand"])
        scored = _cell_labels(scored, cfg)
        scored["in_final_support"] = [
            (s, a) in supported for s, a in zip(scored["speed_bin"], scored["angle_bin"])
        ]
        scored["residual_fit"] = "final_2023_24_nfl_demand"
        scored["residual_is_out_of_sample"] = year == 2025
        nfl_parts.append(scored)
    nfl = pd.concat(nfl_parts, ignore_index=True) if nfl_parts else pd.DataFrame()
    if len(nfl):
        nfl.to_parquet(out / "nfl_cuts_window5.parquet", index=False)

    dev_cuts = combine.loc[combine["draft_year"].isin([2023, 2024])]
    attempts = collapse_attempts(dev_cuts)
    # Equal-attempt check: collapse_attempts averages cuts inside an attempt.
    population = {
        "primary_cuts_2023_24": int(len(dev_cuts)),
        "players_with_any_primary_cut_2023_24": int(dev_cuts["nfl_id"].nunique()),
        "attempts_2023_24": int(dev_cuts["attempt_uid"].nunique()),
        "attempts_with_multiple_cuts": int((dev_cuts.groupby("attempt_uid").size() > 1).sum()),
        "cuts_in_final_support_2023_24": int(dev_cuts["in_final_support"].sum()),
        "note": (
            "The 406-cut feasibility count is every primary-family Combine cut in 2023-24 "
            "after the entry-speed gate. It is not restricted to common-support cells or to "
            "the 49 modeled receivers. The primary trait uses all primary-family cuts for a "
            "player, not only support cells. Support restricts the translation claim."
        ),
    }
    train_ids = set(manifest["train_ids"])
    modeled = dev_cuts.loc[dev_cuts["nfl_id"].isin(train_ids)]
    population["primary_cuts_among_49_training_players"] = int(len(modeled))
    population["training_players_with_a_cut"] = int(modeled["nfl_id"].nunique())
    (out / "population.json").write_text(json.dumps(population, indent=2))

    # Out-of-fold 2023 residuals for the reliability claim. Does not use 2025 outcomes.
    cuts_2023 = load_cuts(project, [2023], "combine", 5)
    traits, oof_attempts, icc = oof_measurement(cuts_2023, cfg, prefix="")
    oof_attempts.to_parquet(out / "oof_2023_attempt_residuals.parquet", index=False)
    traits.to_parquet(out / "oof_2023_player_traits.parquet", index=False)
    attempts.to_parquet(out / "final_fit_2023_24_attempt_residuals.parquet", index=False)

    hold_traits = pd.read_parquet(project.interim / "heldout" / "combine_traits.parquet")
    hold_nfl = pd.read_parquet(project.interim / "heldout" / "nfl_traits.parquet")
    hold_traits.to_parquet(out / "holdout_combine_traits.parquet", index=False)
    hold_nfl.to_parquet(out / "holdout_nfl_traits.parquet", index=False)
    both = hold_traits.merge(hold_nfl, on="nfl_id", how="inner")
    both.to_parquet(out / "translation_2025_rows.parquet", index=False)
    trans_2024 = pd.read_parquet(project.interim / "development" / "translation_2024.parquet")
    trans_2024.to_parquet(out / "translation_2024_rows.parquet", index=False)

    reliability = _reliability_report(oof_attempts, dev_cuts)
    reliability["exported_icc_matches_development_file"] = bool(
        np.isclose(reliability["icc_anova"]["icc"], icc.get("icc", np.nan), atol=1e-8)
    )
    (out / "reliability.json").write_text(json.dumps(reliability, indent=2, default=_json))

    calibration = _demand_calibration(oof_attempts if len(oof_attempts) else attempts)
    calibration.to_csv(out / "demand_calibration_oof_2023.csv", index=False)
    return {"population": population, "reliability": reliability, "icc_runtime": icc}


def _json(value):
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return str(value)


def _reliability_report(attempts: pd.DataFrame, all_cuts: pd.DataFrame) -> dict:
    """ANOVA ICC plus the SD of receiver means, labeled separately."""
    point = reliability_icc(attempts)
    groups = []
    for _, g in attempts.groupby("nfl_id"):
        values = g["residual"].dropna().to_numpy(dtype=float)
        if len(values) >= 2:
            groups.append(values)
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    grand = np.average(means, weights=ns)
    ssw = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    msw = ssw / (ns.sum() - len(groups))
    ssb = float(np.sum(ns * (means - grand) ** 2))
    msb = ssb / (len(groups) - 1)
    n0 = (ns.sum() - np.sum(ns**2) / ns.sum()) / (len(groups) - 1)
    latent_var = (msb - msw) / n0
    # Same drill family, same player, at least two attempts. Not a coaching threshold.
    prim = all_cuts.loc[all_cuts["drill_role"].eq("primary")] if "drill_role" in all_cuts.columns else all_cuts
    if "residual" not in prim.columns and "attempt_uid" in prim.columns:
        family_attempts = prim.groupby(["nfl_id", "drill_family"])["attempt_uid"].nunique()
    else:
        family_attempts = prim.groupby(["nfl_id", "drill_family"])["attempt_uid"].nunique() if len(prim) else pd.Series(dtype=int)
    repeat_groups = int((family_attempts >= 2).sum()) if len(family_attempts) else 0
    same = []
    if "residual" in attempts.columns and "drill_family" in attempts.columns:
        for (_, _), g in attempts.groupby(["nfl_id", "drill_family"]):
            if len(g) >= 2:
                same.append(float(g["residual"].std(ddof=1)))
    return {
        "label": "cross-drill consistency of out-of-fold attempt residuals; most drills occur once",
        "formula": "(MSB - MSW) / (MSB + (n0 - 1) * MSW), one-way random effects, players with >= 2 attempts",
        "bootstrap_unit": "players, with duplicate draws relabeled so a player is not pooled with himself",
        "icc_anova": point,
        "msw": msw,
        "msb": msb,
        "n0_harmonic": float(n0),
        "within_sd_sqrt_msw": float(np.sqrt(max(msw, 0.0))),
        "sd_of_receiver_means": float(np.std(means, ddof=1)),
        "latent_between_sd": float(np.sqrt(latent_var)) if latent_var > 0 else None,
        "latent_between_note": (
            "The SD of receiver means includes estimation error. "
            "The latent between-receiver SD is sqrt(max(0, (MSB-MSW)/n0))."
        ),
        "same_drill_family_groups_with_2plus_attempts": repeat_groups,
        "same_drill_within_sd_values_n": len(same),
        "same_drill_within_sd_median": float(np.median(same)) if same else None,
        "repeatability_threshold": (
            "Not estimated. Comparable same-condition repeats are too few to support a "
            "detectable-change rule. The 0.3-yard practice statement is withdrawn."
        ),
        "trait_reliability_of_the_mean": (
            None
            if latent_var <= 0
            else float(latent_var / (latent_var + msw / n0))
        ),
    }


def _demand_calibration(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if attempts.empty or "residual" not in attempts.columns:
        return pd.DataFrame()
    for family, g in attempts.groupby("drill_family"):
        rows.append({"slice": f"family:{family}", "n": int(len(g)), "mean_residual": float(g["residual"].mean())})
    attempts = attempts.copy()
    attempts["speed_bin"] = pd.cut(attempts["speed_in"], [2, 4, 6, 8, 13], include_lowest=True)
    attempts["angle_bin"] = pd.cut(attempts["abs_angle_deg"], [30, 50, 70, 90, 110], include_lowest=True)
    for label, col in (("speed", "speed_bin"), ("angle", "angle_bin")):
        for key, g in attempts.groupby(col, observed=False):
            if len(g) == 0:
                continue
            rows.append({"slice": f"{label}:{key}", "n": int(len(g)), "mean_residual": float(g["residual"].mean())})
    return pd.DataFrame(rows)


def common_penalty_diagnostic(project: Project) -> dict:
    """Same ridge penalty for M1 and M2, chosen on 2023-24 grouped folds only."""
    out = project.outputs / "post_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    cfg = project.config
    routes, _ = load_routes(project, [2023, 2024], allow_outcomes=False)
    cuts = load_cuts(project, [2023, 2024], "combine", 5)
    train = _restrict(routes, int(cfg["routes"]["min_routes"]))
    alphas = [float(a) for a in cfg["models"]["alphas"]]
    n_splits = min(int(cfg["models"]["inner_folds"]), train["nfl_id"].nunique())
    splitter = GroupKFold(n_splits=n_splits)
    scores = {a: {"M1": [], "M2": []} for a in alphas}
    train = train.reset_index(drop=True)
    for train_idx, test_idx in splitter.split(train, groups=train["nfl_id"]):
        tr, te = train.iloc[train_idx], train.iloc[test_idx]
        for alpha in alphas:
            for level in ("M1", "M2"):
                pred, *_ = fit_predict_split(tr, te, cuts, level, alpha, cfg)
                block = _metrics(te, {level: pred})
                scores[alpha][level].append(block[level]["mae"])
    summary = []
    for alpha in alphas:
        m1 = float(np.mean(scores[alpha]["M1"]))
        m2 = float(np.mean(scores[alpha]["M2"]))
        summary.append({"alpha": alpha, "m1_oof_mae": m1, "m2_oof_mae": m2, "mean_mae": 0.5 * (m1 + m2), "delta": m2 - m1})
    chosen = min(summary, key=lambda row: row["mean_mae"])["alpha"]
    # Retrospective 2025 application. The penalty was not chosen on 2025.
    hold_routes, _ = load_routes(project, [2025], allow_outcomes=True)
    hold = _restrict(hold_routes, int(cfg["routes"]["min_routes"]))
    hold_cuts = load_cuts(project, [2023, 2024, 2025], "combine", 5)
    preds = {}
    for level in ("M1", "M2"):
        pred, *_ = fit_predict_split(train, hold, hold_cuts, level, chosen, cfg)
        preds[level] = pred
    retrospective = _metrics(hold, preds)
    retrospective["label"] = "post-evaluation common-penalty diagnostic; 2025 was not used to choose alpha"
    retrospective["alpha"] = chosen
    retrospective["development_grid"] = summary
    (out / "common_penalty.json").write_text(json.dumps(retrospective, indent=2))
    return retrospective


def decompose_examples(project: Project) -> pd.DataFrame:
    """Split Noel/Lane/Thornton prediction gaps into trait, shared terms, and intercept."""
    bundle = joblib.load(project.interim / "frozen" / "models.joblib")
    traits = pd.read_parquet(project.interim / "heldout" / "combine_traits.parquet")
    routes, _ = load_routes(project, [2025], allow_outcomes=True)
    rows = []
    for nfl_id, role in ((58284, "rule_selected_noel"), (58333, "rule_selected_lane"), (58313, "posthoc_failure_thornton")):
        part = routes.loc[routes["nfl_id"] == nfl_id].copy()
        part = part.merge(traits, on="nfl_id", how="left")
        b1, m1 = bundle["models"]["M1"]
        b2, m2 = bundle["models"]["M2"]
        x1 = b1.transform(part)
        x2 = b2.transform(part)
        p1 = float(m1.intercept_ + x1.mean(axis=0) @ m1.coef_)
        p2 = float(m2.intercept_ + x2.mean(axis=0) @ m2.coef_)
        trait_idx = [i for i, name in enumerate(b2.columns_) if name == "trait" or name.startswith("trait")]
        trait_part = float(sum(x2.mean(axis=0)[i] * m2.coef_[i] for i in trait_idx))
        shared = [name for name in b1.columns_ if name in b2.columns_]
        shared_delta = 0.0
        for name in shared:
            i1 = b1.columns_.index(name)
            i2 = b2.columns_.index(name)
            shared_delta += float(x2.mean(axis=0)[i2] * m2.coef_[i2] - x1.mean(axis=0)[i1] * m1.coef_[i1])
        only_m1 = [name for name in b1.columns_ if name not in b2.columns_]
        only_m1_part = float(sum(x1.mean(axis=0)[b1.columns_.index(name)] * m1.coef_[b1.columns_.index(name)] for name in only_m1))
        intercept_delta = float(m2.intercept_ - m1.intercept_)
        total = p2 - p1
        reconstructed = intercept_delta + trait_part + shared_delta - only_m1_part
        # Within M2, zero the standardized trait (training-mean contrast).
        x2_zero = x2.copy()
        for i in trait_idx:
            x2_zero[:, i] = 0.0
        p2_zero = float(m2.intercept_ + x2_zero.mean(axis=0) @ m2.coef_)
        trait_value = float(part["trait"].iloc[0]) if part["trait"].notna().any() else np.nan
        rows.append(
            {
                "nfl_id": nfl_id,
                "role": role,
                "trait": trait_value,
                "mean_pred_m1": p1,
                "mean_pred_m2": p2,
                "total_m2_minus_m1": total,
                "intercept_difference": intercept_delta,
                "direct_trait_term": trait_part,
                "shared_features_after_each_models_scaling": shared_delta,
                "m1_only_features": only_m1_part,
                "reconstructed_difference": reconstructed,
                "reconstruction_error": reconstructed - total,
                "within_m2_trait_contrast": p2 - p2_zero,
                "poor_exit_rewarded": bool(trait_value < 0 and (p2 - p2_zero) > 0),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(project.outputs / "post_evaluation" / "prediction_decomposition.csv", index=False)
    return frame


def _require_addendum(project: Project) -> None:
    path = project.reports / "post_evaluation_addendum.md"
    text = path.read_text() if path.exists() else ""
    if "PREDECLARED" not in text or "Drill dependence" not in text or "Endpoint alignment" not in text:
        raise RuntimeError("Write the post-evaluation addendum before running exploratory diagnostics.")


def exploratory(project: Project) -> dict:
    """Two predeclared development diagnostics. 2025 is not used to choose anything."""
    _require_addendum(project)
    out = project.outputs / "post_evaluation"
    cfg = project.config
    cuts = load_cuts(project, [2023, 2024], "combine", 5)
    cuts = cuts.loc[cuts["drill_role"].eq("primary")].copy()
    bundle_model = joblib.load(project.interim / "frozen" / "models.joblib")["demand"]
    scored = collapse_attempts(add_residuals(cuts, bundle_model))
    families = [f for f, n in scored["drill_family"].value_counts().items() if n >= 30]
    rows = []
    for family in families:
        kept = scored.loc[scored["drill_family"] != family]
        full = scored.groupby("nfl_id")["residual"].median()
        left = kept.groupby("nfl_id")["residual"].median()
        both = pd.DataFrame({"full": full, "left": left}).dropna()
        both = both.loc[kept.groupby("nfl_id").size().reindex(both.index).fillna(0) >= 2]
        rho = float(both["full"].corr(both["left"], method="spearman")) if len(both) >= 8 else np.nan
        rows.append({"left_out_family": family, "players": int(len(both)), "spearman_with_full_trait": rho})
    drill = pd.DataFrame(rows)
    drill.to_csv(out / "exploratory_drill_dependence.csv", index=False)

    routes, _ = load_routes(project, [2023, 2024], allow_outcomes=False)
    routes = _restrict(routes, int(cfg["routes"]["min_routes"]))
    nfl = _eligible_nfl_cuts(load_cuts(project, [2023, 2024], "nfl", 5), cfg)
    manifest = json.loads((project.reports / "freeze_manifest.json").read_text())
    supported = _support_lookup(manifest)
    nfl = _cell_labels(nfl, cfg)
    nfl["in_final_support"] = [(s, a) in supported for s, a in zip(nfl["speed_bin"], nfl["angle_bin"])]
    cut_keys = set(zip(nfl.loc[nfl["in_final_support"], "game_id"], nfl.loc[nfl["in_final_support"], "play_id"], nfl.loc[nfl["in_final_support"], "nfl_id"]))
    routes["has_supported_cut"] = [
        (int(g), int(p), int(i)) in cut_keys
        for g, p, i in zip(routes["game_id"], routes["play_id"], routes["nfl_id"])
    ]
    traits = scored.groupby("nfl_id")["residual"].median()
    all_mean = routes.groupby("nfl_id")["separation_at_pass_forward"].mean()
    cut_routes = routes.loc[routes["has_supported_cut"]]
    cut_mean = cut_routes.groupby("nfl_id")["separation_at_pass_forward"].mean()
    joined = pd.DataFrame({"trait": traits, "all_routes": all_mean, "cut_routes": cut_mean}).dropna(subset=["trait"])
    coverage = {
        "modeled_players": int(routes["nfl_id"].nunique()),
        "players_with_a_supported_nfl_cut": int(cut_routes["nfl_id"].nunique()),
        "eligible_routes": int(len(routes)),
        "routes_with_supported_cut": int(routes["has_supported_cut"].sum()),
        "spearman_trait_vs_all_route_mean": float(joined["trait"].corr(joined["all_routes"], method="spearman")),
        "spearman_trait_vs_cut_route_mean": float(joined.dropna(subset=["cut_routes"])["trait"].corr(joined.dropna(subset=["cut_routes"])["cut_routes"], method="spearman")) if joined["cut_routes"].notna().sum() >= 8 else None,
        "n_for_cut_correlation": int(joined["cut_routes"].notna().sum()),
        "label": "development only; not used to revise the 2025 model",
    }
    (out / "exploratory_endpoint.json").write_text(json.dumps(coverage, indent=2))
    return {"drill": rows, "endpoint": coverage}


def run_all(project: Project | None = None) -> None:
    project = project or Project()
    archive_original(project)
    export_measurements(project)
    print("[posteval] common penalty", flush=True)
    common = common_penalty_diagnostic(project)
    print("[posteval] common alpha", common["alpha"], "retrospective delta", common.get("delta_m2_minus_m1"), flush=True)
    decomp = decompose_examples(project)
    print(decomp.to_string(index=False), flush=True)
    print("[posteval] exploratory", flush=True)
    exploratory(project)
    append_experiment(
        project,
        "post-evaluation",
        "diagnostics",
        "Archived original freeze. Exported cuts, reliability components, common-penalty diagnostic, decomposition, and two predeclared development checks. 2025 was not used to choose them.",
        "ok",
        False,
    )


if __name__ == "__main__":
    run_all()
