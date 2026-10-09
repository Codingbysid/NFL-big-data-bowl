"""Development, freeze, held-out evaluation, and the evidence report.

The 2025 class is not read until evaluate-heldout. That command checks the
freeze manifest and does not refit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bdb2027.cohort import apply_funnel, assert_no_cohort_overlap, prepare_plays
from bdb2027.config import Project, code_hash, config_hash
from bdb2027.evaluation import (
    equal_player_weights,
    leave_one_player_influence,
    paired_bootstrap_delta,
    summarize_frames,
)
from bdb2027.extract import cohort_dir, extract
from bdb2027.features import (
    add_residuals,
    apply_shrinkage,
    bootstrap_icc,
    collapse_attempts,
    fit_trait_bundle,
    reliability_icc,
    support_mask,
    trait_table,
)
from bdb2027.io import append_experiment, cached_fingerprints, data_version
from bdb2027.models import (
    LEVEL_NUMERIC,
    DesignBuilder,
    attach_combine_trait,
    choose_alpha,
    fit_level,
    fit_predict_split,
    predict_level,
)
from bdb2027.plots import (
    plot_failures,
    plot_model_comparison,
    plot_profile,
    plot_support,
    plot_trajectory,
    plot_translation,
)

PRIMARY_LEVELS = ["M0", "M1", "M2"]
SENSITIVITY_LEVELS = ["M2_speed", "M2_shrunk", "M1_forty", "M1_ngs", "M2_ngs", "M2_draft"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def _require_protocol(project: Project) -> None:
    path = project.reports / "protocol.md"
    if not path.exists():
        raise RuntimeError("reports/protocol.md is missing. Run `python -m bdb2027 protocol` before develop.")
    if config_hash(project.config) not in path.read_text():
        raise RuntimeError(
            "protocol.md does not match the current configuration hash. "
            "Rewrite the protocol before fitting outcome models."
        )


def load_cuts(project: Project, years: list[int], source: str, window: int) -> pd.DataFrame:
    parts = []
    for year in years:
        path = cohort_dir(project, int(year)) / f"{source}_cuts.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run extract for {year} first.")
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame.loc[frame["sg_window"] == int(window)].copy()
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["nfl_id"] = out["nfl_id"].astype(int)
    return out


def load_routes(project: Project, years: list[int], allow_outcomes: bool) -> tuple[pd.DataFrame, pd.Series]:
    parts = []
    for year in years:
        path = cohort_dir(project, int(year)) / "plays.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run extract for {year} first.")
        parts.append(pd.read_parquet(path))
    plays = prepare_plays(pd.concat(parts, ignore_index=True))
    if "nfl_position" not in plays.columns:
        plays["nfl_position"] = project.config["position"]
    plays = apply_funnel(plays, project.config)
    plays = plays.loc[plays["eligible"]].copy()
    holdout = int(project.config["holdout_min_draft_year"])
    if (plays["draft_year"].astype(int) >= holdout).any() and not allow_outcomes:
        raise PermissionError("Held-out routes were loaded before evaluate-heldout.")
    players = pd.read_parquet(project.interim / "wr_players.parquet")
    names = players.set_index("nfl_id")["display_name"]
    keep = [
        "nfl_id", "draft_year", "height", "weight", "age", "age_convention", "forty", "ten_yd_split",
        "vertical", "broad_jump", "three_cone", "short_shuttle", "ngs_athleticism_score",
        "ngs_college_production_score", "ngs_final_score", "undrafted", "draft_overall_pick",
    ]
    merged = plays.merge(players[keep], on=["nfl_id", "draft_year"], how="left", validate="many_to_one")
    merged["nfl_id"] = merged["nfl_id"].astype(int)
    merged["draft_year"] = merged["draft_year"].astype(int)
    if merged["separation_at_pass_forward"].isna().any():
        raise ValueError("Eligible routes include missing separation. They must stay excluded, not filled.")
    return merged, names


def _restrict(routes: pd.DataFrame, minimum: int) -> pd.DataFrame:
    counts = routes.groupby("nfl_id").size()
    keep = set(counts[counts >= minimum].index.astype(int))
    return routes.loc[routes["nfl_id"].isin(keep)].copy()


def _metrics(routes: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    y = routes["separation_at_pass_forward"].to_numpy(dtype=float)
    out = {}
    for level, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)
        ok = np.isfinite(y) & np.isfinite(pred)
        if ok.sum() == 0:
            out[level] = {"n_players": 0, "mae": None}
            continue
        idx = np.flatnonzero(ok)
        out[level] = summarize_frames(routes.iloc[idx]["nfl_id"], y[idx], pred[idx])
    if "M1" in out and "M2" in out and out["M1"].get("mae") is not None and out["M2"].get("mae") is not None:
        out["delta_m2_minus_m1"] = out["M2"]["mae"] - out["M1"]["mae"]
    return out


def _prediction_frame(routes: pd.DataFrame, predictions: dict[str, np.ndarray], split: str) -> pd.DataFrame:
    out = routes[["nfl_id", "draft_year", "game_id", "play_id", "route_ran", "coverage_man_zone"]].copy()
    out = out.rename(columns={"coverage_man_zone": "coverage"})
    out["y"] = routes["separation_at_pass_forward"].to_numpy(dtype=float)
    for level, pred in predictions.items():
        out[f"pred_{level}"] = np.asarray(pred, dtype=float)
    out["split"] = split
    return out


def oof_predictions(routes: pd.DataFrame, cuts: pd.DataFrame, level: str, cfg: dict):
    routes = routes.reset_index(drop=True)
    n_groups = int(routes["nfl_id"].nunique())
    n_splits = min(int(cfg["models"]["inner_folds"]), n_groups)
    pred = np.full(len(routes), np.nan)
    alphas = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(routes, groups=routes["nfl_id"]), start=1):
        train_ids = set(routes.iloc[train_idx]["nfl_id"].astype(int))
        test_ids = set(routes.iloc[test_idx]["nfl_id"].astype(int))
        assert_no_cohort_overlap(train_ids, test_ids)
        alpha = choose_alpha(routes.iloc[train_idx], cuts, level, cfg)
        predicted, *_ = fit_predict_split(routes.iloc[train_idx], routes.iloc[test_idx], cuts, level, alpha, cfg)
        pred[test_idx] = predicted
        alphas.append(alpha)
        print(f"  oof {level} fold {fold}/{n_splits} alpha={alpha}", flush=True)
    return pred, alphas


def _eligible_nfl_cuts(cuts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cuts.empty:
        return cuts
    allowed = set(cfg["routes"]["eligible"])
    return cuts.loc[cuts["route_ran"].isin(allowed)].copy()


def oof_measurement(cuts: pd.DataFrame, cfg: dict, prefix: str = ""):
    """Player-held-out demand adjustment. Used for reliability and translation."""
    players = np.array(sorted(cuts.loc[cuts["drill_role"].eq("primary"), "nfl_id"].unique()))
    if len(players) < 5:
        return pd.DataFrame(), pd.DataFrame(), {}
    dummy = pd.DataFrame({"nfl_id": players})
    n_splits = min(int(cfg["models"]["inner_folds"]), len(players))
    splitter = GroupKFold(n_splits=n_splits)
    trait_parts = []
    attempt_parts = []
    for train_idx, test_idx in splitter.split(dummy, groups=dummy["nfl_id"]):
        train_ids = set(dummy.iloc[train_idx]["nfl_id"].astype(int))
        test_ids = set(dummy.iloc[test_idx]["nfl_id"].astype(int))
        model, traits, _n0 = fit_trait_bundle(cuts, train_ids, cfg, prefix=prefix)
        trait_parts.append(traits.loc[traits["nfl_id"].isin(test_ids)])
        test_cuts = cuts.loc[cuts["nfl_id"].isin(test_ids) & cuts["drill_role"].eq("primary")]
        if not test_cuts.empty:
            attempt_parts.append(collapse_attempts(add_residuals(test_cuts, model)))
    traits = pd.concat(trait_parts, ignore_index=True) if trait_parts else pd.DataFrame()
    attempts = pd.concat(attempt_parts, ignore_index=True) if attempt_parts else pd.DataFrame()
    icc = bootstrap_icc(attempts, n_boot=400, seed=int(cfg["models"]["random_seed"])) if len(attempts) else {}
    return traits, attempts, icc


def _man_metrics(routes: pd.DataFrame, predictions: dict[str, np.ndarray], cfg: dict) -> dict:
    mask = routes["coverage_man_zone"].astype(str).eq("MAN_COVERAGE").to_numpy()
    part = routes.loc[mask].copy()
    part_preds = {level: np.asarray(pred)[mask] for level, pred in predictions.items() if level in PRIMARY_LEVELS}
    minimum = int(cfg["models"]["man_coverage_min_routes"])
    kept = _restrict(part, minimum)
    if kept.empty:
        return {"n_players": 0}
    # Predictions were filtered with `mask` in the original row order. Realign by index.
    aligned = {}
    for level, pred in part_preds.items():
        series = pd.Series(pred, index=part.index)
        aligned[level] = series.loc[kept.index].to_numpy()
    return _metrics(kept, aligned)


def _threshold_metrics(routes: pd.DataFrame, predictions: dict[str, np.ndarray], cfg: dict) -> dict:
    out = {}
    counts = routes.groupby("nfl_id").size()
    for minimum in cfg["routes"]["min_routes_sensitivity"]:
        keep = set(counts[counts >= int(minimum)].index.astype(int))
        mask = routes["nfl_id"].isin(keep).to_numpy()
        part = routes.loc[mask]
        part_preds = {level: np.asarray(pred)[mask] for level, pred in predictions.items() if level in PRIMARY_LEVELS}
        out[str(minimum)] = _metrics(part, part_preds)
    return out


def _shuffle_profiles(routes: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = routes.copy()
    players = out["nfl_id"].drop_duplicates().to_numpy()
    donors = rng.permutation(players)
    mapping = dict(zip(players, donors))
    profile = out.groupby("nfl_id")[columns].first()
    donor_id = out["nfl_id"].map(mapping)
    for column in columns:
        out[column] = donor_id.map(profile[column]).to_numpy()
    return out


def _permutation(train: pd.DataFrame, test: pd.DataFrame, cuts: pd.DataFrame, alpha_m1: float, alpha_m2: float, cfg: dict) -> dict:
    rng = np.random.default_rng(int(cfg["models"]["random_seed"]))
    train_ids = set(train["nfl_id"].astype(int))
    base, _ = attach_combine_trait(train, cuts, train_ids, cfg)
    test_scored, _ = attach_combine_trait(test, cuts, train_ids, cfg)
    columns = [c for c in ["height", "weight", "age", "forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle", "trait", "entry_speed", "trait_shrunk", "undrafted", "draft_overall_pick"] if c in base.columns]
    reps = int(cfg["models"]["permutation_reps"])
    deltas = []
    for rep in range(reps):
        shuffled = _shuffle_profiles(base, columns, rng)
        pred_m1, *_ = fit_predict_split(shuffled, test_scored, cuts, "M1", alpha_m1, cfg, recompute_trait=False)
        pred_m2, *_ = fit_predict_split(shuffled, test_scored, cuts, "M2", alpha_m2, cfg, recompute_trait=False)
        block = _metrics(test_scored, {"M1": pred_m1, "M2": pred_m2})
        deltas.append(block.get("delta_m2_minus_m1"))
        if (rep + 1) % 5 == 0:
            print(f"  permutation {rep + 1}/{reps}", flush=True)
    arr = np.array(deltas, dtype=float)
    return {
        "n": reps,
        "mean_delta": float(np.nanmean(arr)),
        "p2_5": float(np.nanpercentile(arr, 2.5)),
        "p97_5": float(np.nanpercentile(arr, 97.5)),
        "label": "combine profiles shuffled within the 2023 training class; 2024 routes unchanged",
    }


def _refit_bootstrap(train: pd.DataFrame, test: pd.DataFrame, cuts: pd.DataFrame, alpha_m1: float, alpha_m2: float, cfg: dict) -> dict:
    rng = np.random.default_rng(int(cfg["models"]["random_seed"]) + 1)
    ids = train["nfl_id"].drop_duplicates().to_numpy()
    reps = int(cfg["models"]["bootstrap_refit"])
    deltas = []
    for rep in range(reps):
        draws = rng.choice(ids, size=len(ids), replace=True)
        route_parts = []
        cut_parts = []
        for k, nfl in enumerate(draws):
            routes = train.loc[train["nfl_id"] == nfl].copy()
            routes["nfl_id"] = 10_000_000 + k
            route_parts.append(routes)
            taken = cuts.loc[cuts["nfl_id"] == nfl].copy()
            if not taken.empty:
                taken["nfl_id"] = 10_000_000 + k
                taken["attempt_uid"] = taken["attempt_uid"].astype(str) + f"|b{k}"
                cut_parts.append(taken)
        boot_routes = pd.concat(route_parts, ignore_index=True)
        boot_cuts = pd.concat(cut_parts + [cuts.loc[cuts["nfl_id"].isin(set(test["nfl_id"]))]], ignore_index=True)
        pred_m1, *_ = fit_predict_split(boot_routes, test, boot_cuts, "M1", alpha_m1, cfg)
        pred_m2, *_ = fit_predict_split(boot_routes, test, boot_cuts, "M2", alpha_m2, cfg)
        block = _metrics(test, {"M1": pred_m1, "M2": pred_m2})
        deltas.append(block.get("delta_m2_minus_m1"))
        if (rep + 1) % 10 == 0:
            print(f"  refit bootstrap {rep + 1}/{reps}", flush=True)
    arr = np.array(deltas, dtype=float)
    return {
        "n": reps,
        "mean_delta": float(np.nanmean(arr)),
        "p2_5": float(np.nanpercentile(arr, 2.5)),
        "p97_5": float(np.nanpercentile(arr, 97.5)),
        "label": "development players resampled; models refit at the frozen penalty; 2024 players fixed",
    }


def _correlation(frame: pd.DataFrame, left: str, right: str) -> dict:
    part = frame.dropna(subset=[left, right])
    if len(part) < 5:
        return {"n": int(len(part)), "spearman": None, "pearson": None}
    return {
        "n": int(len(part)),
        "spearman": float(part[left].corr(part[right], method="spearman")),
        "pearson": float(part[left].corr(part[right], method="pearson")),
    }


def develop(project: Project) -> None:
    _require_protocol(project)
    if (cohort_dir(project, 2025) / "plays.parquet").exists():
        print("[develop] held-out extract exists and will not be read", flush=True)
    cfg = project.config
    print("[develop] loading 2023-2024", flush=True)
    routes, _names = load_routes(project, [2023, 2024], allow_outcomes=False)
    cuts = load_cuts(project, [2023, 2024], "combine", cfg["tracking"]["sg_window"])
    minimum = int(cfg["routes"]["min_routes"])
    train = _restrict(routes.loc[routes["draft_year"] == 2023], minimum)
    test = _restrict(routes.loc[routes["draft_year"] == 2024], minimum)
    assert_no_cohort_overlap(train["nfl_id"], test["nfl_id"])
    dev_dir = project.interim / "development"
    dev_dir.mkdir(parents=True, exist_ok=True)

    oof_preds = {}
    oof_alphas = {}
    for level in PRIMARY_LEVELS:
        print(f"[develop] out-of-fold {level}", flush=True)
        oof_preds[level], oof_alphas[level] = oof_predictions(train, cuts, level, cfg)
    oof_metrics = _metrics(train, oof_preds)

    chrono_preds = {}
    chrono_alphas = {}
    fitted = {}
    for level in PRIMARY_LEVELS + SENSITIVITY_LEVELS:
        print(f"[develop] 2024 check {level}", flush=True)
        chrono_alphas[level] = choose_alpha(train, cuts, level, cfg)
        predicted, builder, model, demand, _scored = fit_predict_split(train, test, cuts, level, chrono_alphas[level], cfg)
        chrono_preds[level] = predicted
        fitted[level] = (builder, model, demand)
    chrono_metrics = _metrics(test, chrono_preds)

    print("[develop] measurement reliability", flush=True)
    cuts_2023 = cuts.loc[cuts["draft_year"] == 2023]
    _traits, attempts, icc = oof_measurement(cuts_2023, cfg, prefix="")
    trait_vs_forty = _correlation(
        _traits.merge(train.groupby("nfl_id")[["forty", "ten_yd_split"]].first().reset_index(), on="nfl_id", how="left"),
        "trait",
        "forty",
    )
    trait_vs_ten = _correlation(
        _traits.merge(train.groupby("nfl_id")[["ten_yd_split"]].first().reset_index(), on="nfl_id", how="left"),
        "trait",
        "ten_yd_split",
    )

    nfl_cuts = _eligible_nfl_cuts(load_cuts(project, [2023, 2024], "nfl", cfg["tracking"]["sg_window"]), cfg)
    nfl_2023 = nfl_cuts.loc[nfl_cuts["draft_year"] == 2023]
    combine_trait_2024, _ = attach_combine_trait(test, cuts, set(train["nfl_id"].astype(int)), cfg)
    combine_player = combine_trait_2024.groupby("nfl_id")[["trait", "entry_speed", "n_attempts"]].first()
    _nfl_model, nfl_traits, _n0 = fit_trait_bundle(nfl_2023, set(nfl_2023["nfl_id"].astype(int)), cfg, prefix="nfl_")
    # The call above fits on 2023 and scores every row in nfl_2023 only. Score 2024 with that model.
    nfl_2024 = nfl_cuts.loc[nfl_cuts["draft_year"] == 2024]
    nfl_2024_traits = apply_shrinkage(
        trait_table(collapse_attempts(add_residuals(nfl_2024, _nfl_model)), int(cfg["models"]["min_attempts_for_trait"]), prefix="nfl_"),
        _n0,
        prefix="nfl_",
    )
    translation = combine_player.join(nfl_2024_traits.set_index("nfl_id")[["nfl_trait", "nfl_n_attempts"]], how="inner")
    translation_stats = _correlation(translation.reset_index(), "trait", "nfl_trait")

    cells = []
    if not cuts_2023.empty and not nfl_2023.empty:
        _c_ok, _n_ok, cells = support_mask(
            cuts_2023.loc[cuts_2023["drill_role"].eq("primary")],
            nfl_2023,
            cfg,
        )

    print("[develop] permutation and refit bootstrap", flush=True)
    permutation = _permutation(train, test, cuts, chrono_alphas["M1"], chrono_alphas["M2"], cfg)
    refit = _refit_bootstrap(train, test, cuts, chrono_alphas["M1"], chrono_alphas["M2"], cfg)

    window_metrics = {}
    for window in cfg["tracking"]["sg_windows_sensitivity"]:
        alt_cuts = load_cuts(project, [2023, 2024], "combine", int(window))
        predicted, *_ = fit_predict_split(train, test, alt_cuts, "M2", chrono_alphas["M2"], cfg)
        window_metrics[str(window)] = _metrics(test, {"M1": chrono_preds["M1"], "M2": predicted})

    payload = {
        "oof_2023": oof_metrics,
        "oof_alphas": {k: v for k, v in oof_alphas.items()},
        "chrono_2024": chrono_metrics,
        "chrono_alphas": chrono_alphas,
        "reliability_oof_2023": icc,
        "trait_vs_forty": trait_vs_forty,
        "trait_vs_ten": trait_vs_ten,
        "translation_2024": translation_stats,
        "support_cells": cells,
        "permutation": permutation,
        "refit_bootstrap": refit,
        "smoothing": window_metrics,
        "man_coverage": _man_metrics(test, chrono_preds, cfg),
        "min_routes": _threshold_metrics(test, chrono_preds, cfg),
        "n_train_players": int(train["nfl_id"].nunique()),
        "n_test_players": int(test["nfl_id"].nunique()),
        "n_train_routes": int(len(train)),
        "n_test_routes": int(len(test)),
    }
    _write_json(dev_dir / "metrics.json", payload)
    frames = [
        _prediction_frame(train, oof_preds, "oof_2023"),
        _prediction_frame(test, chrono_preds, "chrono_2024"),
    ]
    pd.concat(frames, ignore_index=True).to_parquet(dev_dir / "predictions.parquet", index=False)
    translation.reset_index().to_parquet(dev_dir / "translation_2024.parquet", index=False)
    append_experiment(project, "develop", "chrono_2024", json.dumps(_jsonable({"delta": chrono_metrics.get("delta_m2_minus_m1"), "alphas": chrono_alphas})), "ok", True)
    print(f"[develop] 2023 OOF delta {oof_metrics.get('delta_m2_minus_m1')} | 2024 delta {chrono_metrics.get('delta_m2_minus_m1')}", flush=True)


def _training_bundle(project: Project, routes: pd.DataFrame, cuts: pd.DataFrame):
    cfg = project.config
    minimum = int(cfg["routes"]["min_routes"])
    train = _restrict(routes, minimum)
    alphas = {}
    models = {}
    demands = {}
    for level in PRIMARY_LEVELS + SENSITIVITY_LEVELS:
        print(f"[freeze] {level}", flush=True)
        alphas[level] = choose_alpha(train, cuts, level, cfg)
        _pred, builder, model, demand, _scored = fit_predict_split(train, train.iloc[:1], cuts, level, alphas[level], cfg)
        models[level] = (builder, model)
        if demand is not None:
            demands[level] = demand
    demand = demands.get("M2")
    n0 = float(demand_n0(cuts, set(train["nfl_id"].astype(int)), cfg))
    return train, alphas, models, demand, n0


def demand_n0(cuts: pd.DataFrame, train_ids: set[int], cfg: dict) -> float:
    _model, traits, n0 = fit_trait_bundle(cuts, train_ids, cfg, prefix="")
    return float(traits["shrinkage_n0"].iloc[0]) if len(traits) else float("inf")


def freeze(project: Project) -> None:
    _require_protocol(project)
    cfg = project.config
    routes, _names = load_routes(project, [2023, 2024], allow_outcomes=False)
    window = int(cfg["tracking"]["sg_window"])
    cuts = load_cuts(project, [2023, 2024], "combine", window)
    train, alphas, models, demand, n0 = _training_bundle(project, routes, cuts)
    alt = {}
    for extra in cfg["tracking"]["sg_windows_sensitivity"]:
        alt_cuts = load_cuts(project, [2023, 2024], "combine", int(extra))
        alpha = alphas["M2"]
        _pred, builder, model, alt_demand, _scored = fit_predict_split(train, train.iloc[:1], alt_cuts, "M2", alpha, cfg)
        alt[str(extra)] = {"builder": builder, "model": model, "demand": alt_demand, "n0": demand_n0(alt_cuts, set(train["nfl_id"].astype(int)), cfg)}
    nfl_cuts = _eligible_nfl_cuts(load_cuts(project, [2023, 2024], "nfl", window), cfg)
    nfl_model, _nfl_traits, nfl_n0 = fit_trait_bundle(nfl_cuts, set(train["nfl_id"].astype(int)), cfg, prefix="nfl_")
    _c_mask, _n_mask, cells = support_mask(
        cuts.loc[cuts["drill_role"].eq("primary") & cuts["nfl_id"].isin(set(train["nfl_id"]))],
        nfl_cuts.loc[nfl_cuts["nfl_id"].isin(set(train["nfl_id"]))],
        cfg,
    )
    forty = train.groupby("nfl_id")["forty"].first()
    fingerprints = cached_fingerprints(project)
    manifest = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config_hash(cfg),
        "code_hash": code_hash(),
        "data_version": data_version(fingerprints),
        "alphas": alphas,
        "train_ids": sorted(int(i) for i in train["nfl_id"].unique()),
        "n_train_players": int(train["nfl_id"].nunique()),
        "n_train_routes": int(len(train)),
        "feature_names": {level: models[level][0].columns_ for level in models},
        "demand": None if demand is None else demand.to_dict(),
        "shrinkage_n0": n0,
        "nfl_shrinkage_n0": nfl_n0,
        "support_cells": cells,
        "forty_median": float(forty.median()) if forty.notna().any() else None,
        "forty_band_seconds": float(cfg["models"]["forty_band_seconds"]),
        "primary_window": window,
        "note": "Fit on 2023-24 only. 2025 was not used.",
    }
    frozen = project.interim / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    context_cols = cfg["context"]["categorical"] + cfg["context"]["numeric"]
    pool = train[context_cols].copy()
    for column in LEVEL_NUMERIC["M2_draft"]:
        if column not in pool.columns and column in train.columns:
            pool[column] = train[column].to_numpy()
    weights = equal_player_weights(train["nfl_id"])
    joblib.dump(
        {
            "models": models,
            "demand": demand,
            "n0": n0,
            "alt_windows": alt,
            "nfl_demand": nfl_model,
            "nfl_n0": nfl_n0,
            "context_pool": pool,
            "context_weights": weights,
            "train_routes_index": train.index.to_numpy(),
        },
        frozen / "models.joblib",
    )
    _write_json(frozen / "manifest.json", manifest)
    _write_json(project.reports / "freeze_manifest.json", manifest)
    append_experiment(project, "freeze", "manifest", manifest["config_hash"], "ok", True)
    print(f"[freeze] {manifest['n_train_players']} players, {manifest['n_train_routes']} routes", flush=True)


def _load_frozen(project: Project):
    manifest_path = project.reports / "freeze_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("No freeze manifest. Run freeze before evaluate-heldout.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_hash"] != config_hash(project.config):
        raise RuntimeError("Configuration changed after freeze. Refreeze. Do not treat a refit as the original holdout test.")
    if manifest["code_hash"] != code_hash():
        raise RuntimeError(
            "Code changed after freeze. Refreeze and record the defect. "
            "Do not describe a later refit as an untouched holdout test."
        )
    bundle = joblib.load(project.interim / "frozen" / "models.joblib")
    return manifest, bundle


def _score(cuts: pd.DataFrame, demand, n0: float, cfg: dict, prefix: str = "") -> pd.DataFrame:
    if cuts.empty or demand is None:
        return pd.DataFrame(columns=["nfl_id", f"{prefix}trait"])
    primary = cuts.loc[cuts["drill_role"].eq("primary")].copy()
    if primary.empty:
        return pd.DataFrame(columns=["nfl_id", f"{prefix}trait"])
    attempts = collapse_attempts(add_residuals(primary, demand))
    traits = trait_table(attempts, int(cfg["models"]["min_attempts_for_trait"]), prefix=prefix)
    return apply_shrinkage(traits, n0, prefix=prefix)


def _apply_models(routes: pd.DataFrame, traits: pd.DataFrame, models: dict) -> dict[str, np.ndarray]:
    frame = routes.drop(columns=[c for c in traits.columns if c in routes.columns and c != "nfl_id"], errors="ignore")
    frame = frame.merge(traits, on="nfl_id", how="left")
    predictions = {}
    for level, (builder, model) in models.items():
        predictions[level] = predict_level(builder, model, frame)
    return predictions


_PLAYER_COLUMNS = set(LEVEL_NUMERIC["M2_draft"]) | {"trait", "trait_shrunk", "entry_speed", "age_convention"}


def marginal_projection(builder, model, pool: pd.DataFrame, weights: np.ndarray, player: pd.Series) -> float:
    """Replace only prospect-level fields. The training route mix stays intact."""
    sim = pool.copy()
    for column in _PLAYER_COLUMNS:
        if column not in player.index:
            continue
        value = player[column]
        if pd.isna(value):
            sim[column] = np.nan
        else:
            sim[column] = value
    pred = predict_level(builder, model, sim)
    ok = np.isfinite(pred) & np.isfinite(weights)
    if not ok.any():
        return float("nan")
    return float(np.average(pred[ok], weights=weights[ok]))


def evaluate_heldout(project: Project, cohort: int) -> None:
    if int(cohort) != int(project.config["evaluation_cohort"]):
        raise RuntimeError(f"This study's held-out cohort is {project.config['evaluation_cohort']}, not {cohort}.")
    manifest, bundle = _load_frozen(project)
    print("[evaluate] extracting held-out tracking", flush=True)
    extract(project, [int(cohort)], allow_outcomes=True)
    routes, names = load_routes(project, [int(cohort)], allow_outcomes=True)
    assert_no_cohort_overlap(manifest["train_ids"], routes["nfl_id"].unique())
    cfg = project.config
    window = int(cfg["tracking"]["sg_window"])
    cuts = load_cuts(project, [int(cohort)], "combine", window)
    traits = _score(cuts, bundle["demand"], bundle["n0"], cfg, prefix="")
    predictions = _apply_models(routes, traits, bundle["models"])
    minimum = int(cfg["routes"]["min_routes"])
    primary = _restrict(routes, minimum)
    primary_pred = {level: pred[routes["nfl_id"].isin(set(primary["nfl_id"])).to_numpy()] for level, pred in predictions.items()}
    # The mask above keeps routes whose player qualifies, in original order. primary is that filter, so lengths match.
    if len(primary) != len(next(iter(primary_pred.values()))):
        raise RuntimeError("Held-out prediction alignment failed")
    metrics = _metrics(primary, {k: v for k, v in primary_pred.items() if k in PRIMARY_LEVELS})
    sensitivity = _metrics(primary, primary_pred)
    boot = paired_bootstrap_delta(
        primary["nfl_id"],
        primary["separation_at_pass_forward"],
        primary_pred["M1"],
        primary_pred["M2"],
        n_boot=int(cfg["models"]["bootstrap_predictions"]),
        seed=int(cfg["models"]["random_seed"]),
    )
    influence = leave_one_player_influence(
        primary["nfl_id"], primary["separation_at_pass_forward"], primary_pred["M1"], primary_pred["M2"]
    )
    nfl_cuts = _eligible_nfl_cuts(load_cuts(project, [int(cohort)], "nfl", window), cfg)
    nfl_traits = _score(nfl_cuts, bundle["nfl_demand"], bundle["nfl_n0"], cfg, prefix="nfl_")
    player = traits.merge(nfl_traits, on="nfl_id", how="left")
    translation = _correlation(player, "trait", "nfl_trait")
    # Marginal projections use the training context pool and do not use 2025 routes or outcomes.
    pool = bundle["context_pool"]
    weights = np.asarray(bundle["context_weights"], dtype=float)
    player_rows = primary.groupby("nfl_id").first()
    trait_lookup = traits.set_index("nfl_id")
    marginal = []
    builders = bundle["models"]
    for nfl_id, row in player_rows.iterrows():
        features = row.copy()
        if nfl_id in trait_lookup.index:
            for column in trait_lookup.columns:
                if column != "draft_year":
                    features[column] = trait_lookup.loc[nfl_id, column]
        m1 = marginal_projection(builders["M1"][0], builders["M1"][1], pool, weights, features)
        m2 = marginal_projection(builders["M2"][0], builders["M2"][1], pool, weights, features)
        marginal.append({"nfl_id": int(nfl_id), "marginal_m1": m1, "marginal_m2": m2, "marginal_gap": m2 - m1})
    marginal = pd.DataFrame(marginal)
    band = float(manifest["forty_band_seconds"])
    center = manifest["forty_median"]
    forty = player_rows["forty"]
    candidate_ids = [
        int(i)
        for i in player_rows.index
        if i in set(trait_lookup.index) and pd.notna(trait_lookup.loc[i, "trait"]) and pd.notna(forty.get(i)) and abs(float(forty.get(i)) - center) <= band
    ]
    candidates = marginal.loc[marginal["nfl_id"].isin(candidate_ids)].sort_values("marginal_gap")
    selected = []
    if len(candidates):
        selected = [int(candidates.iloc[0]["nfl_id"]), int(candidates.iloc[-1]["nfl_id"])]
    tmp = primary[["nfl_id"]].copy()
    tmp["pred_m1"] = primary_pred["M1"]
    tmp["pred_m2"] = primary_pred["M2"]
    tmp["y"] = primary["separation_at_pass_forward"].to_numpy()
    player_pred = tmp.groupby("nfl_id").agg(observed=("y", "mean"), pred_m1=("pred_m1", "mean"), pred_m2=("pred_m2", "mean"))
    player_pred["abs_error_m2"] = (player_pred["observed"] - player_pred["pred_m2"]).abs()
    failure_id = int(player_pred["abs_error_m2"].idxmax()) if len(player_pred) else None
    alt_metrics = {}
    for extra, packed in bundle["alt_windows"].items():
        alt_cuts = load_cuts(project, [int(cohort)], "combine", int(extra))
        alt_traits = _score(alt_cuts, packed["demand"], packed["n0"], cfg, prefix="")
        alt_pred = predict_level(packed["builder"], packed["model"], primary.drop(columns=[c for c in alt_traits.columns if c in primary.columns and c != "nfl_id"], errors="ignore").merge(alt_traits, on="nfl_id", how="left"))
        alt_metrics[str(extra)] = _metrics(primary, {"M1": primary_pred["M1"], "M2": alt_pred})
    payload = {
        "cohort": int(cohort),
        "primary": metrics,
        "all_models": sensitivity,
        "paired_bootstrap": boot,
        "translation": translation,
        "man_coverage": _man_metrics(primary, primary_pred, cfg),
        "min_routes": _threshold_metrics(routes, predictions, cfg),
        "smoothing": alt_metrics,
        "selected_nfl_ids": selected,
        "failure_nfl_id": failure_id,
        "failure_label": "post-hoc largest absolute M2 error; not used to choose the model",
        "n_candidates_in_forty_band": int(len(candidates)),
        "n_players": int(primary["nfl_id"].nunique()),
        "n_routes": int(len(primary)),
        "manifest_config_hash": manifest["config_hash"],
    }
    out = project.interim / "heldout"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "metrics.json", payload)
    pred_frame = _prediction_frame(primary, {k: primary_pred[k] for k in PRIMARY_LEVELS}, f"holdout_{cohort}")
    pred_frame.to_parquet(out / "predictions.parquet", index=False)
    influence.to_csv(out / "influence.csv", index=False)
    traits.to_parquet(out / "combine_traits.parquet", index=False)
    nfl_traits.to_parquet(out / "nfl_traits.parquet", index=False)
    marginal.to_parquet(out / "marginal.parquet", index=False)
    player_pred.reset_index().to_parquet(out / "player_predictions.parquet", index=False)
    names.rename("display_name").reset_index().to_parquet(out / "names.parquet", index=False)
    append_experiment(
        project,
        "evaluate-heldout",
        f"cohort_{cohort}",
        json.dumps(_jsonable({"delta": metrics.get("delta_m2_minus_m1"), "bootstrap": boot})),
        "ok",
        True,
    )
    print(f"[evaluate] delta M2-M1 {metrics.get('delta_m2_minus_m1')} bootstrap {boot.get('p2_5')} to {boot.get('p97_5')}", flush=True)


def _claim(rows, claim_id, value, unit, population, denominator, uncertainty, status, source, statement):
    rows.append(
        {
            "claim_id": claim_id,
            "statement": statement,
            "value": value,
            "unit": unit,
            "population": population,
            "denominator": denominator,
            "uncertainty": uncertainty,
            "status": status,
            "source": source,
        }
    )


def report(project: Project) -> None:
    dev_path = project.interim / "development" / "metrics.json"
    hold_path = project.interim / "heldout" / "metrics.json"
    if not hold_path.exists():
        outline = project.reports / "result_summary.md"
        outline.write_text(
            "# Result summary\n\nHeld-out evaluation has not been run. "
            "Numbers below are not available.\n\n- [missing] M0/M1/M2 on the 2025 class\n"
            "- [missing] paired bootstrap\n- [missing] Combine-to-NFL translation\n"
        )
        raise RuntimeError(f"Held-out metrics are missing. An incomplete outline is at {outline}.")
    dev = json.loads(dev_path.read_text()) if dev_path.exists() else {}
    hold = json.loads(hold_path.read_text())
    cfg = project.config
    claims = []
    primary = hold["primary"]
    boot = hold["paired_bootstrap"]
    for level in PRIMARY_LEVELS:
        block = primary.get(level, {})
        _claim(
            claims, f"holdout_{level}_mae", block.get("mae"), "yards",
            "2025 wide receivers with at least the minimum number of eligible routes",
            f"{block.get('n_players')} players, {block.get('n_routes')} routes",
            "player-equal mean absolute error",
            "prespecified",
            "data/interim/heldout/metrics.json",
            f"{level} player-mean absolute error on the 2025 class",
        )
    _claim(
        claims, "holdout_delta_m2_m1", primary.get("delta_m2_minus_m1"), "yards",
        "same 2025 receivers as the primary ladder",
        f"{boot.get('n_players')} players",
        f"paired player bootstrap {boot.get('p2_5')} to {boot.get('p97_5')} ({boot.get('n_boot')} resamples)",
        "prespecified",
        "data/interim/heldout/metrics.json",
        "M2 minus M1 player-mean absolute error. Negative means the cut-exit trait reduced error.",
    )
    trans = hold.get("translation") or {}
    _claim(
        claims, "holdout_translation_spearman", trans.get("spearman"), "rank correlation",
        "2025 receivers with both a Combine trait and an NFL trait",
        f"{trans.get('n')} players",
        "not a confidence interval",
        "prespecified secondary",
        "data/interim/heldout/metrics.json",
        "Rank correlation between the Combine residual and the NFL residual",
    )
    if dev:
        icc = dev.get("reliability_oof_2023") or {}
        _claim(
            claims, "icc_2023_oof", icc.get("icc"), "ICC",
            "2023 receivers with at least two primary-family attempts",
            f"{icc.get('n_players')} players",
            f"player bootstrap {icc.get('icc_p2_5')} to {icc.get('icc_p97_5')}",
            "prespecified",
            "data/interim/development/metrics.json",
            "Out-of-fold reliability of the attempt-level Combine residual",
        )
    ledger = pd.DataFrame(claims)
    ledger.to_csv(project.reports / "claim_ledger.csv", index=False)

    # Figures
    window = int(cfg["tracking"]["sg_window"])
    combine = load_cuts(project, [2023], "combine", window)
    speed_outs = combine.loc[(combine["drill_family"] == "SPEED_OUT") & (combine["drill_role"] == "primary")]
    if len(speed_outs):
        median_exit = speed_outs["exit_distance_0p5"].median()
        score = (speed_outs["abs_angle_deg"] - 75).abs() + 10 * (speed_outs["exit_distance_0p5"] - median_exit).abs()
        chosen = speed_outs.loc[score.idxmin()]
        frames = pd.read_parquet(cohort_dir(project, 2023) / "combine_frames.parquet")
        frames = frames[
            (frames["nfl_id"] == chosen["nfl_id"])
            & (frames["event_id"] == chosen["event_id"])
            & (frames["attempt"] == chosen["attempt"])
        ]
        t = pd.to_datetime(frames["time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9
        plot_trajectory(
            t, frames["x"].to_numpy(), frames["y"].to_numpy(),
            project.figures / "fig1_exit_distance.png",
            "A combine speed-out. Identity withheld. Arrow is half a second along the exit.",
            chosen.to_dict(),
            window=window,
        )
    cells = pd.DataFrame(dev.get("support_cells") or [])
    if len(cells):
        plot_support(cells, project.figures / "fig2_support.png", dev.get("reliability_oof_2023") or {})
    traits = pd.read_parquet(project.interim / "heldout" / "combine_traits.parquet")
    nfl_traits = pd.read_parquet(project.interim / "heldout" / "nfl_traits.parquet")
    both = traits.merge(nfl_traits, on="nfl_id", how="inner")
    if {"trait", "nfl_trait"} <= set(both.columns) and both["nfl_trait"].notna().any():
        plot_translation(
            both, project.figures / "fig3_translation.png",
            f"2025 receivers. Rank correlation {trans.get('spearman')}.",
        )
    bar = pd.DataFrame(
        {
            "model": PRIMARY_LEVELS,
            "mae": [primary[level]["mae"] for level in PRIMARY_LEVELS],
            "title": f"2025 player-mean error. M2−M1 = {primary.get('delta_m2_minus_m1')}",
        }
    )
    plot_model_comparison(bar, project.figures / "fig4_model_error.png")
    names = pd.read_parquet(project.interim / "heldout" / "names.parquet")
    player_pred = pd.read_parquet(project.interim / "heldout" / "player_predictions.parquet")
    marginal = pd.read_parquet(project.interim / "heldout" / "marginal.parquet")
    profile_ids = list(hold.get("selected_nfl_ids") or [])
    if hold.get("failure_nfl_id") and hold["failure_nfl_id"] not in profile_ids:
        profile_ids.append(hold["failure_nfl_id"])
    if profile_ids:
        prof = player_pred.loc[player_pred["nfl_id"].isin(profile_ids)].merge(traits, on="nfl_id", how="left")
        prof = prof.merge(names, on="nfl_id", how="left")
        prof["label"] = prof["display_name"].fillna(prof["nfl_id"].astype(str))
        prof.loc[prof["nfl_id"] == hold.get("failure_nfl_id"), "label"] = prof["label"] + " (largest miss)"
        plot_profile(
            prof.rename(columns={"trait_loao_sd": "trait_loao_sd"})[
                ["label", "trait", "trait_loao_sd", "pred_m1", "pred_m2", "observed"]
            ].fillna({"trait_loao_sd": 0}),
            project.figures / "fig5_profiles.png",
        )
    plot_failures(
        player_pred.rename(columns={"pred_m2": "p", "observed": "y"}),
        project.figures / "fig6_predictions.png",
        "2025 player means. The diagonal is perfect calibration.",
    )
    # Player feature table for the handoff.
    players = pd.read_parquet(project.interim / "wr_players.parquet")
    dev_pred = pd.read_parquet(project.interim / "development" / "predictions.parquet") if (project.interim / "development" / "predictions.parquet").exists() else pd.DataFrame()
    hold_pred = pd.read_parquet(project.interim / "heldout" / "predictions.parquet")
    predictions = pd.concat([dev_pred, hold_pred], ignore_index=True)
    predictions.to_csv(project.outputs / "predictions.csv", index=False)
    feature_rows = []
    for split, frame in predictions.groupby("split"):
        grouped = frame.groupby("nfl_id").agg(n_routes=("y", "size"), mean_separation=("y", "mean"), pred_m0=("pred_M0", "mean"), pred_m1=("pred_M1", "mean"), pred_m2=("pred_M2", "mean"))
        grouped["split"] = split
        feature_rows.append(grouped.reset_index())
    features = pd.concat(feature_rows, ignore_index=True)
    features = features.merge(players, on="nfl_id", how="left")
    if (project.interim / "heldout" / "combine_traits.parquet").exists():
        features = features.merge(traits.add_prefix("holdout_").rename(columns={"holdout_nfl_id": "nfl_id"}), on="nfl_id", how="left")
    features.to_csv(project.outputs / "player_features.csv", index=False)
    metrics = {"development": dev, "holdout": hold}
    _write_json(project.outputs / "metrics.json", metrics)
    delta = primary.get("delta_m2_minus_m1")
    threshold = float(cfg["models"]["effect_threshold_mae_yards"])
    if delta is None:
        conclusion = "The held-out comparison did not return a number."
    elif boot.get("p97_5") is not None and boot["p97_5"] < 0 and abs(delta) >= threshold:
        conclusion = "The cut-exit trait reduced player-level error by at least the analyst-chosen threshold, and the bootstrap interval stayed negative."
    elif boot.get("p2_5") is not None and boot["p2_5"] > 0:
        conclusion = "There is no evidence that the cut-exit trait improved on conventional testing. The bootstrap interval on M2 minus M1 stays positive."
    else:
        conclusion = "The incremental value is inconclusive in this cohort. The point estimate does not clear both the bootstrap interval and the analyst-chosen threshold."
    summary = f"""# Result summary

{conclusion}

Primary ladder on the 2025 class, player-mean absolute error in yards of separation:

- M0 context: {primary.get('M0', {}).get('mae')} ({primary.get('M0', {}).get('n_players')} players, {primary.get('M0', {}).get('n_routes')} routes)
- M1 conventional: {primary.get('M1', {}).get('mae')}
- M2 plus cut-exit residual: {primary.get('M2', {}).get('mae')}
- M2 minus M1: {delta}
- Paired player bootstrap: {boot.get('p2_5')} to {boot.get('p97_5')} over {boot.get('n_boot')} resamples

Analyst-chosen threshold: {threshold} yards. It is not an NFL standard.

Combine-to-NFL rank correlation on 2025: {trans.get('spearman')} (n = {trans.get('n')}).

2023 out-of-fold reliability ICC: {(dev.get('reliability_oof_2023') or {}).get('icc')}.

2024 chronological check, M2 minus M1: {(dev.get('chrono_2024') or {}).get('delta_m2_minus_m1')}.

Profile rule: among 2025 receivers inside {hold.get('n_candidates_in_forty_band')} players whose forty was within {cfg['models']['forty_band_seconds']} seconds of the 2023-24 median, the largest positive and negative gaps between the tracking projection and the conventional projection were pre-registered. The largest absolute M2 miss is reported separately and was chosen after seeing the outcomes.

Selected ids: {hold.get('selected_nfl_ids')}. Largest miss: {hold.get('failure_nfl_id')}.
"""
    (project.reports / "result_summary.md").write_text(summary)
    print("[report] wrote metrics, figures, claim ledger, and result summary", flush=True)


def write_protocol(project: Project) -> None:
    cfg = project.config
    digest = config_hash(cfg)
    cuts = cfg["cuts"]
    text = f"""# Protocol

config_hash: {digest}

This file was written before the outcome models were fit. The 2025 class may be counted for availability. Its separation values, distributions, and errors are not used until `evaluate-heldout`.

## Question

Among wide receivers with comparable conventional Combine measurements, does gaining more ground in the half-second after a turn, after adjusting for the speed and angle of that turn, predict more rookie-season separation?

Direction: a larger cut-exit residual is hypothesized to predict larger separation. The residual is in yards. It is not a defender-separation measure and it is not a force.

## Cohort and outcome

Roster: `nfl_position == WR`. Slot alignments stay in. Tight ends are not added.

Routes: rookie season only (`games.season == players.draft_year`), regular season only, penalty code N, a forward pass (`pass_result` in {cfg['routes']['pass_results']}), route in {cfg['routes']['eligible']}. Screens ({cfg['routes']['excluded']}) and rare labels ({cfg['routes']['rare_drop']}) are out. Missing separation stays missing and is not filled with zero.

The label is present on targeted and untargeted routes when a pass is thrown. The estimand is separation on those routes, not separation conditional on a target.

Primary sample: at least {cfg['routes']['min_routes']} eligible routes. Sensitivities use {cfg['routes']['min_routes_sensitivity']} as evaluation filters, without refitting the ladder to chase them. A man-coverage check requires at least {cfg['models']['man_coverage_min_routes']} man routes.

## Tracking trait

Combine families in the primary trait: {list(cfg['drills']['primary'])}. Names such as `SPEED_OUT_LEFT` and `SPEED_OUT_ROUTE_LEFT` are one family. Turn-back drills ({list(cfg['drills']['secondary_turnback'])}) and {cfg['drills']['excluded']} are not in the primary trait.

A cut needs a stable incoming heading and a stable outgoing heading, a net angle from {cuts['angle_min_deg']} to {cuts['angle_max_deg']} degrees, incoming speed at least {cuts['entry_speed_min']} yards per second, and 0.5 seconds of usable tracking on each side. The reference time is the halfway point of the heading change, not the slowest frame. Exit distance is yards traveled along the outgoing direction over the next 0.5 seconds.

The 4 yard-per-second entry gate was set after the outcome-blind gallery. Slower detections were release arcs and low-speed wiggles. The gallery notes are in `reports/review_sheet.csv`.

On NFL routes the same geometry is used, and the half-second window must end before the pass and before contact or a catch. Routes without snap and pass tags are not measured. A missing NFL trait is not zero.

Expected exit distance is a ridge on incoming speed, absolute angle, their interaction, and a shrunk drill-family offset (penalty {cfg['models']['demand_alpha']}). It is fit on training players only, with each training player weighted equally and each attempt inside a player weighted equally. The player trait is the median residual. Fewer than {cfg['models']['min_attempts_for_trait']} attempts leaves the trait missing. Shrinkage is a sensitivity, not the primary predictor.

## Models

The ladder is ridge regression on routes, with each training player's weights summing to 1.

- M0: route, man/zone, cushion, down, distance, yard line, own-side flag, formation, alignment, motion.
- M1: M0 plus height, weight, age at the Combine tracking date (March 1 of the draft year if that date is missing), forty, ten-yard split, vertical, broad jump, three-cone, and short shuttle. Medians and missingness indicators are fit inside the training fold.
- M2: M1 plus the Combine cut-exit residual.

Player id and name are not features. NFL exit residuals, targets, catches, and yards are not features. NGS grades and draft slot are sensitivities. A forty-and-ten complete case is a sensitivity. Replacing the residual with entry speed is the ablation that checks whether the trait is just "ran into the turn fast."

The penalty is chosen from {cfg['models']['alphas']} by player-grouped error inside the training fold. The primary metric is the mean, across players, of the absolute gap between a player's mean actual separation and mean predicted separation. M2 minus M1 is the comparison. Negative means a smaller error. The analyst-chosen bar for a useful gain is {cfg['models']['effect_threshold_mae_yards']} yards. That number is not an NFL standard.

## Chronology

2023 is the development class, with player-grouped out-of-fold predictions. 2024 is a chronological check. The recipe is then refit on 2023 and 2024 and applied once to 2025. A tree model, if it is run, is secondary and its settings are chosen on 2023 only.

Profiles for the writeup are the 2025 receivers inside {cfg['models']['forty_band_seconds']} seconds of the 2023-24 median forty whose pre-draft tracking projection and conventional projection disagree the most in each direction. A large miss is shown as a miss. It is chosen after the outcomes and is labeled that way.

## What this will not claim

A positive point estimate without a stable measurement and a holdout interval is not a scouting rule. An association is not a training effect. Left-versus-right drill names are not limb asymmetry. There is no defender-tracking separation in these files.
"""
    path = project.reports / "protocol.md"
    path.write_text(text)
    append_experiment(project, "protocol", "freeze_rules", digest, "ok", True)
    print(f"[protocol] wrote {path}", flush=True)


def smoke(project: Project) -> None:
    """A labeled subset run. It is not the reported experiment."""
    routes, _names = load_routes(project, [2023, 2024], allow_outcomes=False)
    cfg = project.config
    cuts = load_cuts(project, [2023, 2024], "combine", cfg["tracking"]["sg_window"])
    ids_2023 = sorted(routes.loc[routes["draft_year"] == 2023, "nfl_id"].unique())[:8]
    ids_2024 = sorted(routes.loc[routes["draft_year"] == 2024, "nfl_id"].unique())[:8]
    train = routes.loc[routes["nfl_id"].isin(ids_2023) & (routes["draft_year"] == 2023)]
    test = routes.loc[routes["nfl_id"].isin(ids_2024) & (routes["draft_year"] == 2024)]
    predicted, *_ = fit_predict_split(train, test, cuts, "M1", 3.0, cfg)
    out = project.outputs / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(
        "SMOKE RUN. Deterministic 8-player subset. Not a reported experiment. "
        "Do not cite these errors.\n"
    )
    pd.DataFrame({"nfl_id": test["nfl_id"].to_numpy(), "pred_m1": predicted}).to_csv(out / "predictions.csv", index=False)
    print("[smoke] wrote outputs/smoke. This is not the study result.", flush=True)
