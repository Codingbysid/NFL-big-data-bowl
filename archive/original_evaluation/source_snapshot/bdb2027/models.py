"""Player-equal ridge models. Player id and name are never features.

Learned transforms (imputation, category collapse, scaling, the demand
surface, and the ridge penalty) are fit on the training players of the
current fold only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from bdb2027.evaluation import equal_player_weights, summarize_frames
from bdb2027.features import fit_trait_bundle

# Columns that must never enter a prospect model.
FORBIDDEN_FEATURES = {
    "nfl_id",
    "display_name",
    "separation_at_pass_forward",
    "nfl_trait",
    "nfl_entry_speed",
    "expected_yards_after_catch",
    "yards_gained",
    "rec_yards",
    "rush_yards",
    "yards_after_catch",
    "expected_points_added",
    "pass_result",
    "target",
    "pass_length",
    "dropback_duration",
}

CONVENTIONAL = [
    "height",
    "weight",
    "age",
    "forty",
    "ten_yd_split",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
]
NGS = ["ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]

LEVEL_NUMERIC = {
    "M0": [],
    "M1": CONVENTIONAL,
    "M2": CONVENTIONAL + ["trait"],
    "M2_speed": CONVENTIONAL + ["entry_speed"],
    "M2_shrunk": CONVENTIONAL + ["trait_shrunk"],
    "M1_forty": ["forty", "ten_yd_split"],
    "M1_ngs": CONVENTIONAL + NGS,
    "M2_ngs": CONVENTIONAL + ["trait"] + NGS,
    "M2_draft": CONVENTIONAL + ["trait", "undrafted", "draft_overall_pick"],
}

PRIMARY_LEVELS = ["M0", "M1", "M2"]


def _wmean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    if not mask.any():
        return 0.0
    return float(np.sum(weights[mask] * values[mask]) / np.sum(weights[mask]))


def _wstd(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    if mask.sum() < 2:
        return 1.0
    var = float(np.sum(weights[mask] * (values[mask] - mean) ** 2) / np.sum(weights[mask]))
    return float(np.sqrt(max(var, 0.0)))


@dataclass
class DesignBuilder:
    level: str
    cfg: dict
    impute_: dict = field(default_factory=dict)
    cat_levels_: dict = field(default_factory=dict)
    cat_reference_: dict = field(default_factory=dict)
    scale_mean_: np.ndarray | None = None
    scale_std_: np.ndarray | None = None
    columns_: list[str] = field(default_factory=list)
    keep_idx_: list[int] = field(default_factory=list)
    numeric_names_: list[str] = field(default_factory=list)

    def _numeric_names(self) -> list[str]:
        extras = LEVEL_NUMERIC[self.level]
        blocked = set(extras) & FORBIDDEN_FEATURES
        if blocked:
            raise ValueError(f"Level {self.level} requests forbidden features: {blocked}")
        return list(self.cfg["context"]["numeric"]) + list(extras)

    def _prepare(self, frame: pd.DataFrame, fitting: bool) -> tuple[np.ndarray, list[str]]:
        work = frame.copy()
        num_cols = self._numeric_names() if fitting else [c for c in self.numeric_names_ if not c.endswith("_missing")]
        if fitting:
            self.numeric_names_ = []
        names: list[str] = []
        columns: list[np.ndarray] = []
        source_cols = num_cols
        indicators = []
        for col in source_cols:
            if col not in work.columns:
                work[col] = np.nan
            values = pd.to_numeric(work[col], errors="coerce").astype("float64")
            if fitting:
                per_player = work.assign(_v=values).groupby("nfl_id")["_v"].median()
                self.impute_[col] = float(per_player.median()) if per_player.notna().any() else 0.0
            if col != "undrafted":
                missing = values.isna().astype(float).to_numpy()
                indicators.append((f"{col}_missing", missing))
            filled = values.fillna(self.impute_.get(col, 0.0)).to_numpy(dtype=float)
            columns.append(filled)
            names.append(col)
        for name, missing in indicators:
            columns.append(missing)
            names.append(name)
        if fitting:
            self.numeric_names_ = list(names)

        cat_names: list[str] = []
        cat_cols: list[np.ndarray] = []
        for col in self.cfg["context"]["categorical"]:
            series = work[col].astype("string").fillna("MISSING") if col in work.columns else pd.Series("MISSING", index=work.index)
            series = series.replace({"<NA>": "MISSING", "nan": "MISSING", "": "MISSING"})
            if fitting:
                counts = series.value_counts()
                minimum = int(self.cfg["routes"]["min_category_plays"])
                keep = [v for v in sorted(counts.index.astype(str)) if int(counts[v]) >= minimum and v != "MISSING"]
                levels = keep + ["MISSING", "OTHER"]
                self.cat_levels_[col] = levels
                self.cat_reference_[col] = levels[0]
            levels = self.cat_levels_[col]
            mapped = series.where(series.isin(levels), "OTHER")
            for lev in levels:
                if lev == self.cat_reference_[col]:
                    continue
                cat_cols.append((mapped == lev).astype(float).to_numpy())
                cat_names.append(f"{col}[{lev}]")
        matrix = np.column_stack(columns + cat_cols) if (columns or cat_cols) else np.zeros((len(work), 0))
        all_names = names + cat_names
        return matrix, all_names

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray) -> "DesignBuilder":
        matrix, names = self._prepare(frame, fitting=True)
        means = np.zeros(matrix.shape[1])
        stds = np.ones(matrix.shape[1])
        n_cont = len([c for c in self.numeric_names_ if not str(c).endswith("_missing")])
        # numeric_names_ includes indicators, which were appended after the raw columns.
        # Scale only the raw numeric columns, which occupy the leading block.
        n_raw = len(self._numeric_names())
        for i in range(n_raw):
            means[i] = _wmean(matrix[:, i], sample_weight)
            stds[i] = _wstd(matrix[:, i], sample_weight, means[i])
            if stds[i] < 1e-8:
                stds[i] = 1.0
        self.scale_mean_ = means
        self.scale_std_ = stds
        scaled = (matrix - means) / stds
        keep = [j for j in range(scaled.shape[1]) if np.nanstd(scaled[:, j]) > 1e-8]
        self.keep_idx_ = keep
        self.columns_ = [names[j] for j in keep]
        leaked = set(self.columns_) & FORBIDDEN_FEATURES
        if leaked:
            raise ValueError(f"Design matrix contains forbidden columns: {sorted(leaked)}")
        if any(name.startswith("nfl_") for name in self.columns_):
            raise ValueError(f"NFL movement leaked into the prospect design: {self.columns_}")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix, _ = self._prepare(frame, fitting=False)
        if matrix.shape[1] != len(self.scale_mean_):
            raise ValueError(
                f"Design width {matrix.shape[1]} does not match the fitted width {len(self.scale_mean_)}"
            )
        scaled = (matrix - self.scale_mean_) / self.scale_std_
        return scaled[:, self.keep_idx_]


def _complete_case_mask(frame: pd.DataFrame, level: str) -> np.ndarray:
    if level != "M1_forty":
        return np.ones(len(frame), dtype=bool)
    return frame["forty"].notna().to_numpy() & frame["ten_yd_split"].notna().to_numpy()


def fit_level(train: pd.DataFrame, level: str, alpha: float, cfg: dict):
    mask = _complete_case_mask(train, level)
    used = train.loc[mask].copy()
    if used["nfl_id"].nunique() < 5:
        raise ValueError(f"{level} has only {used['nfl_id'].nunique()} training players after filters")
    y = used["separation_at_pass_forward"].to_numpy(dtype=float)
    keep = np.isfinite(y)
    used = used.loc[keep]
    y = y[keep]
    if np.isnan(y).any():
        raise ValueError("Missing separation entered the training target")
    weights = equal_player_weights(used["nfl_id"])
    # Re-check the player totals.
    check = pd.DataFrame({"nfl_id": used["nfl_id"].to_numpy(), "w": weights})
    totals = check.groupby("nfl_id")["w"].sum()
    if not np.allclose(totals.to_numpy(), 1.0):
        raise ValueError("Sample weights are not equal across players")
    builder = DesignBuilder(level=level, cfg=cfg)
    builder.fit(used, weights)
    model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
    model.fit(builder.transform(used), y, sample_weight=weights)
    return builder, model


def predict_level(builder: DesignBuilder, model: Ridge, frame: pd.DataFrame) -> np.ndarray:
    mask = _complete_case_mask(frame, builder.level)
    pred = np.full(len(frame), np.nan)
    if mask.any():
        pred[np.flatnonzero(mask)] = model.predict(builder.transform(frame.loc[mask]))
    return pred


def attach_combine_trait(routes: pd.DataFrame, cuts: pd.DataFrame, train_ids: set[int], cfg: dict) -> pd.DataFrame:
    """Score the Combine trait from ``train_ids`` only and join it onto routes.

    NFL-movement columns are not joined here.
    """
    model, traits, _n0 = fit_trait_bundle(cuts, train_ids, cfg, prefix="")
    keep = [
        "nfl_id",
        "trait",
        "trait_shrunk",
        "trait_unfiltered",
        "trait_loao_sd",
        "entry_speed",
        "n_attempts",
        "families",
    ]
    slim = traits[keep].copy()
    drop = [c for c in keep if c in routes.columns and c != "nfl_id"]
    base = routes.drop(columns=drop)
    out = base.merge(slim, on="nfl_id", how="left")
    if "nfl_trait" in out.columns:
        raise ValueError("NFL trait was attached to a prospect modeling frame")
    return out, model


def choose_alpha(routes: pd.DataFrame, cuts: pd.DataFrame, level: str, cfg: dict) -> float:
    """Pick the ridge penalty by player-grouped MAE. Traits are refit inside each fold."""
    alphas = [float(a) for a in cfg["models"]["alphas"]]
    n_groups = int(routes["nfl_id"].nunique())
    n_splits = min(int(cfg["models"]["inner_folds"]), n_groups)
    if n_splits < 2:
        return alphas[len(alphas) // 2]
    routes = routes.reset_index(drop=True)
    splitter = GroupKFold(n_splits=n_splits)
    scores = {a: [] for a in alphas}
    for train_idx, test_idx in splitter.split(routes, groups=routes["nfl_id"]):
        train = routes.iloc[train_idx]
        test = routes.iloc[test_idx]
        train_ids = set(train["nfl_id"].astype(int))
        if level in {"M0", "M1", "M1_forty", "M1_ngs"}:
            scored_train, scored_test = train, test
        else:
            both, _ = attach_combine_trait(pd.concat([train, test], ignore_index=True), cuts, train_ids, cfg)
            # concat preserves order of train then test.
            scored_train = both.iloc[: len(train)]
            scored_test = both.iloc[len(train) :]
        for alpha in alphas:
            try:
                builder, model = fit_level(scored_train, level, alpha, cfg)
            except ValueError:
                scores[alpha].append(np.inf)
                continue
            pred = predict_level(builder, model, scored_test)
            y = scored_test["separation_at_pass_forward"].to_numpy(dtype=float)
            ok = np.isfinite(pred) & np.isfinite(y)
            if ok.sum() == 0:
                scores[alpha].append(np.inf)
                continue
            idx = np.flatnonzero(ok)
            scores[alpha].append(summarize_frames(scored_test.iloc[idx]["nfl_id"], y[idx], pred[idx])["mae"])
    means = {a: float(np.mean(v)) for a, v in scores.items()}
    return min(means, key=means.get)


def fit_predict_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cuts: pd.DataFrame,
    level: str,
    alpha: float,
    cfg: dict,
    recompute_trait: bool = True,
):
    train_ids = set(train["nfl_id"].astype(int))
    no_trait = level in {"M0", "M1", "M1_forty", "M1_ngs"}
    if no_trait or not recompute_trait:
        scored_train, scored_test, demand = train, test, None
    else:
        scored_train, demand = attach_combine_trait(train, cuts, train_ids, cfg)
        scored_test, _ = attach_combine_trait(test, cuts, train_ids, cfg)
    builder, model = fit_level(scored_train, level, alpha, cfg)
    pred = predict_level(builder, model, scored_test)
    return pred, builder, model, demand, scored_test
