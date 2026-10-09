"""Demand adjustment, the player cut-exit trait, reliability, and common support.

The demand model is fit on training players only. A new prospect's residual
uses that frozen surface. Nothing here reads separation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ATTEMPT_KEYS = ["nfl_id", "attempt_uid"]


def attempt_uid_combine(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["nfl_id"].astype(str)
        + "|"
        + frame["event_id"].astype(str)
        + "|"
        + frame["attempt"].astype(str)
        + "|"
        + frame["drill_name"].astype(str)
    )


def attempt_uid_nfl(frame: pd.DataFrame) -> pd.Series:
    return frame["nfl_id"].astype(str) + "|" + frame["game_id"].astype(str) + "|" + frame["play_id"].astype(str)


def player_attempt_cut_weights(frame: pd.DataFrame) -> np.ndarray:
    """Each training player sums to 1, then each attempt, then each cut."""
    if frame.empty:
        return np.array([], dtype=float)
    n_players = frame["nfl_id"].nunique()
    attempts = frame.groupby("nfl_id")["attempt_uid"].transform("nunique")
    cuts = frame.groupby(ATTEMPT_KEYS)["attempt_uid"].transform("size")
    weights = 1.0 / (n_players * attempts.to_numpy(dtype=float) * cuts.to_numpy(dtype=float))
    return weights


@dataclass
class DemandModel:
    """Expected exit yards from incoming speed, absolute angle, and drill family.

    Continuous terms are standardized on the training cuts. Family offsets are
    ridge-shrunk toward the reference family. The intercept is not penalized.
    """

    alpha: float = 3.0
    reference_family: str = "SPEED_OUT"
    families_: list[str] = field(default_factory=list)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    feature_names_: list[str] = field(default_factory=list)

    def _design(self, frame: pd.DataFrame, fitting: bool) -> np.ndarray:
        speed = frame["speed_in"].to_numpy(dtype=float)
        angle = frame["abs_angle_deg"].to_numpy(dtype=float)
        raw = np.column_stack([speed, angle])
        if fitting:
            self.mean_ = np.nanmean(raw, axis=0)
            self.std_ = np.nanstd(raw, axis=0)
            self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        z_speed = (speed - self.mean_[0]) / self.std_[0]
        z_angle = (angle - self.mean_[1]) / self.std_[1]
        cols = [z_speed, z_angle, z_speed * z_angle]
        names = ["z_speed", "z_angle", "z_speed_x_z_angle"]
        family = frame["drill_family"].astype(str)
        for name in self.families_:
            if name == self.reference_family:
                continue
            cols.append((family == name).to_numpy(dtype=float))
            names.append(f"family_{name}")
        if fitting:
            self.feature_names_ = names
        return np.column_stack(cols)

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "DemandModel":
        if frame.empty or frame["nfl_id"].nunique() < 2:
            raise ValueError("Demand model needs cuts from at least two training players")
        present = sorted(frame["drill_family"].astype(str).unique())
        if self.reference_family not in present:
            self.reference_family = present[0]
        self.families_ = present
        x = self._design(frame, fitting=True)
        y = frame["exit_distance_0p5"].to_numpy(dtype=float)
        model = Ridge(alpha=self.alpha, fit_intercept=True, random_state=0)
        model.fit(x, y, sample_weight=sample_weight)
        self.coef_ = model.coef_.astype(float)
        self.intercept_ = float(model.intercept_)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Demand model is not fit")
        x = self._design(frame, fitting=False)
        return self.intercept_ + x @ self.coef_

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "reference_family": self.reference_family,
            "families": self.families_,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "coef": None if self.coef_ is None else self.coef_.tolist(),
            "intercept": self.intercept_,
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DemandModel":
        model = cls(alpha=payload["alpha"], reference_family=payload["reference_family"])
        model.families_ = list(payload["families"])
        model.mean_ = np.asarray(payload["mean"], dtype=float)
        model.std_ = np.asarray(payload["std"], dtype=float)
        model.coef_ = np.asarray(payload["coef"], dtype=float)
        model.intercept_ = float(payload["intercept"])
        model.feature_names_ = list(payload["feature_names"])
        return model


def add_residuals(frame: pd.DataFrame, model: DemandModel) -> pd.DataFrame:
    out = frame.copy()
    out["expected_exit"] = model.predict(out)
    out["residual"] = out["exit_distance_0p5"] - out["expected_exit"]
    return out


def collapse_attempts(cuts: pd.DataFrame) -> pd.DataFrame:
    """One row per attempt. Cuts inside an attempt are averaged, not stacked."""
    if cuts.empty:
        return cuts.copy()
    return _collapse(cuts)


def _collapse(cuts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (nfl_id, uid), g in cuts.groupby(ATTEMPT_KEYS, sort=False):
        row = {
            "nfl_id": int(nfl_id),
            "attempt_uid": uid,
            "residual": float(g["residual"].mean()),
            "speed_in": float(g["speed_in"].mean()),
            "abs_angle_deg": float(g["abs_angle_deg"].mean()),
            "exit_distance_0p5": float(g["exit_distance_0p5"].mean()),
            "expected_exit": float(g["expected_exit"].mean()),
            "drill_family": g["drill_family"].iloc[0],
            "draft_year": int(g["draft_year"].iloc[0]),
            "n_cuts": int(len(g)),
        }
        if "route_ran" in g.columns:
            row["route_ran"] = g["route_ran"].iloc[0]
        if "in_support" in g.columns:
            row["in_support"] = bool(g["in_support"].all())
        rows.append(row)
    return pd.DataFrame(rows)


def trait_table(attempts: pd.DataFrame, min_attempts: int, prefix: str = "") -> pd.DataFrame:
    """Player median residual. Estimates with fewer than min_attempts stay missing."""
    rows = []
    value_col = "residual"
    for nfl_id, g in attempts.groupby("nfl_id"):
        values = g[value_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = int(len(values))
        median = float(np.median(values)) if n else np.nan
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n >= 2 else np.nan
        if n >= 2:
            loao = [float(np.median(np.delete(values, i))) for i in range(n)]
            loao_sd = float(np.std(loao, ddof=1)) if n > 2 else float(np.std(loao, ddof=0))
        else:
            loao_sd = np.nan
        entry = float(np.median(g["speed_in"])) if n else np.nan
        families = sorted(g["drill_family"].astype(str).unique()) if n else []
        rows.append(
            {
                "nfl_id": int(nfl_id),
                "draft_year": int(g["draft_year"].iloc[0]),
                f"{prefix}n_attempts": n,
                f"{prefix}trait_unfiltered": median,
                f"{prefix}trait_mean": mean,
                f"{prefix}trait_sd": sd,
                f"{prefix}trait_loao_sd": loao_sd,
                f"{prefix}entry_speed": entry,
                f"{prefix}families": "|".join(families),
                f"{prefix}trait": median if n >= min_attempts else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["nfl_id", "draft_year", f"{prefix}trait", f"{prefix}n_attempts"])
    return pd.DataFrame(rows)


def shrinkage_n0(attempts: pd.DataFrame) -> float:
    """Method-of-moments n0 = within variance / between variance, in attempt units."""
    groups = []
    for _, g in attempts.groupby("nfl_id"):
        values = g["residual"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) >= 2:
            groups.append(values)
    if len(groups) < 3:
        return float("inf")
    within = np.mean([np.var(g, ddof=1) for g in groups])
    means = np.array([np.mean(g) for g in groups])
    between = float(np.var(means, ddof=1))
    if between <= 1e-8:
        return float("inf")
    return float(within / between)


def apply_shrinkage(traits: pd.DataFrame, n0: float, prefix: str = "") -> pd.DataFrame:
    out = traits.copy()
    n = out[f"{prefix}n_attempts"].to_numpy(dtype=float)
    raw = out[f"{prefix}trait_unfiltered"].to_numpy(dtype=float)
    if not np.isfinite(n0):
        shrunk = np.zeros_like(raw)
    else:
        shrunk = raw * (n / (n + n0))
    out[f"{prefix}trait_shrunk_raw"] = shrunk
    out[f"{prefix}trait_shrunk"] = np.where(n >= 1, shrunk, np.nan)
    # Keep the primary missingness rule: too few attempts stay missing.
    missing = out[f"{prefix}trait"].isna()
    out.loc[missing, f"{prefix}trait_shrunk"] = np.nan
    out[f"{prefix}shrinkage_n0"] = n0
    return out


def reliability_icc(attempts: pd.DataFrame) -> dict:
    """One-way ICC of attempt residuals for players with a repeated attempt."""
    groups = []
    for _, g in attempts.groupby("nfl_id"):
        values = g["residual"].dropna().to_numpy(dtype=float)
        if len(values) >= 2:
            groups.append(values)
    if len(groups) < 3:
        return {"n_players": len(groups), "icc": np.nan, "within_sd": np.nan, "between_sd": np.nan}
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([np.mean(g) for g in groups])
    grand = np.average(means, weights=ns)
    ssw = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    dfw = float(ns.sum() - len(groups))
    msw = ssw / dfw
    ssb = float(np.sum(ns * (means - grand) ** 2))
    msb = ssb / (len(groups) - 1)
    n0 = (ns.sum() - np.sum(ns**2) / ns.sum()) / (len(groups) - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) != 0 else np.nan
    return {
        "n_players": int(len(groups)),
        "n_attempts": int(ns.sum()),
        "icc": float(icc),
        "within_sd": float(np.sqrt(max(msw, 0.0))),
        "between_sd": float(np.std(means, ddof=1)),
        "n0_harmonic": float(n0),
    }


def bootstrap_icc(attempts: pd.DataFrame, n_boot: int = 1000, seed: int = 2027) -> dict:
    point = reliability_icc(attempts)
    players = attempts["nfl_id"].unique()
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        take = rng.choice(players, size=len(players), replace=True)
        parts = [attempts.loc[attempts["nfl_id"] == i] for i in take]
        # Relabel copies so a duplicated player is not pooled with himself.
        relabeled = []
        for k, part in enumerate(parts):
            copy = part.copy()
            copy["nfl_id"] = k
            relabeled.append(copy)
        stats.append(reliability_icc(pd.concat(relabeled, ignore_index=True))["icc"])
    arr = np.array(stats, dtype=float)
    arr = arr[np.isfinite(arr)]
    point["icc_p2_5"] = float(np.percentile(arr, 2.5)) if len(arr) else np.nan
    point["icc_p97_5"] = float(np.percentile(arr, 97.5)) if len(arr) else np.nan
    point["n_boot"] = int(n_boot)
    return point


def support_mask(combine: pd.DataFrame, nfl: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Cells of incoming speed by absolute angle that both samples actually occupy."""
    speed_edges = cfg["support"]["speed_edges"]
    angle_edges = cfg["support"]["angle_edges"]
    min_c = int(cfg["support"]["min_cell_combine"])
    min_n = int(cfg["support"]["min_cell_nfl"])

    def _bin(frame):
        speed = pd.cut(frame["speed_in"], speed_edges, include_lowest=True)
        angle = pd.cut(frame["abs_angle_deg"], angle_edges, include_lowest=True)
        return speed, angle

    cs, ca = _bin(combine)
    ns, na = _bin(nfl)
    c_counts = pd.crosstab(cs, ca)
    n_counts = pd.crosstab(ns, na)
    cells = []
    keep = set()
    for s_lab in c_counts.index.union(n_counts.index):
        for a_lab in c_counts.columns.union(n_counts.columns):
            c_n = int(c_counts.loc[s_lab, a_lab]) if s_lab in c_counts.index and a_lab in c_counts.columns else 0
            n_n = int(n_counts.loc[s_lab, a_lab]) if s_lab in n_counts.index and a_lab in n_counts.columns else 0
            ok = c_n >= min_c and n_n >= min_n
            cells.append(
                {
                    "speed_bin": str(s_lab),
                    "angle_bin": str(a_lab),
                    "n_combine": c_n,
                    "n_nfl": n_n,
                    "supported": bool(ok),
                }
            )
            if ok:
                keep.add((str(s_lab), str(a_lab)))
    c_ok = pd.Series([(str(s), str(a)) in keep for s, a in zip(cs, ca)], index=combine.index)
    n_ok = pd.Series([(str(s), str(a)) in keep for s, a in zip(ns, na)], index=nfl.index)
    return c_ok, n_ok, cells


def fit_trait_bundle(cuts: pd.DataFrame, train_ids: set[int], cfg: dict, prefix: str = "") -> tuple[DemandModel, pd.DataFrame, float]:
    """Fit the demand surface on training players and score every cut in ``cuts``."""
    primary = cuts.loc[cuts["drill_role"].eq("primary")].copy()
    train = primary.loc[primary["nfl_id"].isin(train_ids)]
    model = DemandModel(alpha=float(cfg["models"]["demand_alpha"]))
    weights = player_attempt_cut_weights(train)
    model.fit(train, weights)
    scored = add_residuals(primary, model)
    attempts = _collapse(scored)
    n0 = shrinkage_n0(attempts.loc[attempts["nfl_id"].isin(train_ids)])
    traits = apply_shrinkage(trait_table(attempts, int(cfg["models"]["min_attempts_for_trait"]), prefix=prefix), n0, prefix=prefix)
    return model, traits, n0
