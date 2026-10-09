"""Demand penalty on an explicit weighted objective.

The fitted objective is

    sum(w * residual^2) / sum(w) + lambda * ||beta||^2

scikit-learn's Ridge minimizes sum(w * residual^2) + alpha * ||beta||^2
when sample weights are applied by rescaling rows. Those statements match
when alpha = lambda * sum(w). Equal player and attempt weights are unchanged.
Lambda is chosen from Combine exit prediction only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold

from bdb2027.features import DemandModel, player_attempt_cut_weights

PENALTY_GRID = (0.0, 0.01, 0.1, 1.0, 3.0)


def sklearn_alpha(lambda_penalty: float, weight_sum: float) -> float:
    """Convert the normalized penalty into Ridge's alpha."""
    if weight_sum <= 0:
        raise ValueError("Weight sum must be positive")
    return float(lambda_penalty) * float(weight_sum)


class WeightedDemandModel(DemandModel):
    """DemandModel whose alpha argument is lambda, not the raw Ridge alpha."""

    lambda_: float = 0.0

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "WeightedDemandModel":
        if frame.empty or frame["nfl_id"].nunique() < 2:
            raise ValueError("Demand model needs cuts from at least two training players")
        present = sorted(frame["drill_family"].astype(str).unique())
        if self.reference_family not in present:
            self.reference_family = present[0]
        self.families_ = present
        x = self._design(frame, fitting=True)
        y = frame["exit_distance_0p5"].to_numpy(dtype=float)
        weights = np.ones(len(frame)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.lambda_ = float(self.alpha)
        if self.lambda_ <= 0:
            model = LinearRegression(fit_intercept=True)
            model.fit(x, y, sample_weight=weights)
            self.alpha = 0.0
        else:
            model = Ridge(alpha=sklearn_alpha(self.lambda_, float(weights.sum())), fit_intercept=True, random_state=0)
            model.fit(x, y, sample_weight=weights)
            self.alpha = float(self.lambda_)
        self.coef_ = np.asarray(model.coef_, dtype=float)
        self.intercept_ = float(model.intercept_)
        return self


def weighted_mse(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    resid = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.sum(weights * resid**2) / np.sum(weights))


def select_lambda(
    frame: pd.DataFrame,
    lambdas: tuple[float, ...] = PENALTY_GRID,
    n_splits: int = 5,
) -> dict:
    """Choose lambda by player-grouped Combine exit error. Separation is not an input."""
    players = frame["nfl_id"].to_numpy()
    n_players = frame["nfl_id"].nunique()
    splits = min(int(n_splits), int(n_players))
    if splits < 2:
        raise ValueError("Penalty selection needs at least two player groups")
    splitter = GroupKFold(n_splits=splits)
    rows = []
    oof = {lam: np.full(len(frame), np.nan) for lam in lambdas}
    y = frame["exit_distance_0p5"].to_numpy(dtype=float)
    for train_idx, test_idx in splitter.split(frame, groups=players):
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        w_test = player_attempt_cut_weights(test)
        for lam in lambdas:
            model = WeightedDemandModel(alpha=float(lam), reference_family="SPEED_OUT")
            model.fit(train, player_attempt_cut_weights(train))
            pred = model.predict(test)
            oof[lam][test_idx] = pred
            rows.append({"lambda": float(lam), "wmse": weighted_mse(y[test_idx], pred, w_test)})
    summary = []
    for lam in lambdas:
        part = [row["wmse"] for row in rows if row["lambda"] == lam]
        summary.append({"lambda": float(lam), "mean_wmse": float(np.mean(part)), "folds": part})
    chosen = min(summary, key=lambda row: row["mean_wmse"])
    return {"grid": summary, "chosen_lambda": chosen["lambda"], "oof_prediction": oof[chosen["lambda"]]}
