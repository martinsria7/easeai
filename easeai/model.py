from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor


DEFAULT_PARAM_DIST: Dict[str, Iterable] = {
    "n_estimators": [300, 500, 700, 900],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.5, 0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma": [0, 1, 2, 3, 5],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [1, 3, 5, 10],
}


def make_cv(n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> KFold:
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def tune_xgb_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    param_distributions: Optional[Dict[str, Iterable]] = None,
    n_iter: int = 30,
    cv: Optional[KFold] = None,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
    n_jobs: int = -1,
    tree_method: str = "hist",
    verbose: int = 0,
) -> Tuple[XGBRegressor, RandomizedSearchCV]:
    """Tune an XGBRegressor using randomized search."""
    cv = cv or make_cv(random_state=random_state)
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions or DEFAULT_PARAM_DIST,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search


def rfe_xgb(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    target_n: int = 15,
    drop_frac: float = 0.10,
) -> List[str]:
    """Iteratively drop the least important features until target_n remain."""
    if target_n < 1:
        raise ValueError("target_n must be >= 1")
    if not 0 < drop_frac < 1:
        raise ValueError("drop_frac must be between 0 and 1")
    if target_n > X.shape[1]:
        raise ValueError("target_n cannot exceed number of features")

    features = list(X.columns)
    while len(features) > target_n:
        model.fit(X[features], y)
        imp = pd.Series(model.feature_importances_, index=features).sort_values()
        drop_n = max(1, int(len(features) * drop_frac))
        drop_feats = imp.index[:drop_n].tolist()
        features = [f for f in features if f not in drop_feats]
    return features


def evaluate_regression_cv(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[KFold] = None,
    n_jobs: int = -1,
) -> Dict[str, float]:
    """Return mean and SD for RMSE and R^2 across CV folds."""
    cv = cv or make_cv()
    rmse_scores = -cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
    )
    r2_scores = cross_val_score(
        model,
        X,
        y,
        scoring="r2",
        cv=cv,
        n_jobs=n_jobs,
    )
    return {
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_sd": float(rmse_scores.std()),
        "r2_mean": float(r2_scores.mean()),
        "r2_sd": float(r2_scores.std()),
    }
