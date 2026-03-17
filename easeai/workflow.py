from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from xgboost import XGBRegressor

from .data import preprocess_frame
from .explain import compute_shap_values, county_top_drivers, shap_importance_table
from .model import evaluate_regression_cv, make_cv, rfe_xgb, tune_xgb_regressor
from .plotting import save_pdp_plots, save_shap_summary


@dataclass
class WorkflowResults:
    selected_features: List[str]
    metrics: dict
    shap_importance: pd.DataFrame
    top_drivers: pd.DataFrame


class TabularXAIRegressor:
    """End-to-end workflow for tabular regression with XGBoost + SHAP."""

    def __init__(
        self,
        target: str,
        drop_columns: Optional[Iterable[str]] = None,
        target_n_features: int = 15,
        random_state: int = 42,
        n_iter_search: int = 15,
    ):
        self.target = target
        self.drop_columns = list(drop_columns or [])
        self.target_n_features = target_n_features
        self.random_state = random_state
        self.n_iter_search = n_iter_search
        self.search_ = None
        self.best_model_: Optional[XGBRegressor] = None
        self.final_model_: Optional[XGBRegressor] = None
        self.selected_features_: Optional[List[str]] = None
        self.explainer_ = None
        self.shap_values_ = None
        self.X_selected_: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "TabularXAIRegressor":
        X, y = preprocess_frame(
            df,
            target=self.target,
            drop_columns=self.drop_columns,
        )
        cv = make_cv(random_state=self.random_state)
        self.best_model_, self.search_ = tune_xgb_regressor(
            X,
            y,
            cv=cv,
            random_state=self.random_state,
            n_iter=self.n_iter_search,
        )
        self.selected_features_ = rfe_xgb(
            self.best_model_,
            X,
            y,
            target_n=self.target_n_features,
        )
        self.final_model_ = XGBRegressor(
            **self.search_.best_params_,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        self.X_selected_ = X[self.selected_features_].copy()
        self.final_model_.fit(self.X_selected_, y)
        self.explainer_, self.shap_values_ = compute_shap_values(self.final_model_, self.X_selected_)
        self.y_ = y
        self.original_df_ = df.copy()
        return self

    def summarize(self, id_column: Optional[str] = None, name_column: Optional[str] = None) -> WorkflowResults:
        if self.final_model_ is None or self.X_selected_ is None or self.selected_features_ is None:
            raise RuntimeError("Call fit() before summarize().")
        metrics = evaluate_regression_cv(self.final_model_, self.X_selected_, self.y_)
        shap_rank = shap_importance_table(self.shap_values_, self.X_selected_)
        top_drivers = county_top_drivers(
            self.shap_values_,
            self.X_selected_,
            id_col=self.original_df_[id_column] if id_column and id_column in self.original_df_.columns else None,
            name_col=self.original_df_[name_column] if name_column and name_column in self.original_df_.columns else None,
        )
        return WorkflowResults(
            selected_features=self.selected_features_,
            metrics=metrics,
            shap_importance=shap_rank,
            top_drivers=top_drivers,
        )

    def export_artifacts(self, outdir: str, top_k_pdp: int = 6) -> None:
        if self.X_selected_ is None:
            raise RuntimeError("Call fit() before export_artifacts().")
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_shap_summary(self.X_selected_, self.shap_values_, str(outdir / "shap_beeswarm.png"))
        save_shap_summary(self.X_selected_, self.shap_values_, str(outdir / "shap_bar.png"), plot_type="bar")
        top_vars = list(self.X_selected_.columns[:top_k_pdp])
        save_pdp_plots(self.final_model_, self.X_selected_, top_vars, str(outdir / "pdp_plots"))
