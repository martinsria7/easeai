"""easeai: reusable utilities for XGBoost-based tabular regression, feature selection,
and explainability with SHAP and partial dependence plots.
"""

from .data import preprocess_frame
from .model import tune_xgb_regressor, rfe_xgb, evaluate_regression_cv
from .workflow import TabularXAIRegressor, WorkflowResults
from .explain import shap_importance_table, county_top_drivers

__all__ = [
    "preprocess_frame",
    "tune_xgb_regressor",
    "rfe_xgb",
    "evaluate_regression_cv",
    "TabularXAIRegressor",
    "WorkflowResults",
    "shap_importance_table",
    "county_top_drivers",
]

__version__ = "0.1.0"
