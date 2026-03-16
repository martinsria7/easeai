from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values



def shap_importance_table(shap_values, X: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )



def county_top_drivers(
    shap_values,
    X: pd.DataFrame,
    id_col: Optional[pd.Series] = None,
    name_col: Optional[pd.Series] = None,
) -> pd.DataFrame:
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    if id_col is not None:
        shap_df["id"] = list(id_col)
    if name_col is not None:
        shap_df["name"] = list(name_col)
    shap_df["top_driver"] = shap_df[X.columns].abs().idxmax(axis=1)
    return shap_df
