from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def preprocess_frame(
    df: pd.DataFrame,
    target: str,
    drop_columns: Optional[Iterable[str]] = None,
    fill_strategy: str = "median",
    coerce_numeric: bool = True,
    clean_column_names: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare a tabular dataframe for XGBoost regression.

    Parameters
    ----------
    df:
        Input dataframe.
    target:
        Name of target column.
    drop_columns:
        Extra identifier or metadata columns to remove from the feature matrix.
    fill_strategy:
        One of {"median", "mean", "zero", "drop"}.
    coerce_numeric:
        Convert feature columns to numeric, coercing non-numeric values to NaN.
    clean_column_names:
        Replace non-alphanumeric characters with underscores.

    Returns
    -------
    X, y
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    drop_columns = list(drop_columns or [])
    y = pd.to_numeric(df[target], errors="raise")

    feature_drop = [target] + [c for c in drop_columns if c in df.columns]
    X = df.drop(columns=feature_drop).copy()

    if coerce_numeric:
        X = X.apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    if fill_strategy == "median":
        X = X.fillna(X.median(numeric_only=True))
    elif fill_strategy == "mean":
        X = X.fillna(X.mean(numeric_only=True))
    elif fill_strategy == "zero":
        X = X.fillna(0)
    elif fill_strategy == "drop":
        valid_rows = X.notna().all(axis=1)
        X = X.loc[valid_rows].copy()
        y = y.loc[valid_rows].copy()
    else:
        raise ValueError("fill_strategy must be one of: median, mean, zero, drop")

    if clean_column_names:
        X.columns = (
            X.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
            .str.strip("_")
        )

    return X, y
