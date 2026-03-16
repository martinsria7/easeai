import pandas as pd
from sklearn.datasets import make_regression

from easeai import TabularXAIRegressor, preprocess_frame, rfe_xgb
from xgboost import XGBRegressor


def test_preprocess_frame_basic():
    df = pd.DataFrame({
        "target": [1, 2, 3],
        "A weird col": [1, None, 3],
        "id": [10, 11, 12],
    })
    X, y = preprocess_frame(df, target="target", drop_columns=["id"])
    assert list(y) == [1, 2, 3]
    assert "id" not in X.columns
    assert "A_weird_col" in X.columns
    assert X.isna().sum().sum() == 0


def test_rfe_xgb_returns_requested_feature_count():
    X_arr, y = make_regression(n_samples=80, n_features=12, random_state=42)
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(12)])
    y = pd.Series(y)
    model = XGBRegressor(objective="reg:squarederror", n_estimators=10, random_state=42)
    selected = rfe_xgb(model, X, y, target_n=5)
    assert len(selected) == 5


def test_workflow_runs_end_to_end():
    X_arr, y = make_regression(n_samples=120, n_features=10, n_informative=5, random_state=42)
    df = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(10)])
    df["target"] = y
    df["FIPS"] = range(len(df))
    workflow = TabularXAIRegressor(target="target", drop_columns=["FIPS"], target_n_features=5, n_iter_search=2)
    workflow.fit(df)
    results = workflow.summarize(id_column="FIPS")
    assert len(results.selected_features) == 5
    assert "rmse_mean" in results.metrics
    assert not results.shap_importance.empty
    assert "top_driver" in results.top_drivers.columns
