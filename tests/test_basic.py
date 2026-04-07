import pandas as pd
import numpy as np
import easeai as ea


def test_basic_fit():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x1": rng.normal(size=100),
        "x2": rng.normal(size=100),
        "FIPS": range(100),
        "Counties": [f"c{i}" for i in range(100)],
    })
    df["y"] = 2 * df["x1"] - 0.5 * df["x2"] + rng.normal(scale=0.1, size=100)

    workflow = ea.TabularXAIRegressor(
        target="y",
        drop_columns=["FIPS", "Counties"],
        target_n_features=2,
    )

    workflow.fit(df)
    results = workflow.summarize(id_column="FIPS", name_column="Counties")

    assert len(results.selected_features) > 0
    assert results.metrics is not None
