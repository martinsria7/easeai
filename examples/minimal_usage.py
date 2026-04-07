import pandas as pd

from easeai import TabularXAIRegressor

# Replace with your own dataset
# df = pd.read_csv("your_data.csv", encoding="ISO-8859-1")

# Example expected shape:
# df must contain a numeric target column and any identifier columns you want to drop.

def main():
    df = pd.read_csv("Alzheimer_merged1.csv", encoding="ISO-8859-1")
    workflow = TabularXAIRegressor(
        target="AD_PREV_MEAN",
        drop_columns=["Counties", "FIPS"],
        target_n_features=15,
    )
    workflow.fit(df)
    results = workflow.summarize(id_column="FIPS", name_column="Counties")

    print("Selected features:")
    print(results.selected_features)
    print("\nCV metrics:")
    print(results.metrics)
    print("\nTop SHAP features:")
    print(results.shap_importance.head())

    results.top_drivers[["id", "name", "top_driver"]].to_csv("county_top_drivers.csv", index=False)
    workflow.export_artifacts("artifacts")


if __name__ == "__main__":
    main()
