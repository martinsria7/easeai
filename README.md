# EASEai

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg"></a>
  <a href="./LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-orange.svg">
  <img alt="XGBoost" src="https://img.shields.io/badge/model-XGBoost-5C8A3D.svg">
  <img alt="SHAP" src="https://img.shields.io/badge/explainability-SHAP-7A3E9D.svg">
</p>

<p align="center"><strong>Explainable AI for epidemiology and public health.</strong></p>

`easeai` is a lightweight Python package for **tabular regression explainability** built around a practical research workflow:

- XGBoost hyperparameter tuning
- iterative feature elimination based on XGBoost importance
- cross-validated RMSE and R²
- SHAP feature ranking
- partial dependence plots
- row-level dominant driver extraction

It was extracted from a county-level environmental health workflow, but the package is intentionally general so others can use it on any tabular regression dataset.

## Why use EASEai?

Many research notebooks mix together preprocessing, model tuning, feature selection, evaluation, and explainability in one place. `easeai` turns that into a reusable package for researchers who want:

- a quick XGBoost + SHAP baseline
- interpretable feature ranking
- reproducible artifact export
- a cleaner starting point for GitHub or publication-oriented workflows

## Installation

```bash
pip install -e .
```

Or after publication:

```bash
pip install easeai
```

## Minimal example

```python
import pandas as pd
import easeai as ea


df = pd.read_csv("Alzheimer_merged1.csv", encoding="ISO-8859-1")

workflow = ea.TabularXAIRegressor(
    target="AD_PREV_MEAN",
    drop_columns=["Counties", "FIPS"],
    target_n_features=15,
)

workflow.fit(df)
results = workflow.summarize(id_column="FIPS", name_column="Counties")

print(results.selected_features)
print(results.metrics)
print(results.shap_importance.head())

results.top_drivers[["id", "name", "top_driver"]].to_csv("county_top_drivers.csv", index=False)
workflow.export_artifacts("artifacts")
```

## Package structure

```text
easeai/
  data.py        # preprocessing helpers
  model.py       # tuning, CV, recursive elimination
  explain.py     # SHAP summaries and top-driver extraction
  plotting.py    # SHAP and PDP export helpers
  workflow.py    # end-to-end workflow class
```

## Suggested GitHub topics

`xgboost`, `shap`, `explainable-ai`, `tabular-data`, `epidemiology`, `public-health`, `machine-learning`, `python`

## Roadmap

- add classification support
- add permutation importance
- add bootstrap confidence intervals
- add optional map-ready exports
- publish on PyPI

## Development

```bash
pytest
```

## License

MIT

## Author

Ria A. Martins
