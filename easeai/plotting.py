from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay
from PIL import Image



def save_shap_summary(X: pd.DataFrame, shap_values, outpath: str, plot_type: Optional[str] = None, dpi: int = 300):
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()



def save_pdp_plots(model, X: pd.DataFrame, features: Iterable[str], outdir: str, grid_resolution: int = 60):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for feature in features:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[feature],
            kind="average",
            grid_resolution=grid_resolution,
            percentiles=(0.01, 0.99),
            ax=ax,
        )
        ax.set_title(f"PDP: {feature}")
        outpath = outdir / f"pdp_{str(feature)[:40].replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        saved.append(str(outpath))
    return saved



def save_dependence_panel(X: pd.DataFrame, shap_values, top_vars: Iterable[str], outpath: str):
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    axes = axes.ravel()
    for i, var in enumerate(top_vars):
        plt.sca(axes[i])
        shap.dependence_plot(
            ind=var,
            shap_values=shap_values,
            features=X,
            interaction_index="auto",
            show=False,
            ax=axes[i],
        )
        axes[i].set_title(str(var))
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()



def combine_images_side_by_side(left_path: str, right_path: str, outpath: str):
    left = Image.open(left_path)
    right = Image.open(right_path)
    target_h = max(left.height, right.height)
    left = left.resize((int(left.width * target_h / left.height), target_h), Image.Resampling.LANCZOS)
    right = right.resize((int(right.width * target_h / right.height), target_h), Image.Resampling.LANCZOS)
    combined = Image.new("RGB", (left.width + right.width, target_h), (255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    combined.save(outpath)
