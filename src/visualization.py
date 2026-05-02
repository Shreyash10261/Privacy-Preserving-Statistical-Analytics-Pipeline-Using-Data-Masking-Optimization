from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .analysis import numeric_projection


class VisualizationModule:
    def __init__(self, output_dir: str | Path = "outputs/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def correlation_heatmap(self, df: pd.DataFrame, name: str) -> Path:
        corr = numeric_projection(df).corr().fillna(0)
        path = self.output_dir / f"{name}_correlation_heatmap.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
        plt.title(f"{name.title()} Correlation Matrix")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def accuracy_comparison(self, baseline_ml: pd.DataFrame, masked_ml: pd.DataFrame) -> Path:
        path = self.output_dir / "accuracy_comparison.png"
        baseline = baseline_ml.assign(dataset="Original")
        masked = masked_ml.assign(dataset="Best Masked")
        plot_df = pd.concat([baseline, masked], ignore_index=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=plot_df, x="model", y="accuracy", hue="dataset")
        plt.ylim(0, 1)
        plt.title("Machine Learning Accuracy Comparison")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def privacy_utility_tradeoff(self, results: pd.DataFrame) -> Path:
        path = self.output_dir / "privacy_utility_tradeoff.png"
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=results,
            x="utility_score",
            y="privacy_score",
            hue="method",
            size="objective_score",
            sizes=(40, 220),
            alpha=0.8,
        )
        best = results.iloc[0]
        plt.scatter([best["utility_score"]], [best["privacy_score"]], color="black", s=100, marker="X", label="Best")
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        plt.title("Privacy vs Utility Trade-off")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def metric_summary(self, results: pd.DataFrame) -> Path:
        path = self.output_dir / "top_config_scores.png"
        top = results.head(8).copy()
        top["rank"] = [f"#{idx + 1}" for idx in range(len(top))]
        plot_df = top.melt(
            id_vars=["rank", "method"],
            value_vars=["privacy_score", "utility_score", "objective_score"],
            var_name="metric",
            value_name="score",
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=plot_df, x="rank", y="score", hue="metric")
        plt.ylim(0, 1)
        plt.title("Top Masking Configurations")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

