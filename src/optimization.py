from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .analysis import machine_learning_evaluation, statistical_validation
from .evaluation import combined_objective, privacy_metrics, utility_metrics
from .masking import MaskingConfig, MaskingEngine


class OptimizationEngine:
    """Grid-search masking parameters to balance privacy and utility."""

    def __init__(
        self,
        masking_engine: MaskingEngine,
        quasi_identifiers: list[str],
        privacy_weight: float = 0.55,
        random_state: int = 42,
    ) -> None:
        self.masking_engine = masking_engine
        self.quasi_identifiers = quasi_identifiers
        self.privacy_weight = privacy_weight
        self.random_state = random_state

    def default_grid(self) -> list[MaskingConfig]:
        grid: list[MaskingConfig] = []
        for noise in [0.5, 1.5, 3.0, 6.0, 10.0]:
            grid.append(MaskingConfig(method="noise", noise_std=noise, gender_mask_probability=min(noise / 20, 0.5)))
        for bin_width in [5, 10, 15, 20]:
            grid.append(MaskingConfig(method="generalization", generalization_bin=bin_width))
        for epsilon in [0.25, 0.5, 1.0, 2.0, 5.0]:
            grid.append(MaskingConfig(method="differential_privacy", epsilon=epsilon))
        for noise in [1.5, 3.0, 6.0]:
            for bin_width in [10, 20]:
                for epsilon in [0.5, 1.0, 2.0]:
                    grid.append(
                        MaskingConfig(
                            method="hybrid",
                            noise_std=noise,
                            generalization_bin=bin_width,
                            epsilon=epsilon,
                            gender_mask_probability=min(noise / 20, 0.5),
                        )
                    )
        return grid

    def randomized_search(self, n_iter: int = 28) -> list[MaskingConfig]:
        rng = np.random.default_rng(self.random_state)
        methods = np.array(["noise", "generalization", "differential_privacy", "hybrid"])
        configs: list[MaskingConfig] = []

        for _ in range(n_iter):
            method = str(rng.choice(methods, p=[0.22, 0.22, 0.28, 0.28]))
            noise = float(rng.choice([0.25, 0.5, 1.0, 1.5, 3.0, 6.0, 10.0]))
            bin_width = int(rng.choice([5, 10, 15, 20, 25]))
            epsilon = float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]))
            gender_probability = float(min(noise / 20, 0.5))
            dp_fraction = float(rng.choice([0.5, 0.6, 0.7, 0.8]))

            configs.append(
                MaskingConfig(
                    method=method,
                    noise_std=noise,
                    generalization_bin=bin_width,
                    epsilon=epsilon,
                    gender_mask_probability=gender_probability,
                    hybrid_dp_budget_fraction=dp_fraction,
                )
            )

        anchor_configs = [
            MaskingConfig(method="noise", noise_std=1.5, gender_mask_probability=0.075),
            MaskingConfig(method="generalization", generalization_bin=10),
            MaskingConfig(method="differential_privacy", epsilon=1.0),
            MaskingConfig(
                method="hybrid",
                noise_std=3.0,
                generalization_bin=15,
                epsilon=2.0,
                gender_mask_probability=0.15,
                hybrid_dp_budget_fraction=0.7,
            ),
        ]
        return self._deduplicate(anchor_configs + configs)

    def run(
        self,
        original: pd.DataFrame,
        baseline_ml: pd.DataFrame,
        target: pd.Series,
        configs: list[MaskingConfig] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, MaskingConfig]:
        rows = []
        best_score = -1.0
        best_masked = original.copy()
        best_config = MaskingConfig(method="noise", noise_std=0.0)

        for config in configs or self.randomized_search():
            masked = self.masking_engine.apply(original, config)
            masked_ml = machine_learning_evaluation(masked, target=target, random_state=self.random_state)
            privacy = privacy_metrics(original, masked, self.quasi_identifiers)
            utility = utility_metrics(original, masked, baseline_ml, masked_ml)
            score = combined_objective(
                privacy["privacy_score"],
                utility["utility_score"],
                privacy_weight=self.privacy_weight,
            )
            validation = statistical_validation(original, masked)
            mean_kl = float(validation["kl_divergence"].mean()) if not validation.empty else 0.0
            mean_ks = float(validation["ks_statistic"].mean()) if not validation.empty else 0.0
            mean_wasserstein = float(validation["wasserstein_distance"].mean()) if not validation.empty else 0.0

            row = {
                **asdict(config),
                "config_label": config.label(),
                **privacy,
                **utility,
                "objective_score": score,
                "mean_ks_statistic": mean_ks,
                "mean_kl_divergence": mean_kl,
                "mean_wasserstein_distance": mean_wasserstein,
            }
            rows.append(row)

            if score > best_score:
                best_score = score
                best_masked = masked
                best_config = config

        results = pd.DataFrame(rows).sort_values("objective_score", ascending=False).reset_index(drop=True)
        return results, best_masked, best_config

    def _deduplicate(self, configs: list[MaskingConfig]) -> list[MaskingConfig]:
        seen: set[str] = set()
        unique: list[MaskingConfig] = []
        for config in configs:
            label = config.label()
            if label not in seen:
                unique.append(config)
                seen.add(label)
        return unique
