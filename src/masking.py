from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MaskingConfig:
    method: str
    noise_std: float = 0.0
    generalization_bin: int = 10
    epsilon: float = 1.0
    gender_mask_probability: float = 0.0
    hybrid_dp_budget_fraction: float = 0.7

    def label(self) -> str:
        return (
            f"{self.method}|noise={self.noise_std}|bin={self.generalization_bin}|"
            f"eps={self.epsilon}|gender_p={self.gender_mask_probability}|"
            f"dp_fraction={self.hybrid_dp_budget_fraction}"
        )


class MaskingEngine:
    """Configurable data masking strategies for sensitive attributes."""

    def __init__(self, sensitive_attributes: list[str], random_state: int = 42) -> None:
        self.sensitive_attributes = sensitive_attributes
        self.random_state = random_state

    def apply(self, df: pd.DataFrame, config: MaskingConfig) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed_for(config))
        masked = df.copy(deep=True)

        if config.method == "noise":
            return self.noise_addition(masked, config.noise_std, config.gender_mask_probability, rng)
        if config.method == "generalization":
            return self.generalization(masked, config.generalization_bin, config.gender_mask_probability, rng)
        if config.method == "differential_privacy":
            return self.differential_privacy(masked, config.epsilon, rng)
        if config.method == "hybrid":
            masked = self.noise_addition(masked, config.noise_std, 0.0, rng)
            gender_before_generalization = masked["Gender"].copy() if "Gender" in masked else None
            masked = self.generalization(masked, config.generalization_bin, 0.0, rng)
            if gender_before_generalization is not None:
                masked["Gender"] = gender_before_generalization
            # Non-DP masking is followed by a final DP release step. The epsilon
            # reported for the hybrid method is split conservatively so the final
            # quantifiable DP stage has an explicit budget.
            dp_epsilon = max(config.epsilon * config.hybrid_dp_budget_fraction, 1e-6)
            masked = self.differential_privacy(masked, dp_epsilon, rng)
            return masked

        raise ValueError(f"Unknown masking method: {config.method}")

    def noise_addition(
        self,
        df: pd.DataFrame,
        noise_std: float,
        gender_mask_probability: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        if "Age" in self.sensitive_attributes and "Age" in df:
            df["Age"] = (pd.to_numeric(df["Age"], errors="coerce") + rng.normal(0, noise_std, len(df))).clip(18, 90)
        if "Income" in self.sensitive_attributes and "Income" in df:
            income_noise = rng.normal(0, noise_std / 10, len(df))
            df["Income"] = (pd.to_numeric(df["Income"], errors="coerce") + income_noise).clip(0, 1)
        if "Gender" in self.sensitive_attributes and "Gender" in df and gender_mask_probability > 0:
            df["Gender"] = self._mask_categorical(df["Gender"], gender_mask_probability, rng, replacement="Suppressed")
        return df

    def generalization(
        self,
        df: pd.DataFrame,
        bin_width: int,
        gender_mask_probability: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        bin_width = max(int(bin_width), 1)
        if "Age" in self.sensitive_attributes and "Age" in df:
            ages = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].mode().iloc[0])
            starts = (ages // bin_width * bin_width).astype(int)
            ends = starts + bin_width - 1
            df["Age"] = starts.astype(str) + "-" + ends.astype(str)
        if "Income" in self.sensitive_attributes and "Income" in df:
            income = pd.to_numeric(df["Income"], errors="coerce").fillna(0)
            df["Income"] = np.where(income >= 0.5, "High", "Low")
        if "Gender" in self.sensitive_attributes and "Gender" in df:
            probability = max(gender_mask_probability, min(0.5, bin_width / 100))
            df["Gender"] = self._mask_categorical(df["Gender"], probability, rng, replacement="Any")
        return df

    def differential_privacy(
        self,
        df: pd.DataFrame,
        epsilon: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        epsilon = max(float(epsilon), 1e-6)
        if "Age" in self.sensitive_attributes and "Age" in df:
            scale = self._global_sensitivity("Age") / epsilon
            age = self._numeric_age(df["Age"])
            df["Age"] = (age + rng.laplace(0, scale, len(df))).clip(18, 90)
        if "Income" in self.sensitive_attributes and "Income" in df:
            scale = self._global_sensitivity("Income") / epsilon
            income = self._numeric_income(df["Income"])
            df["Income"] = (income + rng.laplace(0, scale, len(df))).clip(0, 1)
        if "Gender" in self.sensitive_attributes and "Gender" in df:
            df["Gender"] = self._randomized_response_binary(df["Gender"], epsilon, rng)
        return df

    def _mask_categorical(
        self,
        values: pd.Series,
        probability: float,
        rng: np.random.Generator,
        replacement: str,
    ) -> pd.Series:
        mask = rng.random(len(values)) < probability
        output = values.astype(str).copy()
        output.loc[mask] = replacement
        return output

    def _numeric_age(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            midpoint = values.astype(str).map(self._range_midpoint)
            numeric = numeric.fillna(midpoint)
        return numeric.fillna(numeric.median()).fillna(39)

    def _numeric_income(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            labels = values.astype(str).str.lower().map({"low": 0.0, "high": 1.0})
            numeric = numeric.fillna(labels)
        return numeric.fillna(numeric.median()).fillna(0)

    def _global_sensitivity(self, attribute: str) -> float:
        bounds = {
            "Age": (18.0, 90.0),
            "Income": (0.0, 1.0),
        }
        low, high = bounds[attribute]
        return high - low

    def _randomized_response_binary(
        self,
        values: pd.Series,
        epsilon: float,
        rng: np.random.Generator,
    ) -> pd.Series:
        epsilon = max(float(epsilon), 1e-6)
        keep_probability = np.exp(epsilon) / (1 + np.exp(epsilon))
        output = values.astype(str).copy()
        original = output.copy()
        categories = sorted(original.dropna().unique().tolist())
        fallback_categories = ["Female", "Male"]
        binary_categories = categories if len(categories) == 2 else fallback_categories
        flip = rng.random(len(output)) > keep_probability

        if len(binary_categories) == 2:
            left, right = binary_categories
            left_mask = flip & (original == left)
            right_mask = flip & (original == right)
            unknown_mask = flip & ~original.isin(binary_categories)
            output.loc[left_mask] = right
            output.loc[right_mask] = left
            output.loc[unknown_mask] = rng.choice(binary_categories, size=unknown_mask.sum())
        else:
            output.loc[flip] = rng.choice(binary_categories, size=flip.sum())
        return output

    def _range_midpoint(self, value: str) -> float:
        if "-" not in value:
            return float("nan")
        try:
            left, right = value.split("-", 1)
            return (float(left) + float(right)) / 2
        except ValueError:
            return float("nan")

    def _seed_for(self, config: MaskingConfig) -> int:
        return abs(hash((self.random_state, config.label()))) % (2**32)
