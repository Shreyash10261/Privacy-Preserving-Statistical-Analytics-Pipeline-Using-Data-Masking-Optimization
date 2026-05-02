from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_COLUMNS = ["Age", "Gender", "Income", "Education", "Occupation"]
SENSITIVE_ATTRIBUTES = ["Age", "Gender", "Income"]


class DataIngestion:
    """Load an Adult Income-style dataset with a deterministic synthetic fallback."""

    def __init__(self, data_dir: str | Path = "data", random_state: int = 42) -> None:
        self.data_dir = Path(data_dir)
        self.random_state = random_state

    def load(self, sample_size: int = 5000) -> pd.DataFrame:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        local = self._load_local_dataset()
        if local is not None:
            return self._finalize(local, sample_size)

        openml = self._load_openml_adult()
        if openml is not None:
            return self._finalize(openml, sample_size)

        synthetic = self._make_synthetic_adult(sample_size=max(sample_size, 2500))
        out = self.data_dir / "synthetic_adult_income.csv"
        synthetic.to_csv(out, index=False)
        return self._finalize(synthetic, sample_size)

    def _load_local_dataset(self) -> pd.DataFrame | None:
        for filename in ("adult.csv", "adult_income.csv", "synthetic_adult_income.csv"):
            path = self.data_dir / filename
            if path.exists():
                return pd.read_csv(path)
        return None

    def _load_openml_adult(self) -> pd.DataFrame | None:
        try:
            from sklearn.datasets import fetch_openml

            bunch = fetch_openml("adult", version=2, as_frame=True, parser="auto")
            df = bunch.frame.copy()
            if "class" not in df.columns and getattr(bunch, "target", None) is not None:
                df["class"] = bunch.target
            normalized = self._normalize_adult_schema(df)
            normalized.to_csv(self.data_dir / "adult_openml_cached.csv", index=False)
            return normalized
        except Exception:
            return None

    def _normalize_adult_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        lookup = {str(col).strip().lower().replace("-", "_"): col for col in df.columns}

        def pick(candidates: Iterable[str]) -> pd.Series:
            for name in candidates:
                key = name.lower().replace("-", "_")
                if key in lookup:
                    return df[lookup[key]]
            raise KeyError(f"Missing one of: {list(candidates)}")

        age = pd.to_numeric(pick(["Age", "age"]), errors="coerce")
        gender = pick(["Gender", "Sex", "sex"]).astype(str).str.strip()
        education = pick(["Education", "education"]).astype(str).str.strip()
        occupation = pick(["Occupation", "occupation"]).astype(str).str.strip()

        income_raw = pick(["Income", "class", "income", "target"]).astype(str).str.strip()
        income = income_raw.str.contains(">50K|1|high", case=False, regex=True).astype(int)

        return pd.DataFrame(
            {
                "Age": age,
                "Gender": gender,
                "Income": income,
                "Education": education,
                "Occupation": occupation,
            }
        )

    def _make_synthetic_adult(self, sample_size: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        educations = np.array(
            ["HS-grad", "Some-college", "Bachelors", "Masters", "Assoc", "Doctorate"]
        )
        occupations = np.array(
            [
                "Adm-clerical",
                "Craft-repair",
                "Exec-managerial",
                "Prof-specialty",
                "Sales",
                "Tech-support",
                "Other-service",
            ]
        )
        genders = np.array(["Female", "Male"])

        age = np.clip(rng.normal(39, 13, sample_size).round(), 18, 75).astype(int)
        gender = rng.choice(genders, sample_size, p=[0.48, 0.52])
        education = rng.choice(educations, sample_size, p=[0.33, 0.22, 0.2, 0.1, 0.1, 0.05])
        occupation = rng.choice(
            occupations,
            sample_size,
            p=[0.16, 0.16, 0.14, 0.18, 0.16, 0.08, 0.12],
        )

        education_boost = pd.Series(education).map(
            {
                "HS-grad": -0.7,
                "Some-college": -0.25,
                "Assoc": 0.05,
                "Bachelors": 0.65,
                "Masters": 1.0,
                "Doctorate": 1.35,
            }
        )
        occupation_boost = pd.Series(occupation).map(
            {
                "Exec-managerial": 0.9,
                "Prof-specialty": 0.75,
                "Tech-support": 0.35,
                "Sales": 0.05,
                "Craft-repair": -0.1,
                "Adm-clerical": -0.35,
                "Other-service": -0.85,
            }
        )
        gender_effect = np.where(gender == "Male", 0.12, -0.08)
        age_effect = (age - 38) / 24
        logits = -1.1 + age_effect + education_boost.to_numpy() + occupation_boost.to_numpy() + gender_effect
        probabilities = 1 / (1 + np.exp(-logits))
        income = rng.binomial(1, probabilities)

        return pd.DataFrame(
            {
                "Age": age,
                "Gender": gender,
                "Income": income,
                "Education": education,
                "Occupation": occupation,
            }
        )

    def _finalize(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        if set(PROJECT_COLUMNS).issubset(df.columns):
            normalized = df[PROJECT_COLUMNS].copy()
        else:
            normalized = self._normalize_adult_schema(df)

        normalized = normalized.replace({"?": np.nan, "nan": np.nan, "None": np.nan})
        normalized["Age"] = pd.to_numeric(normalized["Age"], errors="coerce")
        normalized["Income"] = pd.to_numeric(normalized["Income"], errors="coerce")
        normalized = normalized.dropna(subset=PROJECT_COLUMNS)
        normalized["Age"] = normalized["Age"].clip(18, 90).round().astype(int)
        normalized["Income"] = (normalized["Income"] >= 0.5).astype(int)
        normalized["Gender"] = normalized["Gender"].astype(str).str.strip()
        normalized["Education"] = normalized["Education"].astype(str).str.strip()
        normalized["Occupation"] = normalized["Occupation"].astype(str).str.strip()

        if len(normalized) > sample_size:
            normalized = normalized.sample(sample_size, random_state=self.random_state)
        return normalized.reset_index(drop=True)


class SensitiveAttributeIdentifier:
    """Simple rule-based sensitive attribute marker."""

    def __init__(self, sensitive_attributes: list[str] | None = None) -> None:
        self.sensitive_attributes = sensitive_attributes or SENSITIVE_ATTRIBUTES

    def identify(self, df: pd.DataFrame) -> list[str]:
        return [column for column in self.sensitive_attributes if column in df.columns]


def prepare_ml_frame(df: pd.DataFrame, include_income_feature: bool = False) -> pd.DataFrame:
    features = df.copy()
    if not include_income_feature and "Income" in features.columns:
        features = features.drop(columns=["Income"])
    return pd.get_dummies(features, dummy_na=False).fillna(0)


def extract_binary_target(df: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(df["Income"], errors="coerce").fillna(0)
    return (target >= 0.5).astype(int)
