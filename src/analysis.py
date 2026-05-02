from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import extract_binary_target, prepare_ml_frame


@dataclass
class StatisticalValidationResult:
    feature: str
    t_statistic: float
    approximate_p_value: float
    ks_statistic: float
    kl_divergence: float
    wasserstein_distance: float


def numeric_projection(df: pd.DataFrame) -> pd.DataFrame:
    projected = pd.DataFrame(index=df.index)
    if "Age" in df:
        projected["Age"] = df["Age"].map(_range_midpoint).fillna(pd.to_numeric(df["Age"], errors="coerce"))
    if "Income" in df:
        income_numeric = pd.to_numeric(df["Income"], errors="coerce")
        if income_numeric.isna().any():
            income_numeric = df["Income"].astype(str).str.lower().map({"low": 0.0, "high": 1.0})
        projected["Income"] = income_numeric
    for column in ["Gender", "Education", "Occupation"]:
        if column in df:
            projected[f"{column}_Code"] = pd.Categorical(df[column].astype(str)).codes
    return projected.apply(pd.to_numeric, errors="coerce").fillna(0)


def statistical_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    matrix = numeric_projection(df)
    return {
        "mean": matrix.mean(),
        "variance": matrix.var(),
        "correlation": matrix.corr(numeric_only=True).fillna(0),
    }


def machine_learning_evaluation(
    df: pd.DataFrame,
    target: pd.Series | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    y = extract_binary_target(df) if target is None else target.astype(int).reset_index(drop=True)
    x = prepare_ml_frame(df, include_income_feature=False).reset_index(drop=True)

    if y.nunique() < 2:
        return pd.DataFrame(
            [
                {"model": "Logistic Regression", "accuracy": 0.0, "rmse": 1.0, "precision": 0.0, "recall": 0.0},
                {"model": "Decision Tree", "accuracy": 0.0, "rmse": 1.0, "precision": 0.0, "recall": 0.0},
                {"model": "Random Forest", "accuracy": 0.0, "rmse": 1.0, "precision": 0.0, "recall": 0.0},
            ]
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "rmse": math.sqrt(mean_squared_error(y_test, predictions)),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def statistical_validation(original: pd.DataFrame, masked: pd.DataFrame) -> pd.DataFrame:
    original_matrix = numeric_projection(original)
    masked_matrix = numeric_projection(masked)
    rows: list[StatisticalValidationResult] = []

    for column in sorted(set(original_matrix.columns).intersection(masked_matrix.columns)):
        a = original_matrix[column].to_numpy(dtype=float)
        b = masked_matrix[column].to_numpy(dtype=float)
        rows.append(
            StatisticalValidationResult(
                feature=column,
                t_statistic=_welch_t_statistic(a, b),
                approximate_p_value=_normal_approx_p_value(_welch_t_statistic(a, b)),
                ks_statistic=_ks_statistic(a, b),
                kl_divergence=_kl_divergence(a, b),
                wasserstein_distance=_wasserstein_distance(a, b),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def _range_midpoint(value: object) -> float:
    text = str(value)
    if "-" not in text:
        return float("nan")
    try:
        left, right = text.split("-", 1)
        return (float(left) + float(right)) / 2
    except ValueError:
        return float("nan")


def _welch_t_statistic(a: np.ndarray, b: np.ndarray) -> float:
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    denom = math.sqrt(var_a / len(a) + var_b / len(b))
    if denom == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / denom)


def _normal_approx_p_value(t_statistic: float) -> float:
    return float(math.erfc(abs(t_statistic) / math.sqrt(2)))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([a, b])))
    if len(values) == 0:
        return 0.0
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _kl_divergence(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    low = min(float(np.min(a)), float(np.min(b)))
    high = max(float(np.max(a)), float(np.max(b)))
    if low == high:
        return 0.0
    p, edges = np.histogram(a, bins=bins, range=(low, high), density=True)
    q, _ = np.histogram(b, bins=edges, density=True)
    p = p + 1e-9
    q = q + 1e-9
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except Exception:
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        n = min(len(a_sorted), len(b_sorted))
        if n == 0:
            return 0.0
        quantiles = np.linspace(0, 1, n)
        a_q = np.quantile(a_sorted, quantiles)
        b_q = np.quantile(b_sorted, quantiles)
        return float(np.mean(np.abs(a_q - b_q)))
