from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import numeric_projection


def privacy_metrics(original: pd.DataFrame, masked: pd.DataFrame, quasi_identifiers: list[str]) -> dict[str, float]:
    qis = [column for column in quasi_identifiers if column in masked.columns]
    grouped = masked[qis].astype(str).groupby(qis, dropna=False).size()
    per_record_group_size = masked[qis].astype(str).merge(
        grouped.rename("group_size").reset_index(),
        on=qis,
        how="left",
    )["group_size"]

    k_anonymity = float(grouped.min()) if len(grouped) else 0.0
    average_equivalence_class_size = float(grouped.mean()) if len(grouped) else 0.0
    reidentification_risk = float((1 / per_record_group_size).mean()) if len(per_record_group_size) else 1.0
    k_score = min(k_anonymity / 10, 1.0)
    equivalence_score = min(average_equivalence_class_size / 25, 1.0)
    privacy_score = float(np.clip(0.45 * k_score + 0.25 * equivalence_score + 0.30 * (1 - reidentification_risk), 0, 1))

    return {
        "k_anonymity": k_anonymity,
        "average_equivalence_class_size": average_equivalence_class_size,
        "reidentification_risk": reidentification_risk,
        "privacy_score": privacy_score,
    }


def utility_metrics(
    original: pd.DataFrame,
    masked: pd.DataFrame,
    baseline_ml: pd.DataFrame,
    masked_ml: pd.DataFrame,
) -> dict[str, float]:
    baseline_accuracy = float(baseline_ml["accuracy"].mean())
    masked_accuracy = float(masked_ml["accuracy"].mean())
    accuracy_drop = max(0.0, baseline_accuracy - masked_accuracy)
    accuracy_retention = masked_accuracy / baseline_accuracy if baseline_accuracy else 0.0
    correlation_preservation = _correlation_preservation(original, masked)
    statistical_similarity = _statistical_similarity(original, masked)
    information_loss = _information_loss(original, masked)
    utility_score = float(
        np.clip(
            0.35 * accuracy_retention
            + 0.25 * correlation_preservation
            + 0.25 * statistical_similarity
            + 0.15 * (1 - information_loss),
            0,
            1,
        )
    )

    return {
        "baseline_accuracy": baseline_accuracy,
        "masked_accuracy": masked_accuracy,
        "accuracy_drop": accuracy_drop,
        "information_loss": information_loss,
        "correlation_preservation": correlation_preservation,
        "statistical_similarity": statistical_similarity,
        "utility_score": utility_score,
        "utility_loss": 1 - utility_score,
    }


def combined_objective(privacy_score: float, utility_score: float, privacy_weight: float = 0.55) -> float:
    return float(privacy_weight * privacy_score + (1 - privacy_weight) * utility_score)


def _information_loss(original: pd.DataFrame, masked: pd.DataFrame) -> float:
    losses = []
    original_projection = numeric_projection(original)
    masked_projection = numeric_projection(masked)
    for column in set(original_projection.columns).intersection(masked_projection.columns):
        a = original_projection[column].to_numpy(dtype=float)
        b = masked_projection[column].to_numpy(dtype=float)
        data_range = max(np.max(a) - np.min(a), 1e-9)
        losses.append(float(np.mean(np.abs(a - b)) / data_range))

    for column in ["Gender", "Education", "Occupation"]:
        if column in original.columns and column in masked.columns:
            losses.append(float((original[column].astype(str) != masked[column].astype(str)).mean()))
    return float(np.clip(np.mean(losses) if losses else 0.0, 0, 1))


def _correlation_preservation(original: pd.DataFrame, masked: pd.DataFrame) -> float:
    original_corr = numeric_projection(original).corr().fillna(0)
    masked_corr = numeric_projection(masked).corr().fillna(0)
    columns = sorted(set(original_corr.columns).intersection(masked_corr.columns))
    if len(columns) < 2:
        return 0.0
    a = original_corr.loc[columns, columns].to_numpy()
    b = masked_corr.loc[columns, columns].to_numpy()
    upper = np.triu_indices_from(a, k=1)
    diff = np.mean(np.abs(a[upper] - b[upper]))
    return float(np.clip(1 - diff, 0, 1))


def _statistical_similarity(original: pd.DataFrame, masked: pd.DataFrame) -> float:
    original_projection = numeric_projection(original)
    masked_projection = numeric_projection(masked)
    similarities = []
    for column in set(original_projection.columns).intersection(masked_projection.columns):
        a = original_projection[column].to_numpy(dtype=float)
        b = masked_projection[column].to_numpy(dtype=float)
        scale = max(np.std(a), 1e-9)
        normalized_mean_gap = abs(np.mean(a) - np.mean(b)) / scale
        normalized_std_gap = abs(np.std(a) - np.std(b)) / scale
        similarities.append(np.exp(-(normalized_mean_gap + normalized_std_gap) / 2))
    return float(np.clip(np.mean(similarities) if similarities else 0.0, 0, 1))
