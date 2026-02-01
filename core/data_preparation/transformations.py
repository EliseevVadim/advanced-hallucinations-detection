import ast

import numpy as np
import pandas as pd


def parse_logprobs(logprobs: pd.Series) -> np.ndarray | None:
    if pd.isna(logprobs):
        return None
    try:
        logprobs = ast.literal_eval(logprobs)
        return np.array(logprobs, dtype=float)
    except Exception:
        return None


def calculate_logprobs_entropy(logprobs: pd.Series) -> np.ndarray | None:
    probabilities = np.exp(logprobs)
    probabilities /= np.sum(probabilities)
    return -(probabilities * np.log(probabilities + 1e-9)).sum()


def calculate_logprobs_stats(logprobs: np.ndarray, low_logprob_threshold: float = -5.0) -> pd.Series:
    if logprobs is None or logprobs.size == 0:
        return pd.Series({
            "mean_logprob": np.nan,
            "median_logprob": np.nan,
            "min_logprob": np.nan,
            "max_logprob": np.nan,
            "std_logprob": np.nan,
            "low_logprob_frac": np.nan
        })

    return pd.Series({
        "mean_logprob": float(np.mean(logprobs)),
        "median_logprob": float(np.median(logprobs)),
        "min_logprob": float(np.min(logprobs)),
        "max_logprob": float(np.max(logprobs)),
        "std_logprob": float(np.std(logprobs)),
        "low_logprob_frac": float(np.mean(logprobs < low_logprob_threshold))
    })


def create_target_score_and_column(data: pd.DataFrame, components: list[dict],
                                   score_column: str = "target_score", target_column: str = "target",
                                   positive_class_threshold: float = 0.5,
                                   eps: float = 1e-9) -> pd.DataFrame:
    data = data.copy()
    total_score = np.zeros(len(data), dtype=float)
    total_weight = 0.0

    for component in components:
        column = component["column"]
        weight = component.get("weight", 1.0)
        higher_is_better = component.get("higher_is_better", False)
        normalize = component.get("normalize", False)

        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        values = data[column].astype(float)

        if normalize:
            minimal_value = values.min()
            maximum_value = values.max()
            values = (values - minimal_value) / (maximum_value - minimal_value + eps)

        if higher_is_better:
            values = 1 - values

        total_score += values * weight
        total_weight += weight

    if total_weight == 0.0:
        raise ValueError("Sum of weights must be > 0.0")

    data[score_column] = (total_score / total_weight).astype(float)
    data[target_column] = (data[score_column] >= positive_class_threshold).astype(float)
    return data
