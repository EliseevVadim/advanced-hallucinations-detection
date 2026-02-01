import json
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from core.hallucinations_judging.constants import *


def load_prompt(path: str) -> ChatPromptTemplate:
    prompt = Path(path).read_text(encoding="utf-8")

    system_prompt, request = prompt.split(INPUT_MARK)
    system_prompt = system_prompt.replace(SYSTEM_MARK, "").strip()
    request = request.strip()

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", request)
    ])


def load_temp_file(path: str) -> set:
    labeled_ids = set()
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                labeled_ids.add(record["id"])
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return labeled_ids


def _compute_risk(token_logprobs: list[float], confidence_threshold: float) -> float:
    logprobs_count = len(token_logprobs)
    logprobs = np.asarray(token_logprobs, dtype=float)
    mean_logprobs = np.mean(logprobs)
    std_logprobs = np.std(logprobs)
    low_confidence_fraction = (logprobs < confidence_threshold).mean()

    risk = -mean_logprobs + 0.5 * std_logprobs + 0.3 * low_confidence_fraction

    risk = risk / np.log(logprobs_count + 1)
    return risk


def filter_by_risk_and_temperature(data: pd.DataFrame, confidence_threshold: float = -5.0) -> pd.DataFrame:
    temp_df = data.copy()
    temp_df = temp_df[temp_df["completion_tokens"] > 0]
    temp_df["_risk"] = temp_df["token_logprobs"].map(lambda x: _compute_risk(x, confidence_threshold))

    selected_rows = []

    for _, group in temp_df.groupby("id", sort=False):
        deterministic = group[group["temperature"] == 0]
        if len(deterministic) > 0:
            selected_rows.append(deterministic.iloc[0])
        stochastic = group[group["temperature"] > 0]
        if len(stochastic) > 0:
            min_idx = stochastic["_risk"].idxmin()
            max_idx = stochastic["_risk"].idxmax()

            selected_rows.append(stochastic.loc[min_idx])

            if max_idx != min_idx:
                selected_rows.append(stochastic.loc[max_idx])

            median_risk = stochastic["_risk"].median()
            mid_idx = (stochastic["_risk"] - median_risk).abs().idxmin()

            if mid_idx not in {min_idx, max_idx}:
                selected_rows.append(stochastic.loc[mid_idx])

    filtered_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    return filtered_df
