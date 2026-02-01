import ast
import json
import os
import random

import numpy as np
import pandas as pd


def init_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_json(path: str) -> dict:
    with open(path) as json_file:
        return json.load(json_file)


def build_queries_df(queries_path: str) -> pd.DataFrame:
    queries = os.listdir(queries_path)
    queries = [read_json(os.path.join(queries_path, query)) for query in queries]

    rows = []

    for item in queries:
        base = {
            "id": item["id"],
            "Prompt": item["prompt"],
            "expected_answer": item["expected_answer"],
            "model": item["output"]["model"],
        }

        results = item["output"].get("results", {})

        for result_id, result in results.items():
            row = base | {
                "result_id": int(result_id),
                "temperature": result.get("temperature"),
                "top_p": result.get("top_p"),
                "completion_tokens": result.get("completion_tokens"),
                "tokens": result.get("tokens"),
                "Answer": "".join(result.get("tokens")),
                "token_logprobs": result.get("token_logprobs"),
                "finish_reason": result.get("finish_reason"),
            }
            rows.append(row)

    queries_df = pd.DataFrame(rows)
    return queries_df


def remove_non_text_answers(data: pd.DataFrame) -> pd.DataFrame:
    mask = data["Answer"].astype(str).str.contains(r"[A-Za-zА-Яа-я0-9]", regex=True, na=False)
    print(f"Удалено неинформативных записей: {len(data[~mask])}")
    return data.loc[mask].reset_index(drop=True)
