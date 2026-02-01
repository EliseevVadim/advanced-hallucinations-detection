from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from tqdm import tqdm

from core.semantic_similarity_check.utils import count_parameters, free_memory


class SimilarityChecker:
    def __init__(self, model_name: str, device: str, metric: str, prefix: str = None, normalize: bool = True,
                 batch_size: int = 64):
        self.model_name = model_name
        if metric not in ["cosine", "euclidean", "manhattan", "dot"]:
            raise ValueError(f"metric must be 'cosine' or 'euclidean' or 'manhattan' or 'dot', got {metric}")
        self.metric = metric
        self.prefix = prefix
        self.normalize = normalize
        self.batch_size = batch_size

        self.model = SentenceTransformer(self.model_name, trust_remote_code=True, device=device)
        print(f"Model was successfully loaded. It has {count_parameters(self.model):,} parameters.")
        print("Model architecture:")
        print(self.model)

    def calculate_embeddings(self, answers: list[str], expected_answers: list[str]) -> tuple[
        ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]]:
        if len(answers) != len(expected_answers):
            raise ValueError(f"Number of answers must equal number of expected answers: {len(answers)}")
        answers = [f"{self.prefix}: {answer}" for answer in answers]
        expected_answers = [f"{self.prefix}: {expected_answer}" for expected_answer in expected_answers]

        answers_embeddings = self.model.encode(answers, convert_to_numpy=True, normalize_embeddings=self.normalize,
                                               batch_size=self.batch_size, show_progress_bar=False)
        expected_answers_embeddings = self.model.encode(expected_answers, convert_to_numpy=True,
                                                        normalize_embeddings=self.normalize,
                                                        batch_size=self.batch_size, show_progress_bar=False)

        return answers_embeddings, expected_answers_embeddings

    def calculate_similarity(self, answers_embeddings: np.ndarray,
                             expected_answers_embeddings: np.ndarray) -> np.ndarray:
        match self.metric:
            case "cosine":
                similarities = cosine_similarity(
                    answers_embeddings, expected_answers_embeddings
                )
                return np.diag(similarities).astype(float).tolist()
            case "euclidean":
                distances = euclidean_distances(
                    answers_embeddings, expected_answers_embeddings
                )
                return (1.0 / (1.0 + np.diag(distances))).astype(float).tolist()
            case "manhattan":
                distances = manhattan_distances(
                    answers_embeddings, expected_answers_embeddings
                )
                return (1.0 / (1.0 + np.diag(distances))).astype(float).tolist()
            case "dot":
                dot_products = np.sum(
                    answers_embeddings * expected_answers_embeddings, axis=1
                )
                return dot_products.astype(float).tolist()
            case _:
                raise ValueError(f"metric must be 'cosine' or 'euclidean' or 'manhattan' or 'dot', got {self.metric}")

    def check_similarity_on_dataframe(self, df: pd.DataFrame, answer_col: str, expected_answer_col: str,
                                      output_col: str) -> pd.DataFrame:
        df = df.copy()
        answers = df[answer_col].tolist()
        expected_answers = df[expected_answer_col].tolist()

        answers = [f"{self.prefix}: {answer}" for answer in answers]
        expected_answers = [f"{self.prefix}: {expected_answer}" for expected_answer in expected_answers]
        similarities = []

        with tqdm(total=len(df), desc="Calculating similarity", unit="rows", dynamic_ncols=True) as pbar:
            for i in range(0, len(df), self.batch_size):
                answers_batch = answers[i: i + self.batch_size]
                expected_answers_batch = expected_answers[i: i + self.batch_size]

                answers_embeddings, expected_answers_embeddings = self.calculate_embeddings(answers_batch,
                                                                                            expected_answers_batch)
                batch_scores = self.calculate_similarity(answers_embeddings, expected_answers_embeddings)
                similarities.extend(batch_scores)

                pbar.update(len(batch_scores))
                free_memory()

        df[output_col] = similarities
        return df
