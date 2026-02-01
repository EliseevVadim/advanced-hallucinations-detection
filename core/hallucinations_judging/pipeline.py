import json

import pandas as pd
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from core.hallucinations_judging.schema import JudgeOutput
from core.hallucinations_judging.utils import load_prompt, load_temp_file


class HallucinationsJudge:
    def __init__(self, prompt_path: str, model_name: str, api_url: str, api_key: str = None,
                 temperature: float = 0.0, max_tokens: int = 50, seed: int = 0):
        self.prompt = load_prompt(prompt_path)

        self.llm = ChatOpenAI(
            base_url=api_url,
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )

        self.chain = self.llm.with_structured_output(JudgeOutput)

    def calculate_hallucination_score(self, prompt: str, answer: str, expected_answer: str,
                        max_retries: int = 5) -> float | None:
        for attempt in range(max_retries + 1):
            try:
                result = self.chain.invoke(
                    self.prompt.format_messages(
                        prompt=prompt,
                        answer=answer,
                        expected_answer=expected_answer
                    )
                )

                return result.hallucination_score
            except (ValueError, OutputParserException):
                if attempt == max_retries:
                    return None
        return None


def label_with_checkpointing(data: pd.DataFrame, judge: HallucinationsJudge, temp_file: str) -> None:
    try:
        labeled_ids = load_temp_file(temp_file)

        print(f"Already labeled {len(labeled_ids)} records")

        with open(temp_file, "a", encoding="utf-8") as f:
            for idx, row in tqdm(data.iterrows(), total=len(data)):
                if idx in labeled_ids:
                    continue

                score = judge.calculate_hallucination_score(prompt=row["Prompt"], answer=row["Answer"],
                                                            expected_answer=row["expected_answer"])

                record = {
                    "id": idx,
                    "hallucination_score": score
                }

                f.write(json.dumps(record) + "\n")
                f.flush()
        print(f"All records are labeled")
    except KeyboardInterrupt:
        print("Process stopped by user")
        labeled_ids = load_temp_file(temp_file)

        print(f"Already labeled {len(labeled_ids)} records")


def attach_hallucinations_scores(data: pd.DataFrame, scores_path: str, index_col: str = "index") -> pd.DataFrame:
    scores = pd.read_json(scores_path, lines=True)
    result = data.merge(
        scores[["id", "hallucination_score"]],
        left_index=True,
        right_on="id",
        how="left",
    )
    return result
