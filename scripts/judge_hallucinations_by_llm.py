import pandas as pd

from core.hallucinations_judging.pipeline import HallucinationsJudge, label_with_checkpointing, \
    attach_hallucinations_scores

data_path = "../data"
api_url = "http://127.0.0.1:1234/v1"
api_key = "local"
model_name = "google/gemma-3-12b"
prompt_path = "../prompts/judge_hallucination.txt"
scores_path = f"{data_path}/temp/judge_log.jsonl"

data = pd.read_csv(f"{data_path}/filtered_queries.csv", index_col=0)

judge = HallucinationsJudge(
    api_url=api_url,
    api_key=api_key,
    model_name=model_name,
    prompt_path=prompt_path,
    temperature=0.1,
    seed=316
)

label_with_checkpointing(
    data=data,
    judge=judge,
    temp_file=scores_path
)

data_with_hallucination_score = attach_hallucinations_scores(
    data=data,
    scores_path=scores_path
)

data_with_hallucination_score.to_csv(f"{data_path}/out/queries_with_hallucination_score.csv")
print("Hallucination scores saved to out/queries_with_hallucination_score.csv")
