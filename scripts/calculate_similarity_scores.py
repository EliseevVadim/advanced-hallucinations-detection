import pandas as pd

from core.semantic_similarity_check.pipeline import SimilarityChecker
from core.semantic_similarity_check.utils import get_best_device, init_random_seed

data_path = "../data"
model_name = "nomic-ai/nomic-embed-text-v1.5"

device = get_best_device()

init_random_seed(316)

print("Device that will be used:", device)

similarity_checker = SimilarityChecker(
    model_name=model_name,
    device=str(device),
    metric="cosine",
    normalize=True,
    batch_size=32,
    prefix="search_query"
)

data = pd.read_csv(f"{data_path}/out/queries_with_hallucination_score.csv", index_col=0)

data_with_similarity_scores = similarity_checker.check_similarity_on_dataframe(df=data, answer_col="Answer",
                                                                               expected_answer_col="expected_answer",
                                                                               output_col="similarity_score")

data_with_similarity_scores.to_csv(f"{data_path}/out/queries_with_similarity_score.csv")
print("Similarity scores saved to out/queries_with_similarity_score.csv")
