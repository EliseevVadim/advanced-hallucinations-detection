import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.data_preparation.transformations import create_target_score_and_column


def plot_correlation_matrix(data: pd.DataFrame) -> None:
    numeric_cols = data.select_dtypes(include="number")
    correlations = numeric_cols.corr()

    plt.figure(figsize = (12, 10))
    sns.heatmap(correlations, annot=True, cbar=True, square=False, fmt=".2f")

    plt.title("Correlation Matrix Between Numeric Columns")
    plt.show()


def plot_pie_chart(data: pd.DataFrame, column: str, title: str) -> None:
    counts = data[column].value_counts(dropna=False)
    colors = sns.color_palette("viridis", len(counts))
    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=counts.index.astype(str),
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )
    plt.setp(autotexts, size=12, weight="bold", color="black")
    plt.setp(texts, size=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.show()


def build_weights_heatmap(
    df: pd.DataFrame,
    hallucination_score_column: str = "hallucination_score",
    similarity_score_column: str = "similarity_score",
    step: float = 0.1,
):
    weights = np.round(np.arange(0.0, 1.0 + step, step), 2)
    matrix = np.full((len(weights), len(weights)), np.nan)

    for i, hallucination_weight in enumerate(weights):
        for j, similarity_weight in enumerate(weights):
            if hallucination_weight == 0 and similarity_weight == 0:
                continue

            components = [
                {
                    "column": hallucination_score_column,
                    "weight": hallucination_weight,
                    "higher_is_better": False,
                    "normalize": False,
                },
                {
                    "column": similarity_score_column,
                    "weight": similarity_weight,
                    "higher_is_better": True,
                    "normalize": False,
                },
            ]

            scored = create_target_score_and_column(df, components)
            matrix[i, j] = scored["target"].mean()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=weights,
        yticklabels=weights,
        cmap="viridis",
        annot=True,
        fmt=".2f",
    )

    plt.xlabel("Similarity weight")
    plt.ylabel("Hallucination weight")
    plt.title("Fraction of target=1 for different weight combinations")
    plt.show()
