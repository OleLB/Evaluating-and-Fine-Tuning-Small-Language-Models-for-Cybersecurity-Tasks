
from scipy import stats
import numpy as np
from judge.score_db_utils import get_all_unscored_by_llm, update_llm_score, get_connection
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import os

def plot_score_histogram(model_name: str) -> None:
    """
    Plot a histogram of LLM evaluation scores and save to /results/.

    Args:
        scores: List of integer scores (1-10)
        model_name: Name of the model (used in title and filename)
    """
    # connect to DB
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT LLM_score
            FROM Scores
            WHERE model_name = ? AND LLM_score IS NOT NULL
            """,
            (model_name,),
        )
        scores = [row[0] for row in cursor.fetchall()]

    n = len(scores)
    mean = np.mean(scores)
    sd = np.std(scores, ddof=1)
    se = sd / np.sqrt(n)
    ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(scores, bins=range(1, 12), align="left", color="steelblue",
            edgecolor="white", linewidth=0.8, rwidth=0.85)

    ax.axvline(mean, color="crimson", linewidth=1.8, linestyle="--", label=f"Mean: {mean:.2f}")
    ax.axvspan(ci_lower, ci_upper, alpha=0.28, color="crimson", label=f"95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")

    ax.set_title(f"Score Distribution — {model_name}", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.legend(fontsize=10)

    # stats_text = f"N={n}  |  Mean={mean:.2f}  |  SD={sd:.2f}"
    # ax.text(0.5, -0.12, transform=ax.transAxes,
    #         ha="center", fontsize=9.5, color="gray")

    os.makedirs("results", exist_ok=True)
    output_path = f"results/histogram_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")



def get_ci(scores, confidence_level: float = 0.95) -> Optional[Tuple[float, float]]:
    n = len(scores)
    if n < 1:
        return None

    avg = np.mean(scores)
    sd = np.std(scores, ddof=1)  # Sample standard deviation
    se = sd / np.sqrt(n)  # Standard error

    a = 1 - confidence_level
    t_value = stats.t.ppf(1 - a/2, df=n-1)
    ci = (avg - t_value * se, avg + t_value * se)

    return ci


def get_confidence_intervals(model: str, confidence_level: float = 0.95) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Return the CI for nodel score. Using t-distribtion.
    df = n-1, where n = score count
    [avg +- t_value,df * (sd / sqrt(n))]
    Returns:
        tuple(human_ci, llm_ci)
    """
    scores = {'human': [], 'llm': []}
    # get all scores for the model
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT Human_score, LLM_score
            FROM Scores
            WHERE model_name = ?
            """,
            (model,),
        )
        for row in cursor.fetchall():
            human_score, llm_score = row
            if human_score is not None:
                scores['human'].append(human_score)
            if llm_score is not None:
                scores['llm'].append(llm_score)

    # llm judge first
    human_ci = get_ci(scores['human'], confidence_level)
    llm_ci = get_ci(scores['llm'], confidence_level)
    return human_ci, llm_ci


def find_average_scores_by_model(model: str) -> Tuple[Optional[float], Optional[float]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                AVG(Human_score),
                AVG(LLM_score)
            FROM Scores
            WHERE (Human_score IS NOT NULL AND model_name = ?)
               OR (LLM_score IS NOT NULL AND model_name = ?)
            """,
            (model, model),
        )
        result = cursor.fetchone()
        return result if result else (None, None)
    


if __name__ == "__main__":
    models = ["mistral-nemo-cve2", "llama3.1_cve"]

    for model in models:
        avg_human, avg_llm = find_average_scores_by_model(model)
        print(f"Average scores for model '{model}': Human_score = {avg_human}, LLM_score = {avg_llm}")
        human_ci, llm_ci = get_confidence_intervals(model)
        print(f"Model: {model}")
        print(f"  Human_score 95% CI: {human_ci}")
        print(f"  LLM_score 95% CI: {llm_ci}")

        plot_score_histogram(model)