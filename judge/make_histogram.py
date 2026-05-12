
from judge.score_stats import get_ci
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from judge.score_db_utils import get_connection
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

    # replace : with _ for filename
    model_name = model_name.replace(":", "_")
    
    n = len(scores)
    mean = np.mean(scores)
    sd = np.std(scores, ddof=1)
    se = sd / np.sqrt(n)
    ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
    fig, ax = plt.subplots(figsize=(5, 3))                                           # ← smaller figure
    ax.hist(scores, bins=range(1, 12), align="left", color="steelblue",
            edgecolor="white", linewidth=0.8, rwidth=0.85)
    ax.axvline(mean, color="crimson", linewidth=1.8, linestyle="--", label=f"Mean: {mean:.2f}")
    ax.axvspan(ci_lower, ci_upper, alpha=0.28, color="crimson", label=f"95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")
    # ax.set_title(f"Score Distribution — {model_name}", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Score", fontsize=16)                                              # ← bigger
    ax.set_ylabel("Count", fontsize=16)                                              # ← bigger
    ax.set_xticks(range(1, 11))
    ax.tick_params(axis="both", labelsize=14)                                        # ← bigger tick labels
    ax.set_xlim(0.5, 10.5)
    ax.legend(fontsize=9)
    os.makedirs("results", exist_ok=True)
    output_path = f"results/histogram_{model_name}.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")



if __name__ == "__main__":
    models = ["llama3.1:8b", "deepseek_coder_cve", "deepseek-coder", "mistral_nemo_cve", "llama3.1_cve", "mistral-nemo:12b-instruct-2407-q8_0"]

    for model in models:
        plot_score_histogram(model)