from scipy import stats
import numpy as np
from judge.score_db_utils import get_connection
from typing import Tuple, Optional


def get_ci(scores, confidence_level: float = 0.95) -> Optional[Tuple[float, float]]:
    n = len(scores)
    if n < 1:
        return None
    avg = np.mean(scores)
    sd = np.std(scores, ddof=1)
    se = sd / np.sqrt(n)
    a = 1 - confidence_level
    t_value = stats.t.ppf(1 - a/2, df=n-1)
    ci = (avg - t_value * se, avg + t_value * se)
    return ci


def get_confidence_intervals(model: str, confidence_level: float = 0.95) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    scores = {'human': [], 'llm': []}
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


def get_median_scores(model: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns the median (human_median, llm_median) for a given model.
    Returns (None, None) if no scores are found.
    """
    scores = {'human': [], 'llm': []}
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

    human_median = float(np.median(scores['human'])) if scores['human'] else None
    llm_median = float(np.median(scores['llm'])) if scores['llm'] else None
    return human_median, llm_median


def get_failure_rate(model: str, threshold: int = 3) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns the critical failure rate (human_rate, llm_rate) for a given model.
    Failure is defined as score < threshold (default: 3).
    Rate = count(score < threshold) / total scores.
    Returns (None, None) if no scores are found.
    """
    scores = {'human': [], 'llm': []}
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

    human_rate = sum(s < threshold for s in scores['human']) / len(scores['human']) if scores['human'] else None
    llm_rate = sum(s < threshold for s in scores['llm']) / len(scores['llm']) if scores['llm'] else None
    return human_rate, llm_rate


def get_excellent_rate(model: str, threshold: int = 8) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns the excellent rate (human_rate, llm_rate) for a given model.
    Excellent is defined as score > threshold (default: 8).
    Rate = count(score > threshold) / total scores.
    Returns (None, None) if no scores are found.
    """
    scores = {'human': [], 'llm': []}
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

    human_rate = sum(s > threshold for s in scores['human']) / len(scores['human']) if scores['human'] else None
    llm_rate = sum(s > threshold for s in scores['llm']) / len(scores['llm']) if scores['llm'] else None
    return human_rate, llm_rate


def get_most_frequent_score(model: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns the most frequent (mode) score (human_mode, llm_mode) for a given model.
    If multiple scores share the highest frequency, the lowest value is returned.
    Returns (None, None) if no scores are found.
    """
    scores = {'human': [], 'llm': []}
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

    human_mode = int(stats.mode(scores['human'], keepdims=False).mode) if scores['human'] else None
    llm_mode = int(stats.mode(scores['llm'], keepdims=False).mode) if scores['llm'] else None
    return human_mode, llm_mode


def get_sd(model: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns the standard deviation (human_sd, llm_sd) for a given model.
    Returns (None, None) if no scores are found.
    """
    scores = {'human': [], 'llm': []}
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

    human_sd = float(np.std(scores['human'], ddof=1)) if scores['human'] else None
    llm_sd = float(np.std(scores['llm'], ddof=1)) if scores['llm'] else None
    return human_sd, llm_sd


if __name__ == "__main__":
    models = ["llama3.1:8b", "deepseek_coder_cve", "deepseek-coder", "mistral_nemo_cve", "llama3.1_cve", "mistral-nemo:12b-instruct-2407-q8_0"]
    for model in models:
        print(f"\n=== Model: {model} ===")

        avg_human, avg_llm = find_average_scores_by_model(model)
        print(f"  Average:        Human={avg_human}, LLM={avg_llm}")

        human_ci, llm_ci = get_confidence_intervals(model)
        print(f"  95% CI:         Human={human_ci}, LLM={llm_ci}")

        human_median, llm_median = get_median_scores(model)
        print(f"  Median:         Human={human_median}, LLM={llm_median}")

        human_failure, llm_failure = get_failure_rate(model)
        human_failure_str = f"{human_failure:.2%}" if human_failure is not None else "N/A"
        llm_failure_str = f"{llm_failure:.2%}" if llm_failure is not None else "N/A"
        print(f"  Failure rate:   Human={human_failure_str}, LLM={llm_failure_str}")

        human_excellent, llm_excellent = get_excellent_rate(model)
        human_excellent_str = f"{human_excellent:.2%}" if human_excellent is not None else "N/A"
        llm_excellent_str = f"{llm_excellent:.2%}" if llm_excellent is not None else "N/A"
        print(f"  Excellent rate: Human={human_excellent_str}, LLM={llm_excellent_str}")

        human_mode, llm_mode = get_most_frequent_score(model)
        print(f"  Mode:           Human={human_mode}, LLM={llm_mode}")

        human_sd, llm_sd = get_sd(model)
        print(f"  Standard deviation: Human={human_sd}, LLM={llm_sd}")