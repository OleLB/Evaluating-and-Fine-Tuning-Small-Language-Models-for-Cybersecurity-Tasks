"""This script uses a LLM to evaluate the performance of a small LLM"""
from judge.score_db_utils import get_all_unscored_by_llm, update_llm_score
from utils.queryGPT import queryGPT

PROMPT_PATH = "prompts/LLM_judge_prompt.txt"
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()


def main(model_to_score: str, judge_model: str):
    print(f"Starting LLM scoring for model: {model_to_score} using judge model: {judge_model}")
    unscored_rows = get_all_unscored_by_llm(model_to_score)

    if not unscored_rows or len(unscored_rows) == 0:
        print("No unscored entries found for the specified model.")
        return

    total_cost = 0.0

    for row in unscored_rows:
        # "input" is resolved via a JOIN in get_all_unscored_by_llm — see score_db_utils.py
        input_text = row["input"]
        output_text = row["output"]
        rag_response = row["rag_output"] if row["rag_output"] else None
        cve_data = row["cve_data"] if row["cve_data"] else None

        prompt = SYSTEM_PROMPT
        prompt += f"\n\nQuestion asked by user:\n{input_text}"
        if rag_response:
            prompt += f"\n\nRAG system response:\n{rag_response}"
        if cve_data:
            prompt += f"\n\nCVE data used in the response:\n{cve_data}"
        prompt += f"\n\nModel's answer to evaluate:\n{output_text}"

        llm_response, usage_cost = queryGPT(prompt, judge_model)

        try:
            llm_reasoning, llm_score_str = llm_response.split("| Score:")
        except ValueError:
            print(f"Warning: unexpected LLM response format for row {row['id']}: '{llm_response}' — skipping")
            continue

        llm_score = llm_score_str.strip()
        llm_reasoning = llm_reasoning.replace("Reasoning:", "").strip()

        try:
            score = int(llm_score.strip())
            if not 1 <= score <= 10:
                raise ValueError
        except ValueError:
            print(f"Warning: unexpected score format for row {row['id']}: '{llm_score}' — skipping")
            continue

        update_llm_score(row["id"], score, llm_reasoning)
        total_cost += usage_cost["total_cost"]
        print(f"Row {row['id']} scored: {score}")

    print(f"Finished scoring. Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    JUDGE_MODEL = "gpt-5"
    MODEL_TO_SCORE = "llama3.1_cve"
    main(MODEL_TO_SCORE, JUDGE_MODEL)