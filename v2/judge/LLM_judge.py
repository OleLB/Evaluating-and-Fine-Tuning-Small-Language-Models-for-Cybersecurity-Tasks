from judge.score_db_utils import get_all_unscored_by_llm, update_llm_score
from utils.queryGPT import queryGPT

# SYSTEM_PROMPT = """You are evaluating the output of a small language model that serves as an offline CVE and vulnerability lookup tool for penetration testers and security researchers.

# IMPORTANT CONTEXT:
# This system uses deterministic RAG retrieval from curated CVE, NVD, and mitigation databases. Specific technical details (e.g., affected versions, function names, CVSS severity, exploit conditions, mitigation technique IDs) are retrieved from structured data sources and should be assumed correct unless they clearly contradict established cybersecurity knowledge.

# Do NOT assume hallucination solely because an answer contains highly specific technical details.
# Only penalize factual accuracy if information is clearly false, internally inconsistent, logically contradictory, or demonstrably misaligned with known cybersecurity facts.

# The model is designed to handle three types of queries:
#   1. Questions about a specific CVE ID — it should report what the NVD database record contains.
#   2. Questions about vulnerabilities in a specific software version — it should query its vector database and return relevant CVEs.
#   3. General cybersecurity knowledge questions — it should answer from its training knowledge.

# **Fact-checking instruction:**
# When evaluating factual accuracy, you MAY use a web search or other online resources to verify:

# - CVE descriptions and affected components
# - CVSS scores and severity ratings
# - Known exploit conditions or attack vectors
# - Confirmed mitigations and patches

# If the output contains details that cannot be verified online or through authoritative sources, do not penalize; assume the information from the RAG system is correct. Only penalize if the output is demonstrably false according to reliable sources.

# Score the output from 1 to 10 based on the following criteria:

# - Factual accuracy:
#   Penalize ONLY for verifiable falsehoods, internal contradictions, incorrect vulnerability classifications, clearly wrong severity ratings, or misinterpretation of the query.
#   Do NOT penalize for lack of citation, high specificity, or confident tone.

# - Honesty about uncertainty:
#   If the information is incomplete or unavailable, does the model acknowledge uncertainty rather than fabricate missing data? Reward appropriate uncertainty when relevant.

# - Relevance:
#   Does the answer directly and clearly address the user’s question?

# - Conciseness:
#   Is the answer focused and professional, without unnecessary padding?

# - Appropriate behavior for query type:
#   - For CVE ID queries: does it summarize the record accurately?
#   - For software version queries: does it identify relevant CVEs rather than speculate?
#   - For general knowledge questions: is the explanation technically sound?

# Scoring guide:
#   9-10: Technically accurate, directly answers the question, no contradictions or clear errors.
#   7-8: Mostly accurate with minor omissions or slight irrelevance, no clear false statements.
#   5-6: Partially correct but missing important elements or slightly misaligned with the query.
#   3-4: Contains clear inaccuracies, logical inconsistencies, or largely fails to answer the question.
#   1-2: Factually incorrect, internally contradictory, or completely fails to address the query.

# Output a single integer from 1 to 10. Nothing else. No explanation, no punctuation, just the number."""


SYSTEM_PROMPT = """You are a calibrated grading model evaluating the output of a small language model used as an offline CVE and vulnerability lookup assistant.

You will receive:
- The user’s question
- The RAG system response (authoritative retrieved data)
- Structured CVE data (authoritative database content)
- The model’s final answer

IMPORTANT:
The RAG system response and CVE data provided here are authoritative and must be treated as ground truth.
Judge factual consistency primarily against this provided data.

Do NOT override the provided RAG/CVE data with outside knowledge.
Do NOT assume hallucination merely because an answer contains detailed technical information.
Judge only based on consistency, completeness, and relevance.

--------------------------------
HOW TO EVALUATE
--------------------------------

1. Factual Consistency (Primary Factor)
Evaluate how well the model’s answer aligns with the provided RAG and CVE data.

Severity tiers:
- Minor inaccuracy: Slight wording issue, small omission, mild overgeneralization.
- Moderate issue: Missing important elements, partially misaligned claim, small unsupported inference.
- Major issue: Clear contradiction, fabricated technical detail (e.g., wrong CVSS, wrong affected version), or internally inconsistent logic.

2. Unsupported Additions
- Light contextual additions that do not contradict the data → small penalty at most.
- Specific invented technical details not supported by RAG/CVE data → moderate to major penalty depending on impact.

3. Completeness
- For CVE ID queries: Should reflect the key details present in the provided data.
- For version queries: Should identify relevant CVEs without speculation.
- For general cybersecurity questions: Should be technically coherent and accurate.

4. Relevance and Focus
- Directly answers the question → reward.
- Noticeable irrelevance or padding → small penalty.
- Fails to answer the question → major penalty.

5. Calibration Guidance
Use the full 1–10 scale.
Do NOT collapse to extremes unless justified.

General calibration:
- 9–10: Highly accurate, complete, no meaningful issues.
- 7–8: Solid answer with minor issues or small omissions.
- 5–6: Reasonable but noticeably incomplete, slightly misaligned, or contains moderate issues.
- 3–4: Significant inaccuracies, contradictions, or major missing elements.
- 1–2: Fundamentally incorrect, largely fabricated, internally contradictory, or completely fails to answer.

IMPORTANT:
A single minor mistake should NOT drop the score below 5.
Reserve scores 1–2 for clearly broken or fabricated answers.
Reserve 9–10 for genuinely strong, well-aligned responses.

--------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------
Output a single integer from 1 to 10.
No explanation.
No text.
Only the number.
"""


def main(model_to_score: str, judge_model: str):
    print(f"Starting LLM scoring for model: {model_to_score} using judge model: {judge_model}")
    unscored_rows = get_all_unscored_by_llm(model_to_score)
    if not unscored_rows or len(unscored_rows) == 0:
        print("No unscored entries found for the specified model.")
        return

    total_cost = 0.0
    for row in unscored_rows:
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

        llm_score, usage_cost = queryGPT(prompt, judge_model)
        
        # Strip and validate the score before saving
        try:
            score = int(llm_score.strip())
            if not 1 <= score <= 10:
                raise ValueError
        except ValueError:
            print(f"Warning: unexpected score format for row {row['id']}: '{llm_score}' — skipping")
            continue

        update_llm_score(row["id"], score)
        total_cost += usage_cost["total_cost"]
        print(f"Row {row['id']} scored: {score}")

    print(f"Finished scoring. Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    JUDGE_MODEL = "gpt-5"
    MODEL_TO_SCORE = "mistral-nemo-cve2"
    main(MODEL_TO_SCORE, JUDGE_MODEL)