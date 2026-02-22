from judge.score_db_utils import get_all_unscored_by_llm, update_llm_score
from utils.queryGPT import queryGPT

# SYSTEM_PROMPT = """You are evaluating the output of a small language model that serves as an offline CVE and vulnerability lookup tool for penetration testers and security researchers.

# The model is designed to handle three types of queries:
#   1. Questions about a specific CVE ID — it should report what the NVD database record contains
#   2. Questions about vulnerabilities in a specific software version — it should query its vector database and return relevant CVEs
#   3. General cybersecurity knowledge questions — it should answer from its training knowledge

# Score the output from 1 to 10 based on the following criteria:
# - Factual accuracy: Does the answer contain verifiable correct information? Penalize heavily for hallucinated CVE details, wrong severity ratings, or fabricated exploit conditions.
# - Honesty about uncertainty: If the data is incomplete or the model doesn't know, does it say so rather than guessing? Reward this behavior.
# - Relevance: Does the answer actually address what was asked?
# - Conciseness: Is the answer appropriately focused, or does it pad with irrelevant information?
# - Appropriate tool use: For software version queries, did it attempt a database lookup? For CVE ID queries, did it avoid unnecessary tool use?

# Scoring guide:
#   9-10: Accurate, honest about gaps, directly answers the question, no hallucinations
#   7-8: Mostly accurate with minor omissions or slight irrelevance, no significant hallucinations
#   5-6: Partially correct but missing key details or includes some unverified claims presented as fact
#   3-4: Significant inaccuracies or hallucinated details, or mostly irrelevant response
#   1-2: Factually wrong, confidently hallucinates, or completely fails to address the question

# Output a single integer from 1 to 10. Nothing else. No explanation, no punctuation, just the number."""


SYSTEM_PROMPT = """You are evaluating the output of a small language model that serves as an offline CVE and vulnerability lookup tool for penetration testers and security researchers.

IMPORTANT CONTEXT:
This system uses deterministic RAG retrieval from curated CVE, NVD, and mitigation databases. Specific technical details (e.g., affected versions, function names, CVSS severity, exploit conditions, mitigation technique IDs) are retrieved from structured data sources and should be assumed correct unless they clearly contradict established cybersecurity knowledge.

Do NOT assume hallucination solely because an answer contains highly specific technical details.
Only penalize factual accuracy if information is clearly false, internally inconsistent, logically contradictory, or demonstrably misaligned with known cybersecurity facts.

The model is designed to handle three types of queries:
  1. Questions about a specific CVE ID — it should report what the NVD database record contains.
  2. Questions about vulnerabilities in a specific software version — it should query its vector database and return relevant CVEs.
  3. General cybersecurity knowledge questions — it should answer from its training knowledge.

**Fact-checking instruction:**
When evaluating factual accuracy, you MAY use a web search or other online resources to verify:

- CVE descriptions and affected components
- CVSS scores and severity ratings
- Known exploit conditions or attack vectors
- Confirmed mitigations and patches

If the output contains details that cannot be verified online or through authoritative sources, do not penalize; assume the information from the RAG system is correct. Only penalize if the output is demonstrably false according to reliable sources.

Score the output from 1 to 10 based on the following criteria:

- Factual accuracy:
  Penalize ONLY for verifiable falsehoods, internal contradictions, incorrect vulnerability classifications, clearly wrong severity ratings, or misinterpretation of the query.
  Do NOT penalize for lack of citation, high specificity, or confident tone.

- Honesty about uncertainty:
  If the information is incomplete or unavailable, does the model acknowledge uncertainty rather than fabricate missing data? Reward appropriate uncertainty when relevant.

- Relevance:
  Does the answer directly and clearly address the user’s question?

- Conciseness:
  Is the answer focused and professional, without unnecessary padding?

- Appropriate behavior for query type:
  - For CVE ID queries: does it summarize the record accurately?
  - For software version queries: does it identify relevant CVEs rather than speculate?
  - For general knowledge questions: is the explanation technically sound?

Scoring guide:
  9-10: Technically accurate, directly answers the question, no contradictions or clear errors.
  7-8: Mostly accurate with minor omissions or slight irrelevance, no clear false statements.
  5-6: Partially correct but missing important elements or slightly misaligned with the query.
  3-4: Contains clear inaccuracies, logical inconsistencies, or largely fails to answer the question.
  1-2: Factually incorrect, internally contradictory, or completely fails to address the query.

Output a single integer from 1 to 10. Nothing else. No explanation, no punctuation, just the number."""


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

        prompt = SYSTEM_PROMPT
        prompt += f"\n\nQuestion asked by user:\n{input_text}\n\nAnswer given by the model:\n{output_text}"

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