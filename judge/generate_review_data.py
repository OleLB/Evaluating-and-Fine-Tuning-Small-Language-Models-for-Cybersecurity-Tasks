"""This script uses an LLM to act as a user, the LLM will ask questions to the lora+rag system, the inputs and outputs will be stored in a database"""
from main import handle_user_query # function to query lora+rag system
from judge.score_db_utils import add_entry # function to add input/output pairs to the database
from utils.queryGPT import queryGPT # function to query the LLM
from utils.querySQLite import getRandomCVEs, getCVEInfo # function to get random CVEs from the database

NUM_TO_GENERATE = 191

PROMPT = """You are roleplaying as a real human user typing a quick message into a cybersecurity chatbot. The chatbot is a vulnerability lookup tool — users typically ask it one of three things:
  1. Look up a specific CVE by ID (e.g. "tell me about CVE-2024-1234" or "whats CVE-2024-1234 about")
  2. Find vulnerabilities for a piece of software (e.g. "any vulns in apache 2.4.1?" or "is nginx 1.18 vulnerable to anything?")
  3. Ask a general cybersecurity question (e.g. "whats the difference between xss and csrf" or "how does sql injection work")

You will be given CVE data. Use it to inspire ONE question of type 1, 2, or 3. Weight toward type 1 and 2.

Rules for realism — this is the most important part:
- Sound like a human typing quickly, not an AI writing a report. Short, direct, possibly with minor typos or informal grammar.
- Ask ONE thing only. Do not combine multiple questions or sub-questions into one message.
- Do not use formal structure like colons, parentheses with version ranges, or bullet points in the question.
- Do not reference things the user wouldn't know just from using the tool (e.g. internal servlet names, header names, source code details)
- Vary the style: sometimes a fragment ("vulns for openssh 8.1?"), sometimes casual ("hey whats the deal with CVE-2021-3156"), sometimes direct ("is wordpress 5.6 affected by anything critical")
- Output ONLY the question. No preamble, no explanation, no quotation marks around it.

CVE data:
"""

PROMPT2 = """You are roleplaying as a real human user typing a quick message into a cybersecurity chatbot. The chatbot is a vulnerability lookup tool — users typically ask it one of three things:
  1. Look up a specific CVE by ID (e.g. "tell me about CVE-2024-1234" or "whats CVE-2024-1234 about")
  2. Find vulnerabilities for a piece of software (e.g. "any vulns in apache 2.4.1?" or "is nginx 1.18 vulnerable to anything?")
  3. Ask a general cybersecurity question (e.g. "whats the difference between xss and csrf" or "how does sql injection work")

You will be given CVE data. Use it to inspire ONE question of type 1, 2, or 3. Weight toward type 1 and 2.

Rules for realism — this is the most important part:
- Sound like a human typing quickly, not an AI writing a report. Short, direct, possibly with minor typos or informal grammar.
- Ask ONE thing only. Do not combine multiple questions or sub-questions into one message.
- Do not use formal structure like colons, parentheses with version ranges, or bullet points in the question.
- Do not reference things the user wouldn't know just from using the tool (e.g. internal servlet names, header names, source code details)
- Vary the style: sometimes a fragment ("vulns for openssh 8.1?"), sometimes casual ("hey whats the deal with CVE-2021-3156"), sometimes direct ("is wordpress 5.6 affected by anything critical")
- Output ONLY the question. No preamble, no explanation, no quotation marks around it.
- Avoid asking if a software is vulnerable to a specific CVE, as this is not how users typically ask questions. Instead, they might ask "is software X vulnerable to anything?" or "are there any vulns for <software_name> version x.y.z?"

CVE data:
"""

#! Currently the questions dont seem human-like. They need to be less formal and less dense. 


def prep_question():
    """
    A large LLM will generate a question.
    Provide details on a random CVE from the database
    There are three types of valid questions:
        - A question about a specific CVE ID (Use the provided CVE ID)
        - A question about a vulnerability described in the CVE data (Use the provided CVE description)
        - A general cybersecurity question (may be unrelated to the CVE data, but must be about cybersecurity)
    Example CVE data:
    CVE-2024-52367,cpe:2.3:a:ibm:concert:1.0.1:*:*:*:*:*:*:*,Application 'concert' by ibm (version 1.0.1)
    IBM Concert Software 1.0.0, 1.0.1, 1.0.2, 1.0.2.1, and 1.0.3 could disclose sensitive system information to an unauthorized actor that could be used in further attacks against the system.
    Examples questions:
    - "Are there any vulnerabilities related to concert version version 1.0.1?"
    - "tell me about CVE-2024-52367"
    - "explain SQLi and how it can be mitigated"
    You should make a question based on this data:
    """
    random_cve_id = getRandomCVEs(limit=1)[0]
    random_cve = getCVEInfo(random_cve_id)
    cve_id = random_cve["CVE_ID"]
    cve_description = random_cve["Description"]
    cve_vulnerable_software = random_cve["Known_Vulnerable_Software"]

    # Add the CVE data to the prompt
    cve_context = f"CVE ID: {cve_id}\nAffected Software: {cve_vulnerable_software}\nDescription: {cve_description}"
    full_prompt = PROMPT2 + cve_context

    question, _ = queryGPT(full_prompt, "gpt-5-mini")
    question = question.strip()

    return question


if __name__ == "__main__":
    model_name = "mistral_nemo_cve"
    for i in range(NUM_TO_GENERATE):
        print(f"Generating entry {i + 1}/{NUM_TO_GENERATE}...")
        question = prep_question()
        print(f"Q: {question}")
        answer, rag_usage, cve_data = handle_user_query(question)
        print(f"RAG usage: {rag_usage}") # Confirmed RAG working here
        print(f"A: {answer[:100]}...")
        add_entry(question, answer, model_name, rag_usage, cve_data)
    print(f"Done. {NUM_TO_GENERATE} entries added to the database.")