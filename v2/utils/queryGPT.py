from openai import OpenAI
from utils.readFile import readFile

"""File to send CVE information to chat-gpt and get MITRE ATT&CK technique labels. Costs $14 per 10.000 requests approx."""


# Read file to get the OpenAI API key
api_key = readFile("keys/openai.key").strip()
client = OpenAI(api_key=api_key)
client.api_key = api_key

AI_MODEL = "gpt-5"

COST_DICT = [
{
    "model": "gpt-5-mini",
    "input_per_1M_tokens": 0.25,
    "output_per_1M_tokens": 2.0
},
{
    "model": "gpt-5",
    "input_per_1M_tokens": 1.25,
    "output_per_1M_tokens": 10.0
}
]

def calculateCost(input_tokens, output_tokens, model: str = AI_MODEL) -> float:
    """
    Calculate the cost of the API call based on token usage.

    Args:
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
    Returns:
        float: The total cost in USD.
    """
    # Cost is in USD per million tokens
    cost_per_1M_input_tokens = next((item["input_per_1M_tokens"] for item in COST_DICT if item["model"] == model), 0.25)
    cost_per_1M_output_tokens = next((item["output_per_1M_tokens"] for item in COST_DICT if item["model"] == model), 2.0)

    total_cost = (input_tokens / 1000000) * cost_per_1M_input_tokens + (output_tokens / 1000000) * cost_per_1M_output_tokens
    return total_cost


def queryGPT(prompt: str, model: str = AI_MODEL):
    """
    Function to send a query to chat-gpt and get a response.

    Args:
        prompt (str): The prompt to send to the model.
        model (str): The AI model to use.

    Returns:
        str: The model's response.
        dict: Token usage and cost information.
    """
    response = client.responses.create(
        model=model,
        input=prompt
    )

    response_text = response.output_text.strip()

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    total_cost = calculateCost(input_tokens, output_tokens, model=model)

    return response_text, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": total_cost
    }


  

if __name__ == "__main__":
    test_prompt = "hello, how are you today."
    response, usage = queryGPT(test_prompt, "gpt-4o")
    print(response)