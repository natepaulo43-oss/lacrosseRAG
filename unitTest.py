#unit tests
#giving questions with expected answers that we already know the answers for

import pytest
from langchain_ollama import OllamaLLM
from query import query_rag

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
Does the actual response contain the same answer as the expected response?
Answer with only one word: 'true' or 'false'.
"""

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)

    #NOT_FOUND is an exact keyword, so check it in code instead of trusting the LLM judge
    if expected_response == "NOT_FOUND" or "NOT_FOUND" in response_text:
        return expected_response == "NOT_FOUND" and "NOT_FOUND" in response_text

    prompt = EVAL_PROMPT.format(
        expected_response = expected_response,
        actual_response = response_text
    )

    model = OllamaLLM(model="mistral", temperature=0)
    evaluation_results_str = model.invoke(prompt)
    print(f"Judge said: {evaluation_results_str.strip()}")

    #check the first word only, "in" would also match things like "not true"
    final_result = evaluation_results_str.strip().lower().strip("'\".")
    if final_result.startswith("true"):
        return True

    elif final_result.startswith("false"):
        return False
    else:
        raise ValueError("Cannot determine if true or false")


TESTS = [
    {"q": "How long is the shot clock?", "expect": "80 seconds", "page": None},
    {"q": "What are the length limits for a short crosse and a long pole?", "expect": "40-42 in; 52-72 in", "page": None},
    {"q": "How many long poles can a team have on the field?", "expect": "4", "page": None},
    {"q": "What is the penalty for a technical foul?", "expect": "30 seconds or loss of possession", "page": None},
    {"q": "Is body checking allowed for 12U players?", "expect": "NOT_FOUND", "page": None},
]

@pytest.mark.parametrize("test", TESTS)
def test_rules(test):
    assert query_and_validate(
        question = test["q"],
        expected_response = test["expect"]
    )
