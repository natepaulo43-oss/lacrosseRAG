#unit tests
#giving questions with expected answers that we already know the answers for

import pytest
from langchain_community.llms.ollama import Ollama
from query import query_rag

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response = expected_response,
        actual_response = response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)

    final_result = evaluation_results_str.strip().lower()
    if "true" in final_result:
        return True

    elif "false" in final_result:
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
