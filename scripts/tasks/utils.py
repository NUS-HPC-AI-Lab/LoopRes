"""Grading for the math tasks: the whole reply, thinking included, against the gold answer, through `math_verify`.

lm-evaluation-harness loads this file by path, so it imports nothing from the repository.
"""

from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

def every_reply(replies, docs):
    """The harness's filter keeping each document's every sampled reply, which `process_results` averages over."""
    return replies


def process_results(doc: dict, results: list[list[str]]) -> dict[str, float]:
    """The share of the document's sampled replies, `results[0]`, whose answer verifies against the gold one."""
    answer = doc["answer"].strip()
    answer = answer if answer.startswith("$") else f"${answer}$"
    gold = parse(answer, extraction_config=[LatexExtractionConfig()])
    # The reply's answer: its boxed expressions, which math_verify may merge into a set, else a LaTeX or plain
    # expression.
    extraction = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False, malformed_operators=False, basic_latex=True, boxed="all", units=True
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        ),
        ExprExtractionConfig(),
    ]
    correct = 0
    for reply in results[0]:
        predicted = parse(reply, extraction_config=extraction, extraction_mode="first_match")
        correct += bool(predicted) and verify(gold, predicted)
    return {"math_verify": correct / len(results[0])}
