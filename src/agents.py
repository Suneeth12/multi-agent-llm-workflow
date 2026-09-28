"""Agent definitions: Analyst writes+runs code, Critic reviews the output."""

from __future__ import annotations

import pandas as pd

from llm import extract_code
from tools import format_result, run_code

ANALYST_SYSTEM = (
    "You are a data analyst agent. Given a question and a dataframe schema, "
    "write a single pandas snippet that assigns the answer to a variable named "
    "`result`. The dataframe is available as `df`. Return only code."
)

CRITIC_SYSTEM = (
    "You are a critic agent reviewing a data analysis. Decide whether the "
    "executed result correctly and completely answers the question. Reply "
    "'APPROVED' if good, otherwise 'REVISE: <reason>'."
)


class AnalystAgent:
    def __init__(self, llm):
        self.llm = llm

    def write_code(self, question: str, schema: str, feedback: str = "") -> str:
        prompt = f"Schema:\n{schema}\n\nQuestion: {question}"
        if feedback:
            prompt += f"\n\nA reviewer asked you to revise: {feedback}"
        return extract_code(self.llm.complete(ANALYST_SYSTEM, prompt))


class CriticAgent:
    def __init__(self, llm):
        self.llm = llm

    def review(self, question: str, code: str, run: dict) -> tuple[bool, str]:
        if not run["ok"]:
            return False, f"ERROR: {run['error']}"
        rendered = format_result(run["result"])
        if not rendered.strip():
            return False, "result is empty"
        verdict = self.llm.complete(
            CRITIC_SYSTEM,
            f"Question: {question}\nCode: {code}\nResult:\n{rendered}")
        return verdict.strip().upper().startswith("APPROVED"), verdict.strip()


def schema_of(df: pd.DataFrame) -> str:
    return "columns: " + ", ".join(f"{c}({df[c].dtype})" for c in df.columns)
