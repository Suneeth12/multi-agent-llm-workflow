"""Orchestrator: routes a question through the analyst -> executor -> critic loop.

The orchestrator owns the control flow: it asks the analyst for code, runs it,
hands the output to the critic, and either returns the answer or feeds the
critique back to the analyst for another attempt (bounded retries). This is the
core agentic pattern — multiple specialized agents + a loop with error recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import AnalystAgent, CriticAgent, schema_of  # noqa: E402
from llm import get_llm  # noqa: E402
from tools import format_result, run_code  # noqa: E402

DATA = ROOT / "data" / "sales.csv"


class Orchestrator:
    def __init__(self, df: pd.DataFrame, max_attempts: int = 3, verbose=True):
        self.df = df
        self.llm, self.live = get_llm()
        self.analyst = AnalystAgent(self.llm)
        self.critic = CriticAgent(self.llm)
        self.max_attempts = max_attempts
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def answer(self, question: str) -> dict:
        schema = schema_of(self.df)
        feedback, trace = "", []
        for attempt in range(1, self.max_attempts + 1):
            code = self.analyst.write_code(question, schema, feedback)
            run = run_code(code, self.df)
            approved, verdict = self.critic.review(question, code, run)
            trace.append({"attempt": attempt, "code": code,
                          "ok": run["ok"], "verdict": verdict})
            self._log(f"  [attempt {attempt}] analyst: {code}")
            self._log(f"  [attempt {attempt}] critic : {verdict}")
            if approved:
                return {"question": question,
                        "answer": format_result(run["result"]),
                        "attempts": attempt, "trace": trace}
            feedback = verdict
        return {"question": question, "answer": "Could not produce an approved "
                "answer.", "attempts": self.max_attempts, "trace": trace}


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run: python3 src/generate_data.py")
    df = pd.read_csv(DATA)
    orch = Orchestrator(df)
    print(f"LLM backend: {type(orch.llm).__name__}"
          f"{' (live)' if orch.live else ' (offline)'}\n")

    questions = sys.argv[1:] or [
        "What is total revenue by region?",
        "Which product has the highest total profit?",
        "What is the average revenue per order?",
    ]
    for q in questions:
        print(f"Q: {q}")
        res = orch.answer(q)
        print(f"A ({res['attempts']} attempt(s)):\n{res['answer']}\n")


if __name__ == "__main__":
    main()
