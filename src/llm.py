"""LLM client with three interchangeable backends.

Backend is chosen automatically, in priority order:
  1. Claude  — when ANTHROPIC_API_KEY is set to a real value.
  2. Local model — when USE_LOCAL_LLM=1 (or LOCAL_LLM_MODEL is set); talks to a
     local Ollama server (default http://localhost:11434) over plain HTTP, so no
     cloud key and no extra Python dependency are needed.
  3. MockLLM — deterministic offline stand-in so the workflow always runs.

The orchestration logic is identical across all three.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# --- Placeholder. Export the real key for live multi-agent runs: ---
#     export ANTHROPIC_API_KEY="sk-ant-..."
PLACEHOLDER_KEY = "sk-ant-REPLACE_ME"
MODEL = "claude-sonnet-4-6"

# --- Local model defaults (override via env vars) ---
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3")


class MockLLM:
    """Pattern-based stand-in. Plays both the analyst and critic roles."""

    def complete(self, system: str, user: str) -> str:
        role = "critic" if "critic" in system.lower() else "analyst"
        return self._critic(user) if role == "critic" else self._analyst(user)

    def _analyst(self, user: str) -> str:
        # Match on the question only, not the schema/feedback that surrounds it.
        m = re.search(r"question:\s*(.*)", user, re.IGNORECASE)
        q = (m.group(1) if m else user).lower()
        if "region" in q and ("revenue" in q or "sales" in q):
            return "result = df.groupby('region')['revenue'].sum().sort_values(ascending=False)"
        if "product" in q and "profit" in q:
            return "result = df.groupby('product')['profit'].sum().sort_values(ascending=False)"
        if "average order" in q or "average revenue" in q or "avg" in q:
            return "result = round(df['revenue'].mean(), 2)"
        if "total revenue" in q or ("total" in q and "revenue" in q):
            return "result = round(df['revenue'].sum(), 2)"
        if "quantity" in q or "units" in q:
            return "result = int(df['quantity'].sum())"
        # Unknown question: first attempt references a wrong column and errors;
        # once the critic feeds back the error, recover with a safe summary.
        # This demonstrates the orchestrator's retry / self-correction loop.
        if "revise" in user.lower() or "error" in user.lower():
            return "result = df.describe()"
        return "result = df['sales'].sum()  # wrong column name on purpose"

    def _critic(self, user: str) -> str:
        # Approve when an executed result is present and non-empty.
        if "ERROR" in user or "result is empty" in user.lower():
            return "REVISE: the code failed or returned nothing; recompute."
        return "APPROVED"


class ClaudeLLM:
    def __init__(self, key: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=MODEL, max_tokens=500, system=system,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text


class LocalLLM:
    """Local model via an Ollama server (no API key, no extra dependency).

    Uses Ollama's /api/chat endpoint. Start a model first, e.g.:
        ollama pull llama3 && ollama serve
    then run with USE_LOCAL_LLM=1 (optionally LOCAL_LLM_MODEL=mistral, etc.).
    """

    def __init__(self, url: str = LOCAL_LLM_URL, model: str = LOCAL_LLM_MODEL):
        self.url = url.rstrip("/") + "/api/chat"
        self.model = model

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    f"Local model '{self.model}' not found on the Ollama server. "
                    f"Pull it first:  ollama pull {self.model}  "
                    f"(or set LOCAL_LLM_MODEL to an installed model).") from exc
            raise
        return data["message"]["content"]

    def ping(self) -> bool:
        try:
            with urllib.request.urlopen(self.url.replace("/api/chat", "/api/tags"),
                                        timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            return False


def _local_requested() -> bool:
    return (os.environ.get("USE_LOCAL_LLM", "").lower() in {"1", "true", "yes"}
            or "LOCAL_LLM_MODEL" in os.environ)


def get_llm():
    """Return (client, is_live). is_live is True for any real (non-mock) backend."""
    key = os.environ.get("ANTHROPIC_API_KEY", PLACEHOLDER_KEY)
    if key and key != PLACEHOLDER_KEY:
        try:
            return ClaudeLLM(key), True
        except Exception:
            pass
    if _local_requested():
        local = LocalLLM()
        if local.ping():
            return local, True
        print(f"[llm] local model requested but Ollama not reachable at "
              f"{LOCAL_LLM_URL}; falling back to MockLLM.")
    return MockLLM(), False


def extract_code(text: str) -> str:
    """Pull a python snippet out of an LLM reply (handles ``` fences)."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()
