"""The code-execution tool the analyst agent calls.

Runs a snippet against the dataframe in a restricted namespace and captures the
`result` variable. This is the "tool use" the analyst agent invokes; in a live
Claude run it would be exposed as a tool definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_code(code: str, df: pd.DataFrame) -> dict:
    """Execute `code` with `df`, `pd`, `np` available; return {result|error}."""
    safe_globals = {"__builtins__": {"round": round, "len": len, "int": int,
                                     "float": float, "min": min, "max": max,
                                     "sum": sum, "sorted": sorted}}
    local = {"df": df, "pd": pd, "np": np, "result": None}
    try:
        exec(code, safe_globals, local)  # noqa: S102 - sandboxed namespace
        result = local.get("result")
        if result is None:
            return {"ok": False, "error": "code did not assign `result`"}
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def format_result(result) -> str:
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return result.to_string()
    return str(result)
