# 🤖 Multi-Agent LLM Workflow

A team of specialized **Claude agents** that collaborate to answer
natural-language questions over a dataset: an **Analyst** writes and runs pandas
code, a **Critic** reviews the output, and an **Orchestrator** runs the loop with
**error recovery**. Natural language in → verified analysis out.

## What this project demonstrates
- **Agent orchestration** — multiple agents with distinct system prompts and
  responsibilities, coordinated by a control loop.
- **Tool use** — the analyst agent calls a code-execution tool that runs its
  pandas snippet against the dataframe and returns the result or the error.
- **Self-correction** — when code fails or the critic rejects the output, the
  critique is fed back and the analyst retries (bounded attempts).
- **Runs offline** — a deterministic MockLLM stands in when no API key is set,
  so the full workflow runs with zero network. Add a real key for live Claude.

## Demo

```text
$ python3 src/orchestrator.py
LLM backend: MockLLM (offline)

Q: What is total revenue by region?
  [attempt 1] analyst: result = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
  [attempt 1] critic : APPROVED
A (1 attempt(s)):
region
North    255567.43
East     202968.39
West     199086.09
South    167449.93
```

**Self-correction loop** (the headline feature):

```text
$ python3 src/orchestrator.py "Give me a statistical summary of the data"

Q: Give me a statistical summary of the data
  [attempt 1] analyst: result = df['sales'].sum()   # wrong column
  [attempt 1] critic : ERROR: KeyError: 'sales'
  [attempt 2] analyst: result = df.describe()
  [attempt 2] critic : APPROVED
A (2 attempt(s)):  <full describe() table>
```

The first attempt references a non-existent column; the executor surfaces the
error, the critic rejects it, and the analyst recovers on attempt 2 — exactly
the agentic retry pattern that makes LLM tool-use robust.

## Architecture

```
            ┌──────────────── Orchestrator ────────────────┐
question ─▶ │  Analyst agent ─▶ code-exec tool ─▶ Critic    │ ─▶ answer
            │        ▲                               │       │
            │        └────────── feedback ◀──────────┘       │
            └───────────────── (retry loop) ────────────────┘
```

- **`src/llm.py`** — Claude client + offline MockLLM, model `claude-sonnet-4-6`.
- **`src/agents.py`** — AnalystAgent (writes code) and CriticAgent (reviews).
- **`src/tools.py`** — sandboxed code-execution tool.
- **`src/orchestrator.py`** — the control loop with bounded retries.

## LLM backends
The agents run on any of three interchangeable backends, selected by env vars —
the orchestration logic is identical:

| Backend | How to enable | Notes |
|---|---|---|
| **Claude** | `export ANTHROPIC_API_KEY=sk-ant-...` | placeholder is `sk-ant-REPLACE_ME` |
| **Local model** | `export USE_LOCAL_LLM=1` (opt. `LOCAL_LLM_MODEL=qwen3:4b`) | runs against a local [Ollama](https://ollama.com) server — no cloud key, no extra Python dependency |
| **MockLLM** | *(default)* | deterministic offline stand-in; always runs |

Selection priority is Claude → local → mock.

```bash
# Drive the agents with a local model (after `ollama pull qwen3:4b`):
USE_LOCAL_LLM=1 LOCAL_LLM_MODEL=qwen3:4b python3 src/orchestrator.py "What is total revenue by region?"
# LLM backend: LocalLLM (live)
```

## Project structure
```
multi-agent-llm-workflow/
├── data/sales.csv          # generated
├── src/
│   ├── generate_data.py    # sample sales table
│   ├── llm.py              # Claude client + offline MockLLM
│   ├── tools.py            # code-execution tool
│   ├── agents.py           # Analyst + Critic agents
│   └── orchestrator.py     # routing + retry loop
├── requirements.txt
├── torun.txt
└── license.md
```

## Run it
```bash
./run.sh        # or see torun.txt
```
