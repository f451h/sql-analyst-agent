"""The agent loop: the model decides which tool to call, we run it, feed the result back, repeat."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .tools import TOOL_SPECS, Workspace

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a careful data analyst with read-only access to a SQLite database of a digital music
store (customers, invoices, tracks, albums, artists, genres, employees, playlists).

How to work:
1. Look before you query: use list_tables and describe_table to check table and column names. Never guess
   column names.
2. Write one SQLite SELECT query at a time with run_sql. If it fails, read the error, fix the query and retry.
3. Use calculator for any arithmetic you do yourself (percentages, growth rates). Don't do maths in your head.
4. Use make_chart when the user asks for a chart, or when a ranking or trend over time is easier to see visually.
5. Finish with a short, direct answer in plain English: the key numbers first, then one or two sentences of
   context. Mention any assumptions (e.g. "revenue = sum of invoice totals").

Dates in the database are stored as text like '2021-01-01 00:00:00'; use strftime() to group by year or month."""


@dataclass
class Step:
    tool: str
    args: dict
    result: dict


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    chart: dict | None = None
    stopped_early: bool = False


class SQLAgent:
    def __init__(self, client=None, model: str | None = None, workspace: Workspace | None = None, max_steps: int = 10):
        if client is None:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY is not set")
            client = Groq(api_key=key)
        self.client = client
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.ws = workspace or Workspace()
        self.max_steps = max_steps

    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        self.ws.chart = None
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or [])[-6:],
                    {"role": "user", "content": question}]
        steps: list[Step] = []
        for _ in range(self.max_steps):
            resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOL_SPECS,
                                                       tool_choice="auto", temperature=0)
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                return AgentResult(answer=msg.content or "", steps=steps, chart=self.ws.chart)
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args, result = {}, {"error": "Arguments were not valid JSON."}
                else:
                    result = self.ws.call(c.function.name, args)
                steps.append(Step(c.function.name, args, result))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result, default=str)})
        return AgentResult(answer="I hit the step limit before finishing. Try asking a narrower question.",
                           steps=steps, chart=self.ws.chart, stopped_early=True)
