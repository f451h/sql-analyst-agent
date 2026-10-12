"""The tools the agent can call. Each returns a JSON-serialisable dict.

Safety: the database is opened in SQLite read-only mode, and queries are also checked before
running, so the model can't modify data even if it tries.
"""
from __future__ import annotations

import ast
import operator
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "chinook.sqlite"
MAX_ROWS = 200
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b", re.I)


class ToolError(Exception):
    pass


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


def check_sql(query: str) -> str:
    q = query.strip().rstrip(";").strip()
    if not q:
        raise ToolError("Empty query.")
    if ";" in q:
        raise ToolError("Only one statement per call is allowed.")
    if not re.match(r"^(select|with)\b", q, re.I):
        raise ToolError("Only SELECT queries (optionally starting with WITH) are allowed.")
    if FORBIDDEN.search(q):
        raise ToolError("The query contains a keyword that isn't allowed in read-only mode.")
    return q


@dataclass
class Workspace:
    """Holds the DB connection and the last query result (used by make_chart)."""
    conn: sqlite3.Connection = field(default_factory=connect)
    last_columns: list[str] = field(default_factory=list)
    last_rows: list[list] = field(default_factory=list)
    chart: dict | None = None

    # ---- tools -------------------------------------------------------------
    def list_tables(self) -> dict:
        names = [r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {"tables": [{"name": n, "rows": self.conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]}
                           for n in names]}

    def describe_table(self, table: str) -> dict:
        tables = {t["name"].lower(): t["name"] for t in self.list_tables()["tables"]}
        if table.lower() not in tables:
            raise ToolError(f"Unknown table '{table}'. Available: {', '.join(tables.values())}")
        name = tables[table.lower()]
        cols = [{"name": r[1], "type": r[2], "primary_key": bool(r[5])}
                for r in self.conn.execute(f'PRAGMA table_info("{name}")')]
        fks = [{"column": r[3], "references": f"{r[2]}.{r[4]}"}
               for r in self.conn.execute(f'PRAGMA foreign_key_list("{name}")')]
        cur = self.conn.execute(f'SELECT * FROM "{name}" LIMIT 3')
        sample = [list(r) for r in cur.fetchall()]
        return {"table": name, "columns": cols, "foreign_keys": fks, "sample_rows": sample}

    def run_sql(self, query: str) -> dict:
        q = check_sql(query)
        try:
            cur = self.conn.execute(q)
        except sqlite3.Error as e:
            raise ToolError(f"SQL error: {e}") from e
        cols = [d[0] for d in cur.description or []]
        rows = [list(r) for r in cur.fetchmany(MAX_ROWS + 1)]
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        self.last_columns, self.last_rows = cols, rows
        return {"columns": cols, "rows": rows[:50], "row_count": len(rows), "truncated": truncated,
                "note": "Only the first 50 rows are shown to you; the full result is kept for charts." if len(rows) > 50 else ""}

    def calculator(self, expression: str) -> dict:
        return {"expression": expression, "result": safe_eval(expression)}

    def make_chart(self, kind: str, x: str, y: str, title: str = "") -> dict:
        if not self.last_rows:
            raise ToolError("Run a query first; the chart uses the last query result.")
        if x not in self.last_columns or y not in self.last_columns:
            raise ToolError(f"Columns must be from the last result: {self.last_columns}")
        if kind not in {"bar", "line", "pie"}:
            raise ToolError("kind must be bar, line or pie")
        xi, yi = self.last_columns.index(x), self.last_columns.index(y)
        self.chart = {"kind": kind, "x": [r[xi] for r in self.last_rows], "y": [r[yi] for r in self.last_rows],
                      "x_label": x, "y_label": y, "title": title}
        return {"status": "chart created", "points": len(self.last_rows)}

    def call(self, name: str, args: dict) -> dict:
        fn = {"list_tables": self.list_tables, "describe_table": self.describe_table, "run_sql": self.run_sql,
              "calculator": self.calculator, "make_chart": self.make_chart}.get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'"}
        try:
            return fn(**args)
        except ToolError as e:
            return {"error": str(e)}
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}


# ---- safe calculator ---------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
        ast.FloorDiv: operator.floordiv}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without eval(): only numbers, + - * / ** % and round/abs/min/max."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ToolError("Exponent too large.")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError("Only basic arithmetic is allowed.")
    try:
        return ev(ast.parse(expression, mode="eval"))
    except ZeroDivisionError as e:
        raise ToolError("Division by zero.") from e
    except SyntaxError as e:
        raise ToolError("Invalid expression.") from e


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "list_tables", "description": "List all tables in the database with their row counts.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "describe_table", "description": "Get a table's columns, types, foreign keys and 3 sample rows.",
        "parameters": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "run_sql", "description": "Run one read-only SQLite SELECT query and get the result rows.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression, e.g. '(523.06 - 401.62) / 401.62 * 100'.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "make_chart", "description": "Draw a chart from the LAST query result. Use when a visual helps.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["bar", "line", "pie"]},
            "x": {"type": "string", "description": "column for the x-axis / labels"},
            "y": {"type": "string", "description": "numeric column for values"},
            "title": {"type": "string"}}, "required": ["kind", "x", "y"]}}},
]
