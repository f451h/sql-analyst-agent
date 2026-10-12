import json
from types import SimpleNamespace as NS

import pytest

from agent import SQLAgent, Workspace
from agent.tools import ToolError, check_sql, safe_eval


@pytest.fixture
def ws():
    return Workspace()


# ---------- tools ----------
def test_list_and_describe(ws):
    tables = {t["name"]: t["rows"] for t in ws.list_tables()["tables"]}
    assert tables["Invoice"] == 412 and tables["Track"] == 3503
    info = ws.describe_table("invoice")          # case-insensitive
    assert info["table"] == "Invoice"
    assert any(c["name"] == "Total" for c in info["columns"])
    assert {"column": "CustomerId", "references": "Customer.CustomerId"} in info["foreign_keys"]


def test_unknown_table_returns_error(ws):
    assert "Unknown table" in ws.call("describe_table", {"table": "Orders"})["error"]


def test_run_sql(ws):
    r = ws.run_sql("SELECT BillingCountry, ROUND(SUM(Total), 2) AS revenue FROM Invoice "
                   "GROUP BY BillingCountry ORDER BY revenue DESC LIMIT 3")
    assert r["columns"] == ["BillingCountry", "revenue"]
    assert r["rows"][0] == ["USA", 523.06]


@pytest.mark.parametrize("bad", [
    "DELETE FROM Invoice",
    "DROP TABLE Track",
    "SELECT 1; DROP TABLE Track",
    "UPDATE Customer SET Email='x'",
    "PRAGMA writable_schema=1",
    "WITH x AS (SELECT 1) DELETE FROM Invoice",
])
def test_write_queries_are_blocked(bad):
    with pytest.raises(ToolError):
        check_sql(bad)


def test_database_is_read_only_even_if_check_is_bypassed(ws):
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        ws.conn.execute("DELETE FROM Invoice")


def test_sql_errors_are_returned_not_raised(ws):
    out = ws.call("run_sql", {"query": "SELECT NoSuchColumn FROM Invoice"})
    assert out["error"].startswith("SQL error")


def test_large_results_are_truncated(ws):
    r = ws.run_sql("SELECT TrackId FROM Track")
    assert r["truncated"] and r["row_count"] == 200 and len(r["rows"]) == 50


def test_calculator():
    assert safe_eval("(523.06 - 401.62) / 401.62 * 100") == pytest.approx(30.237, rel=1e-3)
    assert safe_eval("round(2 ** 10 / 3, 2)") == 341.33
    for bad in ["__import__('os').system('ls')", "open('x')", "9 ** 9999", "1/0"]:
        with pytest.raises(ToolError):
            safe_eval(bad)


def test_chart_needs_valid_columns(ws):
    assert "error" in ws.call("make_chart", {"kind": "bar", "x": "a", "y": "b"})
    ws.run_sql("SELECT Name, Milliseconds FROM Track LIMIT 5")
    assert ws.call("make_chart", {"kind": "bar", "x": "Name", "y": "Milliseconds"})["points"] == 5
    assert ws.chart["x_label"] == "Name"


# ---------- agent loop with a scripted fake model ----------
def _tool_msg(name, args, cid):
    return NS(content="", tool_calls=[NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))])


class FakeClient:
    """Plays back a fixed sequence of model replies and records what the agent sends."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs["messages"])
        return NS(choices=[NS(message=self.replies.pop(0))])


def test_agent_runs_tools_and_answers():
    fake = FakeClient([
        _tool_msg("describe_table", {"table": "Invoice"}, "c1"),
        _tool_msg("run_sql", {"query": "SELECT COUNT(*) AS n FROM Invoice"}, "c2"),
        NS(content="There are 412 invoices.", tool_calls=None),
    ])
    result = SQLAgent(client=fake, workspace=Workspace()).run("How many invoices are there?")
    assert result.answer == "There are 412 invoices."
    assert [s.tool for s in result.steps] == ["describe_table", "run_sql"]
    assert result.steps[1].result["rows"] == [[412]]
    # the tool result is passed back to the model on the next turn
    last_turn = fake.sent[-1]
    assert last_turn[-1]["role"] == "tool" and "412" in last_turn[-1]["content"]


def test_agent_recovers_from_sql_error():
    fake = FakeClient([
        _tool_msg("run_sql", {"query": "SELECT Totl FROM Invoice"}, "c1"),
        _tool_msg("run_sql", {"query": "SELECT ROUND(SUM(Total), 2) AS t FROM Invoice"}, "c2"),
        NS(content="Total revenue is 2328.6.", tool_calls=None),
    ])
    result = SQLAgent(client=fake, workspace=Workspace()).run("Total revenue?")
    assert "error" in result.steps[0].result and result.steps[1].result["rows"] == [[2328.6]]


def test_agent_stops_at_step_limit():
    fake = FakeClient([_tool_msg("list_tables", {}, f"c{i}") for i in range(5)])
    result = SQLAgent(client=fake, workspace=Workspace(), max_steps=3).run("loop forever")
    assert result.stopped_early and len(result.steps) == 3


def test_eval_answer_matching():
    from eval.run_eval import is_correct
    assert is_correct("The USA brought in $523.06, the most of any country.", ["USA", "523.06"])
    assert is_correct("Revenue was 1,976 units", ["1976"])
    assert not is_correct("Canada had the most revenue at 303.96", ["USA", "523.06"])
