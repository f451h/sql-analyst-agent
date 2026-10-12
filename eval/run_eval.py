"""End-to-end evaluation: ask the agent 12 questions with known answers and check its final reply.

The expected answers were computed with hand-written SQL. An answer counts as correct when every
expected value (names and numbers) appears in it; numbers are matched with a small tolerance.

    GROQ_API_KEY=... python -m eval.run_eval
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from agent import SQLAgent

ROOT = Path(__file__).resolve().parents[1]


def is_correct(answer: str, expected: list[str]) -> bool:
    numbers = [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", answer)]
    for value in expected:
        try:
            target = float(value)
        except ValueError:
            if value.lower() not in answer.lower():
                return False
            continue
        tol = max(0.011, abs(target) * 0.001)
        if not any(abs(n - target) <= tol for n in numbers):
            return False
    return True


def main():
    load_dotenv()
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    agent = SQLAgent()
    rows, correct = [], 0
    for q in questions:
        start = time.perf_counter()
        result = agent.run(q["question"])
        ok = is_correct(result.answer, q["expected"])
        correct += ok
        rows.append(f"| {q['question']} | {'✅' if ok else '❌'} | {len(result.steps)} | {time.perf_counter() - start:.1f}s |")
        print(("PASS" if ok else "FAIL"), q["question"], "->", result.answer[:120].replace("\n", " "))
    table = "\n".join(["| Question | Correct | Tool calls | Time |", "|---|---|---|---|", *rows])
    summary = f"Model: {agent.model}\n\nAccuracy: {correct}/{len(questions)}\n\n{table}\n"
    (ROOT / "eval" / "results.md").write_text(summary)
    print(f"\n{correct}/{len(questions)} correct. Results written to eval/results.md")


if __name__ == "__main__":
    main()
