"""Tests for the model ranking in compare_models.py (no LLM needed)."""
from compare_models import rank_runs

GOOD_ESTIMATE = {"daily": [], "pressure_trend": {"label": "rising"}, "warnings": [], "stations_used": []}


def run(model, errors=0, steps=11, seconds=20.0, stopped=None, report=True, estimate=GOOD_ESTIMATE):
    return {"model": model, "tool_errors": errors, "steps": steps, "llm_seconds": seconds,
            "stopped": stopped, "report_saved": report, "estimate": estimate, "reminders": 0}


def winner_of(*runs):
    _, winner = rank_runs(list(runs))
    return winner["model"] if winner else None


CASES = [
    ("disqualified never wins, even if much faster",
     [run("fast", seconds=5, stopped="MAX_STEPS"), run("slow", seconds=60)], "slow"),
    ("no accepted report = disqualified",
     [run("a", report=False, errors=0), run("b", errors=3)], "b"),
    ("forecast unavailable = disqualified",
     [run("a", estimate={"unavailable": "all SUSPECT"}), run("b", errors=2)], "b"),
    ("fewer tool errors beats fewer steps",
     [run("a", errors=1, steps=9), run("b", errors=0, steps=14)], "b"),
    ("same errors: fewer steps wins",
     [run("a", errors=1, steps=12), run("b", errors=1, steps=10)], "b"),
    ("same errors and steps: faster wins",
     [run("a", seconds=30.0), run("b", seconds=12.5)], "b"),
    ("all disqualified: no winner",
     [run("a", stopped="LLM timeout"), run("b", report=False)], None),
    ("single valid run wins", [run("only")], "only"),
]

failures = 0
for name, runs, expected in CASES:
    got = winner_of(*runs)
    failures += got != expected
    print(f"{'PASS' if got == expected else 'FAIL'}  {name}: winner={got}")

ranked, _ = rank_runs([run("x", stopped="MAX_STEPS"), run("y", errors=1), run("z")])
ranks = {r["model"]: r["rank"] for r in ranked}
ok = ranks == {"z": 1, "y": 2, "x": None}
failures += not ok
print(f"{'PASS' if ok else 'FAIL'}  ranks: {ranks}")
print(f"\n{'all passed' if not failures else f'{failures} FAILED'}")
