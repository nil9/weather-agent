"""Run the forecast agent with several models and pick the best run, in code.

Usage:  python compare_models.py [--models qwen2.5:7b qwen2.5:14b]   (default: agent.MODELS)

- The models run ONE AFTER ANOTHER (the GPU has 16 GB; two models at once may not fit).
  Before each run, the previous model is unloaded and the next one is warmed up,
  so loading time doesn't count as LLM time.
- Scoring is plain code (rank_runs), no LLM judge:
    1. any failed guardrail  -> disqualified
    2. then fewer tool errors, then fewer steps, then less LLM time
- The winner's forecast becomes reports/forecast_<data source>.json (what the map shows).
- All runs and the ranking are saved to reports/model_comparison_<data source>.json.
"""
import argparse
import json
import urllib.request

import pandas as pd

import agent
import tools

OLLAMA_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Scoring (pure functions, tested in test_compare.py)
# ---------------------------------------------------------------------------

SCORING_RULE = ("Any failed guardrail = disqualified. Then: fewer tool errors, "
                "then fewer steps, then less LLM time.")


def disqualify_reasons(run: dict) -> list[str]:
    """The guardrails this run failed. Empty list = it passed all of them."""
    reasons = []
    if run.get("stopped"):
        reasons.append(f"stopped: {run['stopped']}")
    if not run.get("report_saved"):
        reasons.append("no report accepted by write_report")
    estimate = run.get("estimate")
    if not estimate:
        reasons.append("no forecast estimate")
    elif "unavailable" in estimate:
        reasons.append("forecast unavailable")
    return reasons


def rank_runs(runs: list[dict]) -> tuple[list[dict], dict | None]:
    """Return (all runs best-first, winner or None). Each run gets
    `disqualified`, `disqualify_reasons` and `rank` (None if disqualified)."""
    for run in runs:
        run["disqualify_reasons"] = disqualify_reasons(run)
        run["disqualified"] = bool(run["disqualify_reasons"])

    def sort_key(run):
        # Python compares tuples left to right: exactly the rule's order.
        return (run["disqualified"], run["tool_errors"], run["steps"], run["llm_seconds"])

    ranked = sorted(runs, key=sort_key)
    position = 0
    for run in ranked:
        run["rank"] = None if run["disqualified"] else (position := position + 1)
    winner = ranked[0] if ranked and not ranked[0]["disqualified"] else None
    return ranked, winner


# ---------------------------------------------------------------------------
# Running the models
# ---------------------------------------------------------------------------

def unload_model(model: str):
    """Ask Ollama to free the model's GPU memory now (keep_alive = 0)."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    request = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=60).read()


def warm_up(model: str):
    """One tiny, untimed call so the model is loaded before the timed run starts."""
    agent.client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Say OK."}],
        max_tokens=2, timeout=300)


def model_slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def run_model(model: str) -> dict:
    """One agent run with `model`. Its report goes to reports/forecast_report_<model>.md."""
    real_write_report = agent.TOOL_FUNCTIONS["write_report"]
    report_file = f"forecast_report_{model_slug(model)}.md"

    def write_report(text: str, filename: str | None = None) -> str:
        # Accept the same arguments as the real tool, but always use this model's file.
        return tools.write_report(text, report_file)

    agent.TOOL_FUNCTIONS["write_report"] = write_report
    try:
        return agent.run_agent_with_stats(agent.TASK, model)
    finally:
        agent.TOOL_FUNCTIONS["write_report"] = real_write_report


def forecast_summary(estimate: dict | None) -> dict | None:
    """A few headline numbers per run (to see if the models agree)."""
    if not estimate or "unavailable" in estimate:
        return None
    return {"stations_used": estimate["stations_used"],
            "wind_max_ms": [d["wind_max_ms"] for d in estimate["daily"]],
            "pressure_trend": estimate["pressure_trend"]["label"],
            "warnings": len(estimate["warnings"])}


def print_table(ranked: list[dict]):
    print("\n| Rank | Model | Guardrails | Tool errors | Steps | LLM time | Reminders |")
    print("|---|---|---|---|---|---|---|")
    for run in ranked:
        rank = "🏆 1" if run["rank"] == 1 else (run["rank"] or "–")
        guard = "✗ " + "; ".join(run["disqualify_reasons"]) if run["disqualified"] else "✓"
        print(f"| {rank} | {run['model']} | {guard} | {run['tool_errors']} | {run['steps']} | "
              f"{run['llm_seconds']:.1f} s | {run['reminders']} |")


def main():
    parser = argparse.ArgumentParser(description="Compare models on the forecast task.")
    parser.add_argument("--models", nargs="+", default=agent.MODELS)
    args = parser.parse_args()

    runs = []
    previous = None
    for model in args.models:
        print(f"\n{'=' * 70}\nMODEL: {model}\n{'=' * 70}")
        if previous:
            unload_model(previous)
        warm_up(model)
        runs.append(run_model(model))
        previous = model
    if previous:
        unload_model(previous)

    ranked, winner = rank_runs(runs)
    print_table(ranked)

    if winner:
        path = tools.save_forecast_json(estimate=winner["estimate"], report=winner["report_path"],
                                        model=winner["model"])
        print(f"\nWinner: {winner['model']} -> {path}")
    else:
        print("\nNo valid run: every model failed a guardrail. reports/{tools.forecast_file()} is unchanged.")

    comparison = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": tools.DATA_SOURCE,
        "task": agent.TASK,
        "temperature": agent.TEMPERATURE,
        "scoring_rule": SCORING_RULE,
        "winner": winner["model"] if winner else None,
        "runs": [{key: run[key] for key in
                  ["model", "rank", "disqualified", "disqualify_reasons", "steps", "tool_calls",
                   "tool_errors", "reminders", "llm_seconds", "stopped", "report_saved",
                   "report_path", "answer"]}
                 | {"forecast": forecast_summary(run["estimate"])}
                 for run in ranked],
    }
    out = tools._safe_path(tools.REPORTS_DIR, tools.comparison_file())
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Comparison saved: {out.relative_to(tools.BASE_DIR)}")


if __name__ == "__main__":
    main()
