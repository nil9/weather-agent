"""Marine weather agent.

Step 3c: the full agent loop.
  1. Send the conversation + tool specs to the LLM.
  2. If it asks for tools: run them, append the results as "tool" messages, go to 1.
  3. If it answers with plain text (no tool calls): that's the final answer. Stop.
MAX_STEPS limits the number of LLM calls, so a confused model can't loop forever.

Step 3d: a readable trace prints each tool choice, its arguments and a
shortened result, so you can follow the agent's "thinking" step by step.
"""
import json

from openai import OpenAI

import tools
from tools import TOOL_SPECS

MODEL = "qwen2.5:7b"
MAX_STEPS = 15         # max number of LLM calls in one run
TRACE_WIDTH = 160      # shorten long arguments/results in the trace to this many characters

# Ollama speaks the OpenAI API. The api_key is required by the SDK but ignored by Ollama.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# The LLM only sends a function NAME (a string). This table maps that name
# to the real Python function. Only functions listed here can ever be run.
TOOL_FUNCTIONS = {
    "list_stations": tools.list_stations,
    "get_observations": tools.get_observations,
    "check_data_quality": tools.check_data_quality,
    "forecast_station": tools.forecast_station,
    "estimate_at_sea_point": tools.estimate_at_sea_point,
    "write_report": tools.write_report,
}

SYSTEM_PROMPT = """You are a marine weather forecaster for a point at sea in a bay.
Four coastal stations (NORTH, EAST, SOUTH, WEST) surround the bay.

Rules:
- Use the tools for ALL numbers. Never calculate, estimate or invent numbers yourself.
- Call one tool at a time and wait for its result.
- Check data quality of every station before using its data.
  Do not use stations whose status is SUSPECT.
- Forecast the sea point for the next 3 days with estimate_at_sea_point.
- For trends and warnings, copy `pressure_trend` and `warnings` from estimate_at_sea_point
  exactly. Do not judge trends or warnings from the numbers yourself.
- Finally write a short Markdown report: a summary, a day-by-day table,
  warnings (e.g. strong wind, falling pressure) and which stations were excluded and why.
- Save the report by CALLING the write_report tool with the report as `text`.
  Printing the report is not saving it. Never say it was saved unless write_report
  returned "saved".
- After write_report succeeded, answer with one short sentence and the saved path.
"""

TASK = "Make the 3-day marine forecast for the sea point and save the report."


def run_tool(name: str, arguments_json: str) -> str:
    """Execute one tool call from the LLM and return its JSON result string.

    Never raises: every problem becomes {"error": "..."}, which goes back to the
    LLM as the tool result, so the model can read it and correct itself.
    """
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown tool '{name}'. Available: {list(TOOL_FUNCTIONS)}"})
    try:
        args = json.loads(arguments_json or "{}")   # '{"station": "EAST"}' -> dict
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Arguments are not valid JSON: {exc}"})
    try:
        return TOOL_FUNCTIONS[name](**args)         # e.g. check_data_quality(station="EAST")
    except TypeError as exc:                        # wrong or missing argument names
        return json.dumps({"error": f"Bad arguments for {name}: {exc}"})
    except Exception as exc:                        # anything else: still don't crash
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def shorten(text: str, width: int = TRACE_WIDTH) -> str:
    """One line, at most `width` characters (for the trace only, never for the LLM)."""
    text = " ".join(str(text).split())       # collapse newlines and extra spaces
    return text if len(text) <= width else text[: width - 3] + "..."


def trace_tool_call(step: int, name: str, arguments_json: str, result: str):
    """Print one tool call: which tool, with what, and what came back."""
    is_error = '"error"' in result[:20]
    print(f"\n[step {step}] LLM chose: {name}")
    print(f"   args:   {shorten(arguments_json or '{}')}")
    print(f"   {'ERROR' if is_error else 'result'}: {shorten(result)}")


def run_agent(task: str) -> str:
    """The agent loop. Returns the LLM's final text answer."""
    # The conversation so far. It grows every step; the LLM sees all of it each time.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SPECS,      # the JSON schemas from tools.py: the LLM's "menu"
            temperature=0,         # as repeatable as possible
        )
        message = response.choices[0].message

        # No tool calls -> the LLM is done. Its text is the final answer.
        if not message.tool_calls:
            print(f"\n[step {step}] LLM answered without a tool call -> done")
            return message.content

        # Some models also say something while calling tools. Show it if so.
        if message.content:
            print(f"\n[step {step}] LLM says: {shorten(message.content)}")

        # 1. Remember what the LLM asked for (it must see its own request next time).
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ],
        })

        # 2. Run every requested tool and send each result back, tagged with its call id.
        for call in message.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            trace_tool_call(step, call.function.name, call.function.arguments, result)

    return f"Stopped: reached MAX_STEPS ({MAX_STEPS}) without a final answer."


def main():
    print(f"TASK: {TASK}")
    answer = run_agent(TASK)
    print("\n=== FINAL ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
