# Architecture

Marine weather agent: four coastal stations around a bay feed a 3-day forecast for one point at sea.
Python tools do every calculation. The local LLM only picks which tools to call and writes the report.

## Progress

| Step | Part | Status |
|------|------|--------|
| 1 | `generate_data.py`: synthetic data in `data/` | ✅ Done |
| 2 | `tools.py`: 6 tools + JSON specs, `test_tools.py` | ✅ Done |
| 3a | `agent.py`: one LLM call, print the raw tool call | ✅ Done |
| 3b | `agent.py`: execute that tool call with `tools.py` | ✅ Done |
| 3c | `agent.py`: full loop, `tool` messages, `MAX_STEPS = 15`, errors go back to the LLM | ✅ Done |
| 3d | `agent.py`: readable trace (tool, args, shortened result) | ✅ Done |
| 3+ | `tools.py`: `estimate_at_sea_point` returns `pressure_trend` + Beaufort `warnings` | ✅ Done |

*Last updated: end of step 3.*

## 1. Components

```mermaid
flowchart LR
    gen["generate_data.py<br/>synthetic data, fixed seed"]
    data[("data/<br/>stations.json<br/>observations.csv")]
    tools["tools.py<br/>6 tools + TOOL_SPECS<br/>(all calculations)"]
    agent["agent.py<br/>chat + tool-calling loop"]
    llm["Ollama LLM<br/>qwen2.5:7b<br/>localhost:11434/v1"]
    reports[("reports/<br/>*.md")]

    gen -->|writes| data
    data -->|read only| tools
    agent -->|"calls tool(args)"| tools
    tools -->|JSON result| agent
    agent -->|"messages + TOOL_SPECS"| llm
    llm -->|"tool_calls or final text"| agent
    tools -->|"write_report (path-checked)"| reports

    classDef done fill:#d4edda,stroke:#28a745,color:#155724
    classDef todo fill:#f1f3f5,stroke:#adb5bd,color:#495057,stroke-dasharray: 5 5
    class gen,data,tools,reports,agent,llm done
```

**Legend:** green = done, grey dashed = not built yet.

Notes:
- The LLM never touches files directly. It writes the report text, and the `write_report` tool saves it.
  So `reports/` is reached only through `tools.py`.
- `tools.py` may only read `data/` and write `reports/`. Paths are checked inside the tools.
- The Ollama model itself isn't part of this repo. It counts as done because `agent.py` now uses it.
- Interpretations such as the pressure trend and wind warnings are worked out in `tools.py`,
  not by the LLM. In an early run the LLM called a rising pressure "falling".

## 2. One agent run (sequence)

```mermaid
sequenceDiagram
    actor U as User
    participant A as agent.py ✅
    participant L as Ollama LLM ✅
    participant T as tools.py ✅
    participant D as data/ ✅
    participant R as reports/ ✅

    U->>A: "Forecast the sea point for 3 days"
    A->>A: messages = [system prompt, user request]

    loop until the LLM replies with no tool_calls (max MAX_STEPS = 15 LLM calls)
        A->>L: chat.completions(messages, tools=TOOL_SPECS)
        L-->>A: assistant message with tool_calls<br/>e.g. check_data_quality("EAST")
        A->>T: run_tool(name, arguments JSON)
        T->>D: read CSV / JSON
        D-->>T: rows
        T-->>A: short JSON string<br/>e.g. {"status": "SUSPECT", ...}<br/>or {"error": "..."} (never a crash)
        A->>A: print trace line, append assistant msg +<br/>tool result (role "tool", tool_call_id)
    end

    Note over L,T: Typical order: list_stations → check_data_quality (each station)<br/>→ get_observations → estimate_at_sea_point (OK stations only) → write_report

    A->>L: chat.completions(messages, tools=TOOL_SPECS)
    L-->>A: tool_call write_report(text)
    A->>T: write_report(text)
    T->>R: save forecast_report.md (inside reports/ only)
    T-->>A: {"saved": "reports/forecast_report.md"}
    A->>L: chat.completions(messages, tools=TOOL_SPECS)
    L-->>A: final answer (plain text, no tool_calls)
    A-->>U: print final answer
```

**Legend:** ✅ done, ⬜ not built yet.

Unknown tool names, broken argument JSON and bad arguments all come back to the LLM as
`{"error": ...}` tool results, so the model can correct itself. If no final answer comes
after `MAX_STEPS` LLM calls, the run stops with a message.

The key idea: the LLM never runs code. It only *asks* for a tool call. `agent.py` runs the real function
and sends the JSON result back as a `tool` message. The loop repeats until the LLM answers with plain text.
