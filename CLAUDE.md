# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A learning project for building an AI agent from scratch: a **marine weather agent**. Four dummy weather stations (NORTH, EAST, SOUTH, WEST) surround a bay, and there is one forecast point in the sea. The agent forecasts the next 3 days at the sea point and writes a report.

The project is built **step by step**, with each step briefly explained. Implement only the step that was asked for, not later ones.

## Design rules (intended architecture)

- **Python tools do all calculations.** The LLM only chooses which tools to call and writes the final report. It must never compute numbers itself. Anything numeric, such as interpolation, trends or fault detection, belongs in a tool.
- **The LLM is local**: Ollama running `qwen2.5:7b`, called through the OpenAI Python SDK (`base_url="http://localhost:11434/v1"`, any dummy `api_key`). The agent loop is a hand-written chat-completions + tool-calling loop.
- **Sandboxed I/O**: tools may only read and write inside `./data` and `./reports`. Validate paths against these directories.
- **No agent frameworks** (no LangChain, LlamaIndex, CrewAI, etc.). Keep the code simple and readable for a learner.

## Data

`generate_data.py` writes synthetic, reproducible data (fixed seed):
- `data/stations.json`: coordinates of the 4 stations and the sea forecast point.
- `data/observations.csv`: 30 days of hourly rows per station: `timestamp, station_id, temperature_c, pressure_hpa, wind_speed_ms, wind_dir_deg, humidity_pct`.

Built-in scenarios that the agent's tools are meant to discover:
- A low-pressure system moves in from the west during the last 2 days.
- Station EAST has a faulty sensor: temperature stuck at one value, and impossible single-hour pressure jumps.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py        # (re)create data/
```
