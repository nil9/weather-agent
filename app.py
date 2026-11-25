"""Small web server for the forecast map.

  GET  /              -> web/index.html (the map page)
  GET  /api/forecast  -> reports/forecast.json (written by `python agent.py`)
  GET  /api/forecast_at?lat=..&lon=..
                      -> forecast for a clicked point, computed in code (no LLM)
  POST /api/chat      -> answer a question about the selected point (chat.py, read-only tools)

Nothing here writes files. User input (coordinates, the chat question) is range- or
length-checked; the LLM can only use read-only tools.
Run:  python app.py   then open http://127.0.0.1:8000
"""
import json
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import chat
import tools

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "web" / "index.html"
FORECAST_JSON = BASE_DIR / "reports" / "forecast.json"

HOST = "127.0.0.1"     # localhost only: other machines on the network can't connect
PORT = 8000

app = FastAPI(title="Marine weather map", docs_url=None, redoc_url=None)


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/forecast")
def forecast():
    if not FORECAST_JSON.exists():
        raise HTTPException(404, "No forecast yet. Run `python agent.py` first.")
    return json.loads(FORECAST_JSON.read_text(encoding="utf-8"))


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class Point(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[ChatMessage] = Field(default_factory=list, max_length=chat.MAX_HISTORY)
    point: Point | None = None          # None = the default sea point


@app.get("/api/forecast_at")
def forecast_at(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Forecast for a clicked point: same inverse-distance logic as the agent, no LLM.
    Station forecasts are cached per agent run, so this is fast after the first call."""
    try:
        return tools.forecast_at(lat, lon)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except tools.OutsideForecastArea as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/chat")
def ask(request: ChatRequest):
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so a slow LLM call
    # doesn't block the map from loading.
    try:
        return chat.answer_question(request.question.strip(),
                                    [m.model_dump() for m in request.history],
                                    request.point.model_dump() if request.point else None)
    except chat.NoForecast as exc:
        raise HTTPException(404, str(exc))
    except tools.OutsideForecastArea as exc:
        raise HTTPException(400, str(exc))
    except chat.ChatUnavailable as exc:
        raise HTTPException(503, f"The local LLM is not reachable (is Ollama running?): {exc}")


if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT} in your browser (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=PORT)
