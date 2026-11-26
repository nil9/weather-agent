"""Small web server for the forecast map.

  GET  /              -> web/index.html (the map page)
  GET  /api/forecast  -> reports/forecast_<data source>.json (written by `python agent.py`)
  GET  /api/forecast_at?lat=..&lon=..
                      -> forecast for a clicked point, computed in code (no LLM)
  POST /api/chat      -> answer a question about the selected point (chat.py, read-only tools)
  GET  /api/model_comparison -> reports/model_comparison_<data source>.json (compare_models.py)
  GET  /api/reference?lat=..&lon=..
                      -> Open-Meteo model values for the point (real data only), for comparison
  GET  /api/route?from_lat=..&from_lon=..&to_lat=..&to_lon=..&speed_kn=6
                      -> shortest water route (routing.py, code only, no LLM)

Data source: WEATHER_DATA_SOURCE=dummy (default) or real. The app never fetches
observations itself (agent.py / real_data.py do); it only fetches Open-Meteo for the
comparison column, cached in data/real/openmeteo/.

Nothing here writes files, except the Open-Meteo cache (real data only). User input (coordinates, the chat question) is range- or
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
import routing
import tools

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "web" / "index.html"

HOST = "127.0.0.1"     # localhost only: other machines on the network can't connect
PORT = 8000

app = FastAPI(title="Marine weather map", docs_url=None, redoc_url=None)


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


def _with_live_note(data: dict) -> dict:
    """Real data: the note in the forecast file was written at agent time. A refresh may
    have failed since (or the data got old), so show the CURRENT note."""
    if tools.DATA_SOURCE == "real":
        import real_data
        info = real_data.data_info()
        if info["data_time"] == data.get("data_time"):
            data = {**data, "data_note": info["data_note"]}
    return {**data, "attribution": _attribution()}


def _attribution() -> dict:
    import real_data                           # constants only, no network
    return real_data.ATTRIBUTION


@app.get("/api/forecast")
def forecast():
    try:
        return _with_live_note(tools._load_forecast_json())
    except tools.DataSourceMismatch as exc:    # dummy forecast while real data is selected, ...
        raise HTTPException(409, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/reference")
def reference(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Open-Meteo at the point, over the same hours as our forecast. Comparison only:
    the agent and the chat never see it."""
    if tools.DATA_SOURCE != "real":
        return {"available": False, "reason": "Comparison only with real data."}
    import real_data
    try:
        daily = tools._load_forecast_json()["forecast"]["daily"]
    except (FileNotFoundError, TypeError) as exc:      # TypeError: forecast is None
        return {"available": False, "reason": f"No forecast to compare with ({exc})."}
    return real_data.model_reference(lat, lon, daily)


@app.get("/api/model_comparison")
def model_comparison():
    path = tools._safe_path(tools.REPORTS_DIR, tools.comparison_file())
    if not path.exists():
        raise HTTPException(404, "No model comparison yet. Run `python compare_models.py`.")
    return json.loads(path.read_text(encoding="utf-8"))


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
        return _with_live_note(tools.forecast_at(lat, lon))
    except tools.LandDataMissing as exc:          # setup problem, not the user's fault
        raise HTTPException(503, str(exc))
    except tools.DataSourceMismatch as exc:
        raise HTTPException(409, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except tools.PointRejected as exc:            # outside the area, or on land
        raise HTTPException(400, str(exc))


@app.get("/api/route")
def sea_route(from_lat: float = Query(ge=-90, le=90), from_lon: float = Query(ge=-180, le=180),
              to_lat: float = Query(ge=-90, le=90), to_lon: float = Query(ge=-180, le=180),
              speed_kn: float = Query(routing.DEFAULT_SPEED_KN, ge=1, le=40)):
    """Shortest water route between two points. Uses only the coastline, so it works the
    same with dummy and real data."""
    try:
        return routing.route(from_lat, from_lon, to_lat, to_lon, speed_kn)
    except tools.LandDataMissing as exc:
        raise HTTPException(503, str(exc))
    except routing.RouteError as exc:             # outside the area, no water nearby, no connection
        raise HTTPException(400, str(exc))


@app.post("/api/chat")
def ask(request: ChatRequest):
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so a slow LLM call
    # doesn't block the map from loading.
    try:
        return chat.answer_question(request.question.strip(),
                                    [m.model_dump() for m in request.history],
                                    request.point.model_dump() if request.point else None)
    except tools.LandDataMissing as exc:
        raise HTTPException(503, str(exc))
    except chat.NoForecast as exc:
        raise HTTPException(404, str(exc))
    except tools.PointRejected as exc:            # outside the area, or on land
        raise HTTPException(400, str(exc))
    except chat.ChatUnavailable as exc:
        raise HTTPException(503, f"The local LLM is not reachable (is Ollama running?): {exc}")


if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT} in your browser (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=PORT)
