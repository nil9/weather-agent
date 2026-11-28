"""Small web server for the forecast map.

  GET  /              -> web/index.html (the map page)
  GET  /api/forecast  -> reports/forecast_<data source>.json (written by `python agent.py`)
  GET  /api/forecast_at?lat=..&lon=..
                      -> forecast for a clicked point, computed in code (no LLM)
  POST /api/chat      -> answer a question about the selected point (chat.py, read-only tools)
  GET  /api/model_comparison -> reports/model_comparison_<data source>.json (compare_models.py)
  GET  /api/reference?lat=..&lon=..
                      -> Open-Meteo model values for the point (real data only), for comparison
  GET  /api/route?from_lat=..&from_lon=..&to_lat=..&to_lon=..&speed_kn=6&draft_m=2&margin_m=1&zones=block
              &land_buffer_m=300&zone_buffer_m=500&passage_land_buffer_m=50&passage_zone_buffer_m=100
              &penalty_factor=10&wind_farms=block|penalty&turbine_radius_m=500
              &weather=show|aware|off&departure=<ISO time, default now>
              &wind_limit_bft=6&waves_limit_m=1.5
                      -> shortest water route (routing.py, code only, no LLM): land, depth, zones, buffers
  GET  /api/route_explain?(same parameters)
                      -> the route plus the detour breakdown (which rule added how many km)
  GET  /api/route_zones -> the restricted zones and wind farms (GeoJSON, OpenStreetMap) for the Route tab
  GET  /api/route_shallow?draft_m=2&margin_m=1
                      -> water shallower than draft + margin (packed grid) for the map layer

Data source: WEATHER_DATA_SOURCE=dummy (default) or real. The app never fetches
observations itself (agent.py / real_data.py do); it only fetches Open-Meteo for the
comparison column, cached in data/real/openmeteo/.

Nothing here writes files, except the Open-Meteo cache (real data only). User input (coordinates, the chat question) is range- or
length-checked; the LLM can only use read-only tools.
Run:  python app.py   then open http://127.0.0.1:8000
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import chat
import clock
import rule_checker
import routing
import tools
import voyage

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


@app.get("/api/clock")
def current_time():
    """The default departure of the Route tab: the real time, or for dummy data the dummy
    calendar ("now" of the synthetic scenario, the same clock as the chat's "today")."""
    return clock.info()


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


class RouteRequest(BaseModel):
    """The route planned in the Route tab (start, destination, settings), sent with chat
    questions so the chat can explain it."""
    from_lat: float = Field(ge=-90, le=90)
    from_lon: float = Field(ge=-180, le=180)
    to_lat: float = Field(ge=-90, le=90)
    to_lon: float = Field(ge=-180, le=180)
    speed_kn: float = Field(routing.DEFAULT_SPEED_KN, ge=1, le=40)
    draft_m: float = Field(routing.DRAFT_M, ge=0.3, le=routing.MAX_DRAFT_M)
    margin_m: float = Field(routing.SAFETY_MARGIN_M, ge=0, le=5)
    zones: Literal["block", "penalty"] = "block"
    land_buffer_m: float = Field(routing.LAND_BUFFER_M, ge=0, le=3000)
    zone_buffer_m: float = Field(routing.ZONE_BUFFER_M, ge=0, le=5000)
    passage_land_buffer_m: float = Field(routing.PASSAGE_LAND_BUFFER_M, ge=0, le=3000)
    passage_zone_buffer_m: float = Field(routing.PASSAGE_ZONE_BUFFER_M, ge=0, le=5000)
    penalty_factor: float = Field(routing.ZONE_PENALTY, ge=0, le=20)
    wind_farms: Literal["block", "penalty"] = "block"
    turbine_radius_m: float = Field(routing.TURBINE_RADIUS_M, ge=routing.TURBINE_RADIUS_RANGE[0],
                                    le=routing.TURBINE_RADIUS_RANGE[1])
    weather: Literal["off", "show", "aware"] = "show"
    departure: datetime | None = None                      # None = now
    wind_limit_bft: int = Field(6, ge=1, le=12)
    waves_limit_m: float = Field(1.5, ge=0.1, le=10)
    gust_limit_bft: int = Field(rule_checker.DEFAULT_GUST_LIMIT_BFT, ge=1, le=12)
    air_draught_m: float = Field(rule_checker.DEFAULT_AIR_DRAUGHT_M, ge=0.5, le=80)
    profile: str | None = Field(None, max_length=40)       # vessel profile (squat Cb, dwt for the rules)
    candidate: Literal["shortest", "balanced", "safest"] | None = None    # chosen in "Compare route candidates"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[ChatMessage] = Field(default_factory=list, max_length=chat.MAX_HISTORY)
    point: Point | None = None          # None = the default sea point
    route: RouteRequest | None = None   # None = no route planned yet


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


def route_query(from_lat: float = Query(ge=-90, le=90), from_lon: float = Query(ge=-180, le=180),
                to_lat: float = Query(ge=-90, le=90), to_lon: float = Query(ge=-180, le=180),
                speed_kn: float = Query(routing.DEFAULT_SPEED_KN, ge=1, le=40),
                draft_m: float = Query(routing.DRAFT_M, ge=0.3, le=routing.MAX_DRAFT_M),
                margin_m: float = Query(routing.SAFETY_MARGIN_M, ge=0, le=5),
                zones: Literal["block", "penalty"] = "block",
                land_buffer_m: float = Query(routing.LAND_BUFFER_M, ge=0, le=3000),
                zone_buffer_m: float = Query(routing.ZONE_BUFFER_M, ge=0, le=5000),
                passage_land_buffer_m: float = Query(routing.PASSAGE_LAND_BUFFER_M, ge=0, le=3000),
                passage_zone_buffer_m: float = Query(routing.PASSAGE_ZONE_BUFFER_M, ge=0, le=5000),
                penalty_factor: float = Query(routing.ZONE_PENALTY, ge=0, le=20),
                wind_farms: Literal["block", "penalty"] = "block",
                turbine_radius_m: float = Query(routing.TURBINE_RADIUS_M, ge=routing.TURBINE_RADIUS_RANGE[0],
                                                le=routing.TURBINE_RADIUS_RANGE[1]),
                weather: Literal["off", "show", "aware"] = "show",
                departure: datetime | None = None,
                wind_limit_bft: int = Query(6, ge=1, le=12),
                waves_limit_m: float = Query(1.5, ge=0.1, le=10),
                gust_limit_bft: int = Query(rule_checker.DEFAULT_GUST_LIMIT_BFT, ge=1, le=12),
                air_draught_m: float = Query(rule_checker.DEFAULT_AIR_DRAUGHT_M, ge=0.5, le=80),
                profile: str | None = Query(None, max_length=40)) -> dict:
    """The route request of the Route tab (points + every setting), shared by the route,
    candidate and passage-plan endpoints."""
    if profile:
        try:
            rule_checker.profile_settings(profile)
        except KeyError as exc:
            raise HTTPException(400, str(exc).strip("'\""))
    q = dict(locals())
    return {"points": [q.pop(k) for k in ("from_lat", "from_lon", "to_lat", "to_lon")], "settings": q}


@app.get("/api/route")
def sea_route(q: dict = Depends(route_query)):
    """Shortest water route between two points, deep enough for the draft, outside (or
    penalising) restricted zones and keeping the safety buffers, plus the rule check against
    rules/passage_rules.yaml ("rule_check"). Legs after the end of the forecast come without
    weather (weather.no_forecast_legs) and rule R-WX-04 warns. Code only.
    A route that isn't possible: 400 with "detail" (the full message) and "problem" (which
    point failed, where, a short reason) for the page's marker and error box."""
    try:
        return _routing_call(voyage.checked_route, *q["points"], q["settings"])
    except HTTPException as exc:
        cause = exc.__cause__ if isinstance(exc.__cause__, routing.RouteError) else None
        if exc.status_code != 400 or cause is None:
            raise
        return JSONResponse(status_code=400, content={"detail": exc.detail, "problem": cause.problem()})


@app.get("/api/route_candidates")
def route_candidates(q: dict = Depends(route_query)):
    """Up to 3 candidate routes (shortest, balanced, safest), each checked against the
    rulebook, then the recommendation (code). Streamed as one JSON object per line, so the
    page can show progress."""
    def lines():
        try:
            for event in voyage.candidates(*q["points"], q["settings"]):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except (tools.LandDataMissing, routing.RoutingDataMissing, routing.RouteError) as exc:
            yield json.dumps({"type": "result", "candidates": [], "error": str(exc)}) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/api/passage_plan")
def passage_plan(kind: Literal["shortest", "balanced", "safest"] = "balanced", q: dict = Depends(route_query)):
    """The passage plan of one candidate: waypoints with ETA, weather and minimum UKC per leg,
    nearest port of refuge per leg, rule-check summary."""
    return _routing_call(voyage.passage_plan, *q["points"], kind, q["settings"])


@app.get("/api/passage_plan.gpx")
def passage_plan_gpx(kind: Literal["shortest", "balanced", "safest"] = "balanced", q: dict = Depends(route_query)):
    """The passage plan as a GPX file (route with waypoints and ETAs). Not for navigation."""
    plan = _routing_call(voyage.passage_plan, *q["points"], kind, q["settings"])
    return Response(voyage.gpx(plan), media_type="application/gpx+xml",
                    headers={"Content-Disposition": f'attachment; filename="passage_plan_{kind}.gpx"'})


@app.get("/api/vessel_profiles")
def vessel_profiles():
    """The vessel profiles (rules/vessel_profiles.yaml) with the settings each one fills."""
    data = rule_checker.load_profiles()
    return {"default": data["default"],
            "profiles": {name: rule_checker.profile_settings(name) for name in data["profiles"]}}


def _routing_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except (tools.LandDataMissing, routing.RoutingDataMissing) as exc:
        raise HTTPException(503, str(exc))
    except routing.RouteError as exc:             # outside the area, no usable water nearby, ...
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/route_explain")
def sea_route_explain(q: dict = Depends(route_query)):
    """The detour breakdown: the same trip with the rules added one by one (code only). Takes
    the same settings as /api/route (a vessel profile adds its squat to the required depth)."""
    return _routing_call(routing.explain, *q["points"], **chat.routing_args(q["settings"], routing.explain))


@app.get("/api/route_shallow")
def sea_route_shallow(draft_m: float = Query(routing.DRAFT_M, ge=0.3, le=routing.MAX_DRAFT_M),
                      margin_m: float = Query(routing.SAFETY_MARGIN_M, ge=0, le=5),
                      profile: str | None = Query(None, max_length=40),
                      speed_kn: float = Query(routing.DEFAULT_SPEED_KN, ge=1, le=40)):
    """Water shallower than draft + squat + margin (squat from the vessel profile, if any)."""
    try:
        squat = voyage.squat_m(profile, speed_kn)
    except KeyError as exc:
        raise HTTPException(400, str(exc).strip("'\""))
    return _routing_call(routing.shallow_mask, draft_m, margin_m, squat_m=squat)


@app.get("/api/route_zones")
def route_zones():
    try:
        return routing.zones_geojson()
    except routing.RoutingDataMissing as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/route_area")
def route_area():
    """The routing area (drawn as a dashed boundary; clicks outside it are refused on the page)."""
    lon0, lat0, lon1, lat1 = routing.ROUTE_BBOX
    return {"lat_min": lat0, "lon_min": lon0, "lat_max": lat1, "lon_max": lon1}


@app.post("/api/chat")
def ask(request: ChatRequest):
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so a slow LLM call
    # doesn't block the map from loading.
    try:
        return chat.answer_question(request.question.strip(),
                                    [m.model_dump() for m in request.history],
                                    request.point.model_dump() if request.point else None,
                                    request.route.model_dump() if request.route else None)
    except tools.LandDataMissing as exc:
        raise HTTPException(503, str(exc))
    except chat.NoForecast as exc:
        raise HTTPException(404, str(exc))
    except tools.PointRejected as exc:            # outside the area, or on land
        raise HTTPException(400, str(exc))
    except chat.ChatUnavailable as exc:
        raise HTTPException(503, f"The local LLM is not reachable (is Ollama running?): {exc}")


@app.on_event("startup")
def warm_up_routing():
    """Build the routing grid and rasters once (about 1 s), so the first route is fast."""
    try:
        routing.warm_up()
    except (tools.LandDataMissing, routing.RoutingDataMissing):
        pass                                      # the Route tab then shows the setup message


if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT} in your browser (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=PORT)
