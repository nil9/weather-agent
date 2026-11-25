"""Tools for the marine weather agent.

Every tool:
  - does ALL the number crunching in Python (the LLM never calculates),
  - returns a SHORT JSON string (a summary, never raw rows),
  - returns {"error": "..."} instead of raising, so the LLM can read the
    problem and try again.

TOOL_SPECS at the bottom describes the tools to the LLM (OpenAI tool format).
"""
import json
import math
import re
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Data-quality thresholds
STUCK_MIN_HOURS = 6          # same value this many hours in a row = stuck sensor
MAX_JUMP = {                 # largest believable change in ONE hour
    "pressure_hpa": 5.0,     # even strong storms rarely exceed ~3 hPa/h
    "temperature_c": 8.0,
    "wind_speed_ms": 15.0,
}
TRAIN_DAYS = 14              # how much history the forecast model learns from
MAX_FORECAST_DAYS = 5        # longest forecast we allow (the agent also checks the task)
TREND_DAMPING = 0.9          # per hour; a trend fades out within ~1 day. If statsmodels
                             # fits this itself it picks ~0.98, and a fast pressure rise
                             # after the low then runs away to >1080 hPa.
PRESSURE_STEADY_HPA = 3.0    # change over the forecast period smaller than this = "steady"
# Beaufort thresholds (m/s) that trigger a marine wind warning, strongest first.
WIND_WARNINGS = [
    (24.5, 10, "STORM"),
    (17.2, 8, "GALE"),
    (13.9, 7, "NEAR GALE"),
    (10.8, 6, "STRONG WIND"),
]
REPORT_FIELDS = ["temp_min_c", "temp_max_c", "pressure_mean_hpa", "wind_max_ms"]

# Memory between tool calls within ONE agent run. estimate_at_sea_point stores its
# result here; write_report checks the report against it. reset_state() clears it.
_last_estimate = None
_last_report_path = None     # set by write_report, used for forecast.json


def reset_state():
    """Forget the last estimate and report (call at the start of every agent run)."""
    global _last_estimate, _last_report_path
    _last_estimate = None
    _last_report_path = None


# ---------------------------------------------------------------------------
# Helpers (not tools)
# ---------------------------------------------------------------------------

def _safe_path(base_dir: Path, filename: str) -> Path:
    """Join base_dir + filename and refuse anything that escapes base_dir.

    resolve() turns '../x', symlinks and absolute paths into a real path,
    so the check below catches every trick to get outside the folder.
    """
    base = base_dir.resolve()
    path = (base / filename).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path '{filename}' is outside the allowed folder '{base.name}/'")
    return path


def _load_stations() -> dict:
    return json.loads(_safe_path(DATA_DIR, "stations.json").read_text())


def _load_observations(station: str) -> pd.DataFrame:
    """All rows of one station, sorted by time, with a datetime index."""
    df = pd.read_csv(_safe_path(DATA_DIR, "observations.csv"), parse_dates=["timestamp"])
    df = df[df["station_id"] == station].sort_values("timestamp")
    return df.set_index("timestamp")


def _check_station(station: str) -> str:
    """Normalise a station name ('north' -> 'NORTH') or raise a helpful error."""
    valid = [s["id"] for s in _load_stations()["stations"]]
    name = str(station).strip().upper()
    if name not in valid:
        raise ValueError(f"Unknown station '{station}'. Valid stations: {valid}")
    return name


def _distance_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle (haversine) distance between two points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _circular_mean_deg(degrees: pd.Series) -> float:
    """Mean of wind directions. A plain mean of 350 and 10 would give 180 (wrong);
    averaging them as unit vectors gives 0 (right)."""
    rad = np.radians(degrees)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def _holt_winters(series: pd.Series, hours: int) -> np.ndarray:
    """Hourly Holt-Winters forecast: level + damped trend + 24 h daily cycle."""
    model = ExponentialSmoothing(
        series.to_numpy(),
        trend="add", damped_trend=True,      # damped: trends fade instead of running away
        seasonal="add", seasonal_periods=24,  # the daily (diurnal) cycle
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # hide statsmodels convergence chatter
        fit = model.fit(damping_trend=TREND_DAMPING)
    return fit.forecast(hours)


def _forecast_station_daily(station: str, days: int) -> list[dict]:
    """Forecast one station and summarise it per day (used by two tools)."""
    df = _load_observations(station)
    train = df[df.index > df.index[-1] - pd.Timedelta(days=TRAIN_DAYS)]

    hours = days * 24
    future_index = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=hours, freq="h")
    fc = pd.DataFrame(index=future_index)
    for col in ["temperature_c", "pressure_hpa", "wind_speed_ms"]:
        fc[col] = _holt_winters(train[col], hours)
    fc["wind_speed_ms"] = fc["wind_speed_ms"].clip(lower=0)  # wind can't be negative

    daily = []
    for date, day in fc.groupby(fc.index.date):
        daily.append({
            "date": str(date),
            "temp_min_c": round(day["temperature_c"].min(), 1),
            "temp_max_c": round(day["temperature_c"].max(), 1),
            "pressure_mean_hpa": round(day["pressure_hpa"].mean(), 1),
            "wind_max_ms": round(day["wind_speed_ms"].max(), 1),
        })
    return daily


def _pressure_trend(first_hpa: float, last_hpa: float) -> dict:
    """Label the pressure change over the forecast as rising / falling / steady."""
    change = round(last_hpa - first_hpa, 1)
    if change >= PRESSURE_STEADY_HPA:
        label = "rising"
    elif change <= -PRESSURE_STEADY_HPA:
        label = "falling"
    else:
        label = "steady"
    return {"label": label, "change_hpa": change, "from_hpa": first_hpa, "to_hpa": last_hpa}


def _wind_warning(wind_max_ms: float) -> str | None:
    """Marine wind warning for a day's max wind, or None if below Beaufort 6."""
    for threshold, beaufort, name in WIND_WARNINGS:
        if wind_max_ms >= threshold:
            return f"{name} (Beaufort {beaufort}, max {wind_max_ms} m/s)"
    return None


AT_STATION_KM = 0.1          # a point this close to a station simply takes that station's forecast


def _idw_weights(stations: list[str], lat: float, lon: float) -> dict[str, float]:
    """Inverse-distance weights (1 / distance^2), normalised to sum to 1.

    Shared by estimate_at_sea_point (agent) and forecast_at (map clicks), so both
    always use exactly the same logic.
    """
    coords = {s["id"]: s for s in _load_stations()["stations"]}
    dist = {s: _distance_km(coords[s]["lat"], coords[s]["lon"], lat, lon) for s in stations}
    nearest = min(dist, key=dist.get)
    if dist[nearest] < AT_STATION_KM:             # on top of a station: avoid dividing by ~0
        return {s: 1.0 if s == nearest else 0.0 for s in stations}
    raw = {s: 1 / d ** 2 for s, d in dist.items()}
    return {s: w / sum(raw.values()) for s, w in raw.items()}


def _combine_daily(forecasts: dict[str, list[dict]], weights: dict[str, float]) -> dict:
    """Weighted daily rows plus ready-made interpretations (warnings, pressure trend),
    so the LLM never has to judge numbers itself."""
    stations = list(forecasts)
    daily = []
    for i in range(len(forecasts[stations[0]])):
        day = {"date": forecasts[stations[0]][i]["date"]}
        for f in REPORT_FIELDS:
            # float(): plain Python numbers, not numpy types (those can't be sent as JSON)
            day[f] = round(float(sum(weights[s] * forecasts[s][i][f] for s in stations)), 1)
        day["wind_warning"] = _wind_warning(day["wind_max_ms"])
        daily.append(day)

    trend = _pressure_trend(daily[0]["pressure_mean_hpa"], daily[-1]["pressure_mean_hpa"])
    warnings = [f"{d['date']}: {d['wind_warning']}" for d in daily if d["wind_warning"]]
    if trend["label"] == "falling":
        warnings.append(f"Falling pressure: {trend['change_hpa']} hPa over the period")
    return {"daily": daily, "pressure_trend": trend, "warnings": warnings}


def _as_json(func):
    """Decorator: turn a tool's dict result (or exception) into a JSON string."""
    def wrapper(*args, **kwargs):
        try:
            return json.dumps(func(*args, **kwargs))
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@_as_json
def list_stations() -> dict:
    """List stations, their coordinates and distance to the sea forecast point."""
    info = _load_stations()
    sea = info["forecast_point"]
    stations = [
        {
            "id": s["id"], "name": s.get("name"), "lat": s["lat"], "lon": s["lon"],
            "distance_to_sea_point_km": round(_distance_km(s["lat"], s["lon"], sea["lat"], sea["lon"]), 1),
        }
        for s in info["stations"]
    ]
    return {"forecast_point": sea, "stations": stations}


@_as_json
def get_observations(station: str, hours: int = 24) -> dict:
    """Summary of the last `hours` of observations for one station."""
    station = _check_station(station)
    hours = int(hours)
    if not 1 <= hours <= 720:
        raise ValueError("hours must be between 1 and 720")
    df = _load_observations(station).tail(hours)

    def stats(col):
        return {"min": round(df[col].min(), 1), "max": round(df[col].max(), 1),
                "mean": round(df[col].mean(), 1), "latest": round(df[col].iloc[-1], 1)}

    p = df["pressure_hpa"]
    return {
        "station": station,
        "period": f"{df.index[0]:%Y-%m-%d %H:%M} to {df.index[-1]:%Y-%m-%d %H:%M} UTC",
        "temperature_c": stats("temperature_c"),
        "pressure_hpa": stats("pressure_hpa"),
        "pressure_change_hpa": round(p.iloc[-1] - p.iloc[0], 1),          # over the whole window
        "pressure_tendency_3h_hpa": round(p.iloc[-1] - p.iloc[-4], 1) if len(p) >= 4 else None,
        "wind_speed_ms": stats("wind_speed_ms"),
        "wind_dir_mean_deg": round(_circular_mean_deg(df["wind_dir_deg"])),
        "humidity_pct": stats("humidity_pct"),
    }


def _quality_issues(station: str) -> list[str]:
    """All data-quality problems of one station; an empty list means OK.

    Not a tool itself: check_data_quality shows it to the LLM, and the forecast
    tools call it internally so they can refuse bad stations on their own.
    """
    df = _load_observations(station)
    issues = []

    # 1. Stuck sensor: the same value repeated for many hours in a row.
    for col in ["temperature_c", "pressure_hpa", "wind_speed_ms"]:
        s = df[col]
        run_id = (s != s.shift()).cumsum()          # new id each time the value changes
        runs = s.groupby(run_id).agg(["size", "first"])
        longest = runs.loc[runs["size"].idxmax()]
        if longest["size"] >= STUCK_MIN_HOURS:
            start = s[run_id == runs["size"].idxmax()].index[0]
            issues.append(f"{col} stuck at {longest['first']} for {int(longest['size'])} h "
                          f"(from {start:%Y-%m-%d %H:%M})")

    # 2. Jumps: hour-to-hour changes larger than physically believable.
    for col, limit in MAX_JUMP.items():
        jumps = df[col].diff().abs()
        bad = jumps[jumps > limit]
        if len(bad):
            issues.append(f"{col}: {len(bad)} single-hour jumps > {limit} "
                          f"(largest {round(bad.max(), 1)})")

    # 3. Gaps: missing hours or empty values.
    expected = pd.date_range(df.index[0], df.index[-1], freq="h")
    missing_hours = len(expected.difference(df.index))
    if missing_hours:
        issues.append(f"{missing_hours} missing hourly timestamps")
    missing_values = int(df.drop(columns="station_id").isna().sum().sum())
    if missing_values:
        issues.append(f"{missing_values} empty values")
    return issues


@_as_json
def check_data_quality(station: str) -> dict:
    """Look for stuck values, impossible jumps and gaps. Returns OK or SUSPECT."""
    station = _check_station(station)
    issues = _quality_issues(station)
    return {"station": station, "status": "SUSPECT" if issues else "OK", "issues": issues}


@_as_json
def forecast_station(station: str, days: int = 3) -> dict:
    """Holt-Winters forecast for one station, summarised per day."""
    station = _check_station(station)
    # Guardrail 2: an unreliable station gets an error, never numbers.
    issues = _quality_issues(station)
    if issues:
        raise ValueError(f"Station {station} is SUSPECT ({'; '.join(issues)}). "
                         "Its data is unreliable, so no forecast is given.")
    days = int(days)
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_FORECAST_DAYS}")
    return {"station": station, "method": "Holt-Winters (damped trend, 24 h cycle)",
            "daily": _forecast_station_daily(station, days)}


@_as_json
def estimate_at_sea_point(stations: list[str], days: int = 3) -> dict:
    """Combine station forecasts into a sea-point forecast (inverse-distance weighting)."""
    if not stations:
        raise ValueError("Give at least one station")
    stations = list(dict.fromkeys(_check_station(s) for s in stations))  # normalise, dedupe

    # Guardrail 1: check quality here, whatever the LLM passed in. SUSPECT stations are dropped.
    excluded = {s: _quality_issues(s) for s in stations}
    excluded = {s: issues for s, issues in excluded.items() if issues}
    stations = [s for s in stations if s not in excluded]

    # Guardrail 2: nothing reliable left -> a clear error, never numbers.
    global _last_estimate
    if not stations:
        reason = f"every requested station is SUSPECT ({', '.join(excluded)})"
        _last_estimate = {"unavailable": reason}   # write_report will expect an "unavailable" report
        raise ValueError(
            f"No reliable station left: {reason}. A sea-point forecast is NOT possible. "
            "Do not invent numbers: write a report saying the forecast is unavailable and why.")
    days = int(days)
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_FORECAST_DAYS}")

    sea = _load_stations()["forecast_point"]
    weights = _idw_weights(stations, sea["lat"], sea["lon"])
    forecasts = {s: _forecast_station_daily(s, days) for s in stations}
    combined = _combine_daily(forecasts, weights)

    result = {"point": sea["id"], "stations_used": stations, "stations_excluded": excluded,
              "weights": {s: round(w, 3) for s, w in weights.items()}, **combined}
    _last_estimate = result      # remembered so write_report can check the report
    return result


def _missing_forecast_numbers(text: str) -> list[str]:
    """Forecast values from the last estimate that do NOT appear in the report text.

    All numbers in the text are compared as floats, so "14" counts as 14.0.
    """
    found = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}
    return [f"{day['date']} {field}={day[field]}"
            for day in _last_estimate["daily"]
            for field in REPORT_FIELDS
            if abs(float(day[field])) not in found]


@_as_json
def write_report(text: str, filename: str = "forecast_report.md") -> dict:
    """Save a Markdown report inside ./reports (and nowhere else).

    Guardrail: the report must match the last estimate_at_sea_point result, so a
    placeholder or invented numbers can't be saved as a "forecast".
    """
    if not filename.endswith(".md"):
        raise ValueError("filename must end with .md")
    path = _safe_path(REPORTS_DIR, filename)   # raises if it points outside reports/
    if _last_estimate is None:
        raise ValueError("No forecast to report yet: call estimate_at_sea_point first. If it "
                         "says no forecast is possible, write a report saying it is unavailable.")
    if "unavailable" in _last_estimate:
        if not re.search(r"unavailable|not possible", text, re.IGNORECASE):
            raise ValueError("The last estimate_at_sea_point said no forecast is possible "
                             f"({_last_estimate['unavailable']}). The report must say the "
                             "forecast is unavailable and why, without forecast numbers.")
    else:
        missing = _missing_forecast_numbers(text)
        if missing:
            raise ValueError("Report rejected: these forecast numbers from estimate_at_sea_point "
                             f"are missing: {', '.join(missing)}. Include the full day-by-day table.")
    REPORTS_DIR.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    global _last_report_path
    _last_report_path = str(path.relative_to(BASE_DIR))
    return {"saved": _last_report_path, "characters": len(text)}


# ---------------------------------------------------------------------------
# Not a tool: called by agent.py after a successful run, never by the LLM.
# ---------------------------------------------------------------------------

def save_forecast_json() -> str:
    """Write reports/forecast.json for the map page, built only from tool results.

    Contains every station (coordinates, quality status, weight), the sea point,
    and the last sea-point forecast. Returns the saved path.
    """
    if _last_estimate is None:
        raise ValueError("No estimate in this run: nothing to save")
    info = _load_stations()
    sea = info["forecast_point"]
    weights = _last_estimate.get("weights", {})

    stations = []
    for s in info["stations"]:
        issues = _quality_issues(s["id"])
        stations.append({
            "id": s["id"], "name": s.get("name"), "lat": s["lat"], "lon": s["lon"],
            "distance_km": round(_distance_km(s["lat"], s["lon"], sea["lat"], sea["lon"]), 1),
            "status": "SUSPECT" if issues else "OK",
            "issues": issues,
            "weight": weights.get(s["id"]),        # None = not used in the forecast
        })

    unavailable = _last_estimate.get("unavailable")
    forecast = None if unavailable else {
        "daily": _last_estimate["daily"],
        "pressure_trend": _last_estimate["pressure_trend"],
        "warnings": _last_estimate["warnings"],
        "stations_used": _last_estimate["stations_used"],
    }
    data = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sea_point": sea,
        "stations": stations,
        "forecast": forecast,                       # None if no forecast was possible
        "unavailable_reason": unavailable,
        "report": _last_report_path,
    }
    path = _safe_path(REPORTS_DIR, "forecast.json")
    REPORTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path.relative_to(BASE_DIR))


# ---------------------------------------------------------------------------
# Tool specs: what the LLM sees (OpenAI chat-completions "tools" format)
# ---------------------------------------------------------------------------

_STATION_PARAM = {"type": "string", "enum": ["NORTH", "EAST", "SOUTH", "WEST"],
                  "description": "Station id"}
_DAYS_PARAM = {"type": "integer", "minimum": 1, "maximum": MAX_FORECAST_DAYS, "default": 3,
               "description": "Number of days to forecast"}


def _spec(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or []},
    }}


TOOL_SPECS = [
    _spec("list_stations",
          "List the weather stations, their coordinates and their distance (km) to the "
          "sea forecast point."),
    _spec("get_observations",
          "Summarise recent observations of one station: min/max/mean/latest for temperature, "
          "pressure, wind and humidity, plus pressure change and 3-hour pressure tendency.",
          {"station": _STATION_PARAM,
           "hours": {"type": "integer", "minimum": 1, "maximum": 720, "default": 24,
                     "description": "How many recent hours to summarise"}},
          ["station"]),
    _spec("check_data_quality",
          "Check one station's sensor data for stuck values, impossible jumps and gaps. "
          "Returns status OK or SUSPECT with a list of issues. Do not trust SUSPECT stations.",
          {"station": _STATION_PARAM}, ["station"]),
    _spec("forecast_station",
          "Forecast one station with Holt-Winters. Returns per-day temperature min/max, "
          "mean pressure and max wind speed. Returns an error for SUSPECT stations.",
          {"station": _STATION_PARAM, "days": _DAYS_PARAM}, ["station"]),
    _spec("estimate_at_sea_point",
          "Forecast the sea point by combining the forecasts of the given stations with "
          "inverse-distance weighting. SUSPECT stations are excluded automatically "
          "(listed in stations_excluded); if none is reliable it returns an error. "
          "Returns per-day values with a wind_warning, the pressure_trend "
          "(rising/falling/steady) and a list of warnings.",
          {"stations": {"type": "array", "items": _STATION_PARAM, "minItems": 1,
                        "description": "Stations to combine"},
           "days": _DAYS_PARAM},
          ["stations"]),
    _spec("write_report",
          "Save the final forecast report as a Markdown file in the reports folder. "
          "The text must contain every daily number from the last estimate_at_sea_point "
          "result (temp min/max, mean pressure, max wind), or it is rejected.",
          {"text": {"type": "string", "description": "The full report in Markdown"},
           "filename": {"type": "string", "default": "forecast_report.md",
                        "description": "File name ending in .md (no folders)"}},
          ["text"]),
]


# ===========================================================================
# Forecast at any point (map clicks) and the chat tools.
#
# Both work on a "forecast data" dict with the same shape as reports/forecast.json:
#   {"sea_point": {...}, "stations": [...], "forecast": {...}, ...}
# The default is forecast.json itself (the agent's sea point). For a clicked point,
# forecast_at() builds the same shape. Nothing here calls the LLM.
# ===========================================================================

MAX_POINT_DISTANCE_KM = 40   # clicks farther than this from every OK station are rejected

# Beaufort scale: lower limit (m/s) of each force 1..12.
BEAUFORT_LIMITS = [0.5, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
SAFE_MAX_BFT = 4             # up to Bft 4 (moderate breeze): safe
CAUTION_MAX_BFT = 6          # Bft 5-6: caution; Bft 7 and up: not safe
VERDICTS = ["safe", "caution", "not safe"]
# What a pressure trend means for the weather (rising pressure = the low is moving away).
PRESSURE_OUTLOOK = {"rising": "improving", "falling": "worsening", "steady": "stable"}


class OutsideForecastArea(ValueError):
    """The clicked point is too far from every OK station."""


def pressure_outlook(trend_label: str) -> str:
    return PRESSURE_OUTLOOK.get(trend_label, "unknown")


def _beaufort(wind_ms: float) -> int:
    """Beaufort force (0-12) for a wind speed in m/s."""
    return int(sum(wind_ms >= limit for limit in BEAUFORT_LIMITS))


def _load_forecast_json() -> dict:
    path = _safe_path(REPORTS_DIR, "forecast.json")
    if not path.exists():
        raise FileNotFoundError("No forecast yet. Run `python agent.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_forecast(data: dict) -> dict:
    if data["forecast"] is None:
        raise ValueError(f"No forecast available: {data.get('unavailable_reason')}")
    return data["forecast"]


# --- Point forecast with a cache per agent run --------------------------------

_station_forecast_cache: dict[str, dict] = {}   # {run id: {station: daily rows}}
_cache_lock = threading.Lock()


def _cached_station_forecasts(run: dict) -> dict[str, list[dict]]:
    """Holt-Winters forecasts of the run's OK stations, computed once per agent run.

    The run id is forecast.json's generated_at: a new `python agent.py` run gives a
    new id, so the cache refills itself. Only the latest run is kept.
    """
    run_id = run["generated_at"]
    with _cache_lock:                    # two first clicks at once: compute only once
        if run_id not in _station_forecast_cache:
            ok = [s["id"] for s in run["stations"] if s["status"] == "OK"]
            days = len(run["forecast"]["daily"])
            _station_forecast_cache.clear()
            _station_forecast_cache[run_id] = {s: _forecast_station_daily(s, days) for s in ok}
        return _station_forecast_cache[run_id]


def forecast_at(lat: float, lon: float) -> dict:
    """Forecast for any point, in code (no LLM), in the same shape as forecast.json.

    Uses only the stations the last agent run marked OK and the same inverse-distance
    weighting as estimate_at_sea_point (_idw_weights + _combine_daily).
    """
    run = _load_forecast_json()
    _require_forecast(run)
    ok = [s for s in run["stations"] if s["status"] == "OK"]
    if not ok:
        raise ValueError("The last agent run has no OK stations.")
    dist = {s["id"]: _distance_km(s["lat"], s["lon"], lat, lon) for s in ok}
    nearest = min(dist, key=dist.get)
    if dist[nearest] > MAX_POINT_DISTANCE_KM:
        raise OutsideForecastArea(
            f"Outside forecast area: the nearest OK station ({nearest}) is "
            f"{dist[nearest]:.1f} km away (max {MAX_POINT_DISTANCE_KM} km).")

    forecasts = _cached_station_forecasts(run)
    weights = _idw_weights(list(forecasts), lat, lon)
    combined = _combine_daily(forecasts, weights)
    stations = [{**s,
                 "distance_km": round(_distance_km(s["lat"], s["lon"], lat, lon), 1),
                 "weight": round(weights[s["id"]], 3) if s["id"] in weights else None}
                for s in run["stations"]]
    return {
        "generated_at": run["generated_at"],
        "sea_point": {"id": "POINT", "name": "Selected point",
                      "lat": round(lat, 4), "lon": round(lon, 4)},
        "stations": stations,
        "forecast": {**combined, "stations_used": list(forecasts)},
        "unavailable_reason": None,
        "report": run.get("report"),
        "selected": True,
    }


def load_forecast_data(point: dict | None = None) -> dict:
    """forecast.json for the default sea point, or forecast_at() for a selected point."""
    return forecast_at(point["lat"], point["lon"]) if point else _load_forecast_json()


# --- Chat tools: pure functions of a forecast data dict ------------------------

def forecast_dates(data: dict | None = None) -> list[str]:
    """Dates covered by the forecast (default: forecast.json); empty if there is none."""
    try:
        forecast = (data or _load_forecast_json())["forecast"]
    except FileNotFoundError:
        return []
    return [d["date"] for d in forecast["daily"]] if forecast else []


def current_pressure_trend(data: dict | None = None) -> str | None:
    """"rising" / "falling" / "steady" for the forecast (default: forecast.json), or None."""
    try:
        forecast = (data or _load_forecast_json())["forecast"]
    except FileNotFoundError:
        return None
    return forecast["pressure_trend"]["label"] if forecast else None


def _get_forecast(data: dict, day: str | None = None) -> dict:
    forecast = _require_forecast(data)
    # Add the Beaufort force (computed here) so the LLM never converts m/s itself.
    daily = [{**d, "wind_max_beaufort": _beaufort(d["wind_max_ms"])} for d in forecast["daily"]]
    if day:
        daily = [d for d in daily if d["date"] == day]
        if not daily:
            raise ValueError(f"Unknown day '{day}'. Forecast days: {forecast_dates(data)}")
    point = data["sea_point"]
    return {"point": point["name"], "lat": point["lat"], "lon": point["lon"], "daily": daily,
            "pressure_trend": forecast["pressure_trend"],
            "pressure_outlook": pressure_outlook(forecast["pressure_trend"]["label"])}


def _get_station_status(data: dict, station: str | None = None) -> dict:
    stations = data["stations"]
    if station:
        name = _check_station(station)
        stations = [s for s in stations if s["id"] == name]
    return {"stations": stations}


def _get_warnings(data: dict) -> dict:
    forecast = _require_forecast(data)
    trend = forecast["pressure_trend"]["label"]
    return {"warnings": forecast["warnings"], "pressure_trend": trend,
            "pressure_outlook": pressure_outlook(trend)}


def _sail_safety(data: dict, day: str) -> dict:
    """Sailing verdict for one forecast day, decided by fixed rules (not by the LLM).

    Bft <= 4: safe, Bft 5-6: caution, Bft >= 7: not safe.
    A falling pressure trend raises the verdict one level.
    """
    forecast = _require_forecast(data)
    days = {d["date"]: d for d in forecast["daily"]}
    if day not in days:
        raise ValueError(f"Unknown day '{day}'. Forecast days: {list(days)}")
    d = days[day]
    bft = _beaufort(d["wind_max_ms"])
    trend = forecast["pressure_trend"]["label"]

    if bft <= SAFE_MAX_BFT:
        level = 0
        reasons = [f"max wind {d['wind_max_ms']} m/s = Beaufort {bft} (up to Bft {SAFE_MAX_BFT} is safe)"]
    elif bft <= CAUTION_MAX_BFT:
        level = 1
        reasons = [f"max wind {d['wind_max_ms']} m/s = Beaufort {bft} (Bft 5-6 means caution)"]
    else:
        level = 2
        reasons = [f"max wind {d['wind_max_ms']} m/s = Beaufort {bft} (Bft 7 or more is not safe)"]
    if trend == "falling":
        reasons.append("pressure is falling, so conditions are worsening")
        if level < 2:
            level += 1
            reasons.append("because of the worsening, the verdict is raised one level")
    elif trend == "rising":
        reasons.append("pressure is rising, so conditions are improving "
                       "(this does not lower the wind-based verdict)")
    else:
        reasons.append("pressure is steady, so conditions are stable")

    return {"date": day, "wind_max_ms": d["wind_max_ms"], "beaufort": bft,
            "pressure_trend": trend, "pressure_outlook": pressure_outlook(trend),
            "verdict": VERDICTS[level], "reasons": reasons}


def make_chat_tools(data: dict) -> dict:
    """The chat's tool table, bound to ONE forecast (one point). Built per request, so
    two users asking about different points at the same time can't mix data."""
    return {
        "get_forecast": _as_json(lambda day=None: _get_forecast(data, day)),
        "get_station_status": _as_json(lambda station=None: _get_station_status(data, station)),
        "get_warnings": _as_json(lambda: _get_warnings(data)),
        "sail_safety": _as_json(lambda day: _sail_safety(data, day)),
    }


# Default-point versions (forecast.json), used by test_tools.py.
@_as_json
def get_forecast(day: str | None = None) -> dict:
    return _get_forecast(_load_forecast_json(), day)


@_as_json
def get_station_status(station: str | None = None) -> dict:
    return _get_station_status(_load_forecast_json(), station)


@_as_json
def get_warnings() -> dict:
    return _get_warnings(_load_forecast_json())


@_as_json
def sail_safety(day: str) -> dict:
    return _sail_safety(_load_forecast_json(), day)


def chat_tool_specs(data: dict | None = None) -> list[dict]:
    """Specs for the chat tools. Built per request, so the `day` enums list the real dates."""
    day_param = {"type": "string", "enum": forecast_dates(data), "description": "Forecast date, YYYY-MM-DD"}
    return [
        _spec("get_forecast",
              "Forecast for the selected point: per day temp min/max, mean pressure, max wind "
              "(m/s and Beaufort) and wind warning, plus the pressure trend and outlook. "
              "Leave out `day` to get all days.",
              {"day": day_param}),
        _spec("get_station_status",
              "Status (OK/SUSPECT), data issues, distance to the selected point and weight of "
              "the stations. Leave out `station` to get all stations.",
              {"station": _STATION_PARAM}),
        _spec("get_warnings", "All warnings for the forecast period at the selected point."),
        _spec("sail_safety",
              "Sailing verdict (safe / caution / not safe) for one day at the selected point, "
              "decided by fixed rules from wind (Beaufort) and pressure trend. "
              "Always use this for safety questions.",
              {"day": day_param},
              ["day"]),
    ]
