"""Tools for the marine weather agent.

Every tool:
  - does ALL the number crunching in Python (the LLM never calculates),
  - returns a SHORT JSON string (a summary, never raw rows),
  - returns {"error": "..."} instead of raising, so the LLM can read the
    problem and try again.

TOOL_SPECS at the bottom describes the tools to the LLM (OpenAI tool format).
"""
import json
import functools
import math
import os
import re
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
import shapely
from shapely.ops import unary_union
from shapely.prepared import prep
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Where the observations come from: "dummy" (generate_data.py, the default) or "real"
# (DWD stations via Bright Sky, fetched by real_data.py). Each has its own folder, and
# Each also has its own forecast file (forecast_dummy.json / forecast_real.json), which
# records the source it was made from, so the two are never mixed.
DATA_SOURCE = os.environ.get("WEATHER_DATA_SOURCE", "dummy")
DATA_SOURCES = {"dummy": "", "real": "real/current/"}      # folder inside data/
if DATA_SOURCE not in DATA_SOURCES:
    raise SystemExit(f"WEATHER_DATA_SOURCE must be one of {list(DATA_SOURCES)}, not {DATA_SOURCE!r}")


REPORT_FOLDERS = {"dummy": "", "real": "real/"}           # report folder inside reports/


def forecast_file() -> str:
    """The map/chat forecast of the current data source, inside reports/."""
    return f"forecast_{DATA_SOURCE}.json"


def comparison_file() -> str:
    """compare_models.py's ranking for the current data source, inside reports/."""
    return f"model_comparison_{DATA_SOURCE}.json"

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


def _data_file(name: str) -> Path:
    """A data file of the CURRENT data source (data/ for dummy, data/real/current/ for real).
    This is the only place that decides which data the tools read."""
    return _safe_path(DATA_DIR, DATA_SOURCES[DATA_SOURCE] + name)


def _load_stations() -> dict:
    return json.loads(_data_file("stations.json").read_text(encoding="utf-8"))


def _station_coords() -> dict[str, dict]:
    """id -> station for the agent's stations plus (real data) the wider pool."""
    info = _load_stations()
    return {s["id"]: s for s in info.get("pool", []) + info["stations"]}


def _load_observations(station: str) -> pd.DataFrame:
    """All rows of one station, sorted by time, with a datetime index."""
    df = pd.read_csv(_data_file("observations.csv"), parse_dates=["timestamp"])
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


def _forecast_station_hourly(station: str, days: int) -> pd.DataFrame:
    """Hourly Holt-Winters forecast of one station (temperature, pressure, wind), cached per
    data source, station, length and observation file. Used by the daily summary below and by
    route_weather.py (wind along a route in dummy mode)."""
    path = _data_file("observations.csv")
    return _forecast_station_hourly_cached(DATA_SOURCE, station, int(days), path.stat().st_mtime).copy()


@functools.lru_cache(maxsize=64)
def _forecast_station_hourly_cached(source: str, station: str, days: int, mtime: float) -> pd.DataFrame:
    df = _load_observations(station)
    train = df[df.index > df.index[-1] - pd.Timedelta(days=TRAIN_DAYS)]

    # The forecast covers `days` calendar days (UTC), starting with the day of the first
    # forecast hour. Dummy data ends at 23:00, so that's 3 full days. Real data ends at the
    # latest hour, so the first day is only the rest of today (marked with from_utc).
    first = df.index[-1] + pd.Timedelta(hours=1)
    hours = (24 - first.hour) + (days - 1) * 24
    future_index = pd.date_range(first, periods=hours, freq="h")
    fc = pd.DataFrame(index=future_index)
    for col in ["temperature_c", "pressure_hpa", "wind_speed_ms"]:
        fc[col] = _holt_winters(train[col], hours)
    fc["wind_speed_ms"] = fc["wind_speed_ms"].clip(lower=0)  # wind can't be negative
    return fc


def _forecast_station_daily(station: str, days: int) -> list[dict]:
    """Forecast one station and summarise it per day (used by two tools)."""
    fc = _forecast_station_hourly(station, days)
    daily = []
    for date, day in fc.groupby(fc.index.date):
        row = {"date": str(date)}
        if day.index[0].hour != 0:
            row["from_utc"] = f"{day.index[0]:%H:%M}"      # partial first day
        row.update({
            "temp_min_c": round(day["temperature_c"].min(), 1),
            "temp_max_c": round(day["temperature_c"].max(), 1),
            "pressure_mean_hpa": round(day["pressure_hpa"].mean(), 1),
            "wind_max_ms": round(day["wind_speed_ms"].max(), 1),
        })
        daily.append(row)
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
    coords = _station_coords()
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
        first = forecasts[stations[0]][i]
        day = {"date": first["date"]}
        if "from_utc" in first:
            day["from_utc"] = first["from_utc"]
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
    Real data: inside ./reports/real/, so a real report never overwrites a dummy one.

    Guardrail: the report must match the last estimate_at_sea_point result, so a
    placeholder or invented numbers can't be saved as a "forecast".
    """
    if not filename.endswith(".md"):
        raise ValueError("filename must end with .md")
    path = _safe_path(REPORTS_DIR, REPORT_FOLDERS[DATA_SOURCE] + filename)   # raises if outside reports/
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    global _last_report_path
    _last_report_path = str(path.relative_to(BASE_DIR))
    return {"saved": _last_report_path, "characters": len(text)}


# ---------------------------------------------------------------------------
# Not a tool: called by agent.py after a successful run, never by the LLM.
# ---------------------------------------------------------------------------

def save_forecast_json(estimate: dict | None = None, report: str | None = None,
                       model: str | None = None) -> str:
    """Write reports/forecast_<data source>.json for the map page, built only from tool results.

    Contains every station (coordinates, quality status, weight), the sea point,
    and the sea-point forecast. By default it uses the last estimate of this process;
    compare_models.py passes the WINNING run's snapshot (`estimate`, `report`, `model`).
    Returns the saved path.
    """
    estimate = estimate if estimate is not None else _last_estimate
    report = report if report is not None else _last_report_path
    if estimate is None:
        raise ValueError("No estimate in this run: nothing to save")
    info = _load_stations()
    sea = info["forecast_point"]
    weights = estimate.get("weights", {})

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

    unavailable = estimate.get("unavailable")
    forecast = None if unavailable else {
        "daily": estimate["daily"],
        "pressure_trend": estimate["pressure_trend"],
        "warnings": estimate["warnings"],
        "stations_used": estimate["stations_used"],
    }
    data = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sea_point": sea,
        "stations": stations,
        "forecast": forecast,                       # None if no forecast was possible
        "unavailable_reason": unavailable,
        "report": report,
        "model": model,                             # which LLM ran the agent (None if unknown)
        **data_source_info(),                       # data_source, data_time, data_note
    }
    path = _safe_path(REPORTS_DIR, forecast_file())
    REPORTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path.relative_to(BASE_DIR))


def data_source_info() -> dict:
    """Which data a forecast is made from, for forecast.json and the page.
    Dummy: the newest observation time. Real: the same, plus a note if the data is old
    or the last refresh failed (from real_data.py)."""
    if DATA_SOURCE == "real":
        import real_data                            # only needed for real data
        info = real_data.data_info()
        return {k: info[k] for k in ["data_source", "data_time", "data_note"]}
    newest = pd.read_csv(_data_file("observations.csv"), usecols=["timestamp"])["timestamp"].max()
    return {"data_source": "dummy", "data_time": newest, "data_note": None}


# ---------------------------------------------------------------------------
# Tool specs: what the LLM sees (OpenAI chat-completions "tools" format)
# ---------------------------------------------------------------------------

NUMBER_WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]


def station_ids() -> list[str]:
    """IDs of the agent's stations for the current data source ([] if there is no data yet)."""
    try:
        return [s["id"] for s in _load_stations()["stations"]]
    except FileNotFoundError:
        return []


def stations_sentence() -> str:
    """The stations line of the agent's system prompt. For the dummy data this is exactly
    "Four coastal stations (NORTH, EAST, SOUTH, WEST) surround the bay." """
    ids = station_ids()
    count = NUMBER_WORDS[len(ids)] if len(ids) < len(NUMBER_WORDS) else str(len(ids))
    return f"{count} coastal stations ({', '.join(ids)}) surround the bay."


def _station_param(ids: list[str] | None = None) -> dict:
    """Station argument for the specs. The enum lists the stations of the current data."""
    ids = station_ids() if ids is None else ids
    return {"type": "string", **({"enum": ids} if ids else {}), "description": "Station id"}
_DAYS_PARAM = {"type": "integer", "minimum": 1, "maximum": MAX_FORECAST_DAYS, "default": 3,
               "description": "Number of days to forecast"}


def _spec(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or []},
    }}


def tool_specs() -> list[dict]:
    """The agent's tool specs, built from the current station list (the agent calls this
    at the start of every run). For the dummy data the enum is NORTH/EAST/SOUTH/WEST."""
    _STATION_PARAM = _station_param()
    return [
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


TOOL_SPECS = tool_specs()      # built at import (test_tools.py prints it)


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


class PointRejected(ValueError):
    """A clicked point we can't forecast for (app.py turns this into HTTP 400)."""


class OutsideForecastArea(PointRejected):
    """The clicked point is too far from every OK station."""


class PointOnLand(PointRejected):
    """The clicked point is on land; this is a marine forecast."""


class LandDataMissing(FileNotFoundError):
    """data/land_kiel_bay.geojson is missing (run download_land.py)."""


# --- Land/sea check (Natural Earth 1:10m land, see download_land.py) -------------

LAND_FILE = "land_kiel_bay.geojson"
COAST_TOLERANCE_KM = 0.5     # points this close to the coast count as water (polygons are coarse)
_KM_PER_DEG_LAT = 110.57
_KM_PER_DEG_LON = 111.32 * math.cos(math.radians(54.5))   # Kiel Bay latitude


def _to_km(lon: float, lat: float) -> tuple[float, float]:
    """Simple local km grid for Kiel Bay. Degrees are not equal in size (1 deg of longitude
    is only ~65 km here), so distances like "500 m" must be measured in km, not degrees.
    Within ~100 km the error is below 1 %."""
    return lon * _KM_PER_DEG_LON, lat * _KM_PER_DEG_LAT


@functools.lru_cache(maxsize=1)
def _land_km():
    """All land polygons (unshrunk) in the local km grid. Loaded once. Used by the
    land/sea check below and by routing.py (which must never touch land)."""
    path = _safe_path(DATA_DIR, LAND_FILE)
    if not path.exists():
        raise LandDataMissing("Land data missing: run `python download_land.py` once.")
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    land = unary_union([shape(f["geometry"]) for f in features])
    # degrees -> local km grid (same scaling as _to_km, applied to all coordinates at once)
    return shapely.transform(land, lambda xy: xy * [_KM_PER_DEG_LON, _KM_PER_DEG_LAT])


@functools.lru_cache(maxsize=1)
def _inland_area():
    """Land shrunk by COAST_TOLERANCE_KM, prepared for fast lookups. Loaded once."""
    return prep(_land_km().buffer(-COAST_TOLERANCE_KM))   # negative buffer = shrink


@functools.lru_cache(maxsize=4096)
def _is_on_land_rounded(lat: float, lon: float) -> bool:
    return _inland_area().contains(Point(*_to_km(lon, lat)))


def is_on_land(lat: float, lon: float) -> bool:
    """True if the point is on land, more than COAST_TOLERANCE_KM from the coast.
    Cached per point (rounded to 4 decimals, about 10 m)."""
    return _is_on_land_rounded(round(lat, 4), round(lon, 4))


def pressure_outlook(trend_label: str) -> str:
    return PRESSURE_OUTLOOK.get(trend_label, "unknown")


def _beaufort(wind_ms: float) -> int:
    """Beaufort force (0-12) for a wind speed in m/s."""
    return int(sum(wind_ms >= limit for limit in BEAUFORT_LIMITS))


class DataSourceMismatch(FileNotFoundError):
    """forecast.json was made from the other data source (dummy vs real).
    A FileNotFoundError, so every caller treats it like "no usable forecast"."""


def _load_forecast_json() -> dict:
    path = _safe_path(REPORTS_DIR, forecast_file())
    if not path.exists():
        hint = "python agent.py" if DATA_SOURCE == "dummy" else "WEATHER_DATA_SOURCE=real python agent.py"
        raise FileNotFoundError(f"No {DATA_SOURCE} forecast yet ({path.name}). Run `{hint}` first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    made_from = data.get("data_source", "dummy")       # older files: always dummy
    # Each source has its own file, so this only trips on a copied or edited file.
    if made_from != DATA_SOURCE:
        raise DataSourceMismatch(
            f"The forecast was made from {made_from} data, but {DATA_SOURCE} data is selected. "
            f"Run `WEATHER_DATA_SOURCE={DATA_SOURCE} python agent.py` first.")
    return data


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


POINT_STATIONS = 4           # real data: a clicked point uses its 4 nearest pool stations
_pool_cache: dict[str, dict] = {}   # real data: {run id: {(kind, station): result}}


def _cached_pool(run: dict, kind: str, station: str, compute):
    """Real data: quality check / forecast of one pool station, computed once per agent run."""
    run_id = run["generated_at"]
    with _cache_lock:
        if run_id not in _pool_cache:
            _pool_cache.clear()
            _pool_cache[run_id] = {}
        bucket = _pool_cache[run_id]
        if (kind, station) not in bucket:
            bucket[(kind, station)] = compute()
        return bucket[(kind, station)]


def _real_point_stations(run: dict, lat: float, lon: float) -> list[dict]:
    """Real data: the POINT_STATIONS nearest pool stations to (lat, lon), each checked
    with the unchanged _quality_issues (in code, no LLM)."""
    if run.get("data_time") != data_source_info()["data_time"]:
        raise DataSourceMismatch(
            "The real data was refreshed after the last agent run, so the forecast and the "
            "stations would not match. Run `WEATHER_DATA_SOURCE=real python agent.py` again.")
    pool = sorted(_station_coords().values(),
                  key=lambda s: _distance_km(s["lat"], s["lon"], lat, lon))[:POINT_STATIONS]
    stations = []
    for s in pool:
        issues = _cached_pool(run, "quality", s["id"], lambda sid=s["id"]: _quality_issues(sid))
        stations.append({"id": s["id"], "name": s.get("name"), "lat": s["lat"], "lon": s["lon"],
                         "status": "SUSPECT" if issues else "OK", "issues": issues})
    return stations


def forecast_at(lat: float, lon: float) -> dict:
    """Forecast for any point, in code (no LLM), in the same shape as forecast.json.

    Dummy data: uses only the stations the last agent run marked OK.
    Real data: uses the 4 nearest pool stations to the point that pass the quality check.
    Both use the same inverse-distance weighting as estimate_at_sea_point
    (_idw_weights + _combine_daily).
    """
    run = _load_forecast_json()
    _require_forecast(run)
    all_stations = _real_point_stations(run, lat, lon) if DATA_SOURCE == "real" else run["stations"]
    ok = [s for s in all_stations if s["status"] == "OK"]
    if not ok:
        if DATA_SOURCE == "real":
            raise PointRejected("No reliable station near this point (all nearby stations are SUSPECT).")
        raise ValueError("The last agent run has no OK stations.")
    dist = {s["id"]: _distance_km(s["lat"], s["lon"], lat, lon) for s in ok}
    nearest = min(dist, key=dist.get)
    if dist[nearest] > MAX_POINT_DISTANCE_KM:
        raise OutsideForecastArea(
            f"Outside forecast area: the nearest OK station ({nearest}) is "
            f"{dist[nearest]:.1f} km away (max {MAX_POINT_DISTANCE_KM} km).")
    if is_on_land(lat, lon):
        raise PointOnLand("This point is on land. Marine forecast only for water.")

    if DATA_SOURCE == "real":
        days = len(run["forecast"]["daily"])
        forecasts = {s["id"]: _cached_pool(run, "forecast", s["id"],
                                           lambda sid=s["id"]: _forecast_station_daily(sid, days))
                     for s in ok}
    else:
        forecasts = _cached_station_forecasts(run)
    weights = _idw_weights(list(forecasts), lat, lon)
    combined = _combine_daily(forecasts, weights)
    stations = [{**s,
                 "distance_km": round(_distance_km(s["lat"], s["lon"], lat, lon), 1),
                 "weight": round(weights[s["id"]], 3) if s["id"] in weights else None}
                for s in all_stations]
    return {
        **{k: run.get(k) for k in ["data_source", "data_time", "data_note", "model"]},
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


def _get_forecast(data: dict, day: str | list[str] | None = None) -> dict:
    """Daily forecast for one date, a list of dates, or (day=None) all days."""
    forecast = _require_forecast(data)
    # Add the Beaufort force (computed here) so the LLM never converts m/s itself.
    daily = [{**d, "wind_max_beaufort": _beaufort(d["wind_max_ms"])} for d in forecast["daily"]]
    if day:
        wanted = [day] if isinstance(day, str) else list(day)   # models also send lists
        unknown = [d for d in wanted if d not in {row["date"] for row in daily}]
        if unknown:
            raise ValueError(f"Unknown day(s) {unknown}. Forecast days: {forecast_dates(data)}")
        daily = [row for row in daily if row["date"] in wanted]
    point = data["sea_point"]
    extra = {}
    if data.get("data_source") == "real":
        # Real data: the Open-Meteo model values for the same hours, next to our estimate.
        reference = _reference_for(data)
        model = {r["date"]: r for r in reference.get("daily", []) if r.get("available", True)} \
            if reference.get("available") else {}
        keep = ["temp_min_c", "temp_max_c", "pressure_mean_hpa", "wind_max_ms"]
        daily = [{**row, "open_meteo": ({**{k: model[row["date"]][k] for k in keep},
                                         "wind_max_beaufort": _beaufort(model[row["date"]]["wind_max_ms"])}
                                        if row["date"] in model else None)} for row in daily]
        extra = {"values_are": "our estimate from the stations; open_meteo = the Open-Meteo model",
                 **({} if reference.get("available") else
                    {"open_meteo_note": reference.get("reason", "Open-Meteo not available")})}
    return {"point": point["name"], "lat": point["lat"], "lon": point["lon"], "daily": daily, **extra,
            "pressure_trend": forecast["pressure_trend"],
            "pressure_outlook": pressure_outlook(forecast["pressure_trend"]["label"])}


def _get_station_status(data: dict, station: str | None = None) -> dict:
    stations = data["stations"]
    if station:
        name = str(station).strip().upper()
        valid = [s["id"] for s in stations]       # the stations of THIS point
        if name not in valid:
            raise ValueError(f"Unknown station '{station}'. Valid stations: {valid}")
        stations = [s for s in stations if s["id"] == name]
    return {"stations": stations}


def _get_warnings(data: dict) -> dict:
    forecast = _require_forecast(data)
    trend = forecast["pressure_trend"]["label"]
    return {"warnings": forecast["warnings"], "pressure_trend": trend,
            "pressure_outlook": pressure_outlook(trend)}


WIND_SOURCE_OURS = "our estimate"            # the 4-station forecast
WIND_SOURCE_MODEL = "Open-Meteo"              # the model forecast (real data only)
WIND_SOURCE_BOTH = "both (equal)"


def _reference_for(data: dict) -> dict:
    """Real data: the Open-Meteo values for this forecast's point, fetched (or read from
    the 60 min cache) the first time sail_safety needs them, then kept in `data`."""
    if "reference" not in data:
        import real_data
        point = data["sea_point"]
        data["reference"] = real_data.model_reference(point["lat"], point["lon"],
                                                      data["forecast"]["daily"])
    return data["reference"]


def _safety_wind(data: dict, d: dict) -> tuple[float, dict, str | None]:
    """The wind the verdict is based on.

    Dummy data: our estimate, as before (no extra fields).
    Real data: the HIGHER of our estimate and Open-Meteo for that day (the safer choice),
    plus which source gave it. If Open-Meteo is not available: our estimate, and it says so.
    Returns (wind, extra result fields, extra reason or None).
    """
    ours = d["wind_max_ms"]
    if data.get("data_source") != "real":
        return ours, {}, None
    reference = _reference_for(data)
    model_day = next((r for r in reference.get("daily", []) if r["date"] == d["date"]
                      and r.get("available", True)), None) if reference.get("available") else None
    model = model_day["wind_max_ms"] if model_day else None
    if model is None:
        return ours, {"wind_our_estimate_ms": ours, "wind_open_meteo_ms": None,
                      "wind_source": WIND_SOURCE_OURS}, \
            "Open-Meteo is not available, so the verdict uses our estimate only"
    if model > ours:
        wind, source = model, WIND_SOURCE_MODEL
    elif ours > model:
        wind, source = ours, WIND_SOURCE_OURS
    else:
        wind, source = ours, WIND_SOURCE_BOTH
    reason = (f"the higher max wind of our estimate ({ours} m/s) and the Open-Meteo model "
              f"({model} m/s) is used: {wind} m/s from {source}")
    return wind, {"wind_our_estimate_ms": ours, "wind_open_meteo_ms": model,
                  "wind_source": source}, reason


def _sail_safety(data: dict, day: str) -> dict:
    """Sailing verdict for one forecast day, decided by fixed rules (not by the LLM).

    Bft <= 4: safe, Bft 5-6: caution, Bft >= 7: not safe.
    A falling pressure trend raises the verdict one level.
    Real data: the wind is the higher of our estimate and Open-Meteo (_safety_wind).
    """
    forecast = _require_forecast(data)
    days = {d["date"]: d for d in forecast["daily"]}
    if day not in days:
        raise ValueError(f"Unknown day '{day}'. Forecast days: {list(days)}")
    d = days[day]
    wind, wind_fields, wind_reason = _safety_wind(data, d)
    bft = _beaufort(wind)
    trend = forecast["pressure_trend"]["label"]

    if bft <= SAFE_MAX_BFT:
        level = 0
        reasons = [f"max wind {wind} m/s = Beaufort {bft}; safe means Beaufort {SAFE_MAX_BFT} or less"]
    elif bft <= CAUTION_MAX_BFT:
        level = 1
        reasons = [f"max wind {wind} m/s = Beaufort {bft}; caution applies from Beaufort {SAFE_MAX_BFT + 1}"]
    else:
        level = 2
        reasons = [f"max wind {wind} m/s = Beaufort {bft}; not safe from Beaufort {CAUTION_MAX_BFT + 1}"]
    if wind_reason:
        reasons.insert(0, wind_reason)
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

    return {"date": day, "wind_max_ms": wind, **wind_fields, "beaufort": bft,
            "pressure_trend": trend, "pressure_outlook": pressure_outlook(trend),
            "verdict": VERDICTS[level], "reasons": reasons,
            "thresholds_bft": {"safe_up_to": SAFE_MAX_BFT, "caution_from": SAFE_MAX_BFT + 1,
                               "not_safe_from": CAUTION_MAX_BFT + 1}}


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
def get_forecast(day: str | list[str] | None = None) -> dict:
    return _get_forecast(_load_forecast_json(), day)


@_as_json
def get_station_status(station: str | None = None) -> dict:
    return _get_station_status(_load_forecast_json(), station)


@_as_json
def get_warnings() -> dict:
    return _get_warnings(_load_forecast_json())


@_as_json
def sail_safety(day: str) -> dict:
    return _sail_safety(load_forecast_data(), day)


def chat_tool_specs(data: dict | None = None) -> list[dict]:
    """Specs for the chat tools. Built per request, so the `day` enums list the real dates."""
    day_param = {"type": "string", "enum": forecast_dates(data), "description": "Forecast date, YYYY-MM-DD"}
    return [
        _spec("get_forecast",
              "Forecast for the selected point: per day temp min/max, mean pressure, max wind "
              "(m/s and Beaufort) and wind warning, plus the pressure trend and outlook. "
              "`day` can be one date or a list of dates; leave it out to get all days.",
              {"day": {"anyOf": [day_param, {"type": "array", "items": day_param}],
                       "description": "One forecast date or a list of dates (YYYY-MM-DD)"}}),
        _spec("get_station_status",
              "Status (OK/SUSPECT), data issues, distance to the selected point and weight of "
              "the stations. Leave out `station` to get all stations.",
              {"station": _station_param([st["id"] for st in (data or _load_forecast_json())["stations"]])}),
        _spec("get_warnings", "All warnings for the forecast period at the selected point."),
        _spec("sail_safety",
              "Sailing verdict (safe / caution / not safe) for one day at the selected point, "
              "decided by fixed rules from wind (Beaufort) and pressure trend. "
              "Always use this for safety questions.",
              {"day": day_param},
              ["day"]),
    ]
