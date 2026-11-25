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
            "id": s["id"], "lat": s["lat"], "lon": s["lon"],
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


@_as_json
def check_data_quality(station: str) -> dict:
    """Look for stuck values, impossible jumps and gaps. Returns OK or SUSPECT."""
    station = _check_station(station)
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

    return {"station": station, "status": "SUSPECT" if issues else "OK", "issues": issues}


@_as_json
def forecast_station(station: str, days: int = 3) -> dict:
    """Holt-Winters forecast for one station, summarised per day."""
    station = _check_station(station)
    days = int(days)
    if not 1 <= days <= 7:
        raise ValueError("days must be between 1 and 7")
    return {"station": station, "method": "Holt-Winters (damped trend, 24 h cycle)",
            "daily": _forecast_station_daily(station, days)}


@_as_json
def estimate_at_sea_point(stations: list[str], days: int = 3) -> dict:
    """Combine station forecasts into a sea-point forecast (inverse-distance weighting)."""
    if not stations:
        raise ValueError("Give at least one station")
    stations = list(dict.fromkeys(_check_station(s) for s in stations))  # normalise, dedupe
    days = int(days)
    if not 1 <= days <= 7:
        raise ValueError("days must be between 1 and 7")

    # Weight = 1 / distance^2  -> closer stations count more. Normalised to sum to 1.
    info = _load_stations()
    sea = info["forecast_point"]
    coords = {s["id"]: s for s in info["stations"]}
    raw = {s: 1 / _distance_km(coords[s]["lat"], coords[s]["lon"], sea["lat"], sea["lon"]) ** 2
           for s in stations}
    weights = {s: w / sum(raw.values()) for s, w in raw.items()}

    forecasts = {s: _forecast_station_daily(s, days) for s in stations}
    fields = ["temp_min_c", "temp_max_c", "pressure_mean_hpa", "wind_max_ms"]
    daily = []
    for i in range(days):
        day = {"date": forecasts[stations[0]][i]["date"]}
        for f in fields:
            day[f] = round(sum(weights[s] * forecasts[s][i][f] for s in stations), 1)
        day["wind_warning"] = _wind_warning(day["wind_max_ms"])
        daily.append(day)

    # Ready-made interpretations, so the LLM never has to judge numbers itself.
    trend = _pressure_trend(daily[0]["pressure_mean_hpa"], daily[-1]["pressure_mean_hpa"])
    warnings = [f"{d['date']}: {d['wind_warning']}" for d in daily if d["wind_warning"]]
    if trend["label"] == "falling":
        warnings.append(f"Falling pressure: {trend['change_hpa']} hPa over the period")

    return {"point": sea["id"], "stations_used": stations,
            "weights": {s: round(w, 3) for s, w in weights.items()}, "daily": daily,
            "pressure_trend": trend, "warnings": warnings}


@_as_json
def write_report(text: str, filename: str = "forecast_report.md") -> dict:
    """Save a Markdown report inside ./reports (and nowhere else)."""
    if not filename.endswith(".md"):
        raise ValueError("filename must end with .md")
    path = _safe_path(REPORTS_DIR, filename)   # raises if it points outside reports/
    REPORTS_DIR.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"saved": str(path.relative_to(BASE_DIR)), "characters": len(text)}


# ---------------------------------------------------------------------------
# Tool specs: what the LLM sees (OpenAI chat-completions "tools" format)
# ---------------------------------------------------------------------------

_STATION_PARAM = {"type": "string", "enum": ["NORTH", "EAST", "SOUTH", "WEST"],
                  "description": "Station id"}
_DAYS_PARAM = {"type": "integer", "minimum": 1, "maximum": 7, "default": 3,
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
          "mean pressure and max wind speed.",
          {"station": _STATION_PARAM, "days": _DAYS_PARAM}, ["station"]),
    _spec("estimate_at_sea_point",
          "Forecast the sea point by combining the forecasts of the given stations with "
          "inverse-distance weighting. Only pass stations whose data quality is OK. "
          "Returns per-day values with a wind_warning, the pressure_trend "
          "(rising/falling/steady) and a list of warnings.",
          {"stations": {"type": "array", "items": _STATION_PARAM, "minItems": 1,
                        "description": "Stations to combine"},
           "days": _DAYS_PARAM},
          ["stations"]),
    _spec("write_report",
          "Save the final forecast report as a Markdown file in the reports folder.",
          {"text": {"type": "string", "description": "The full report in Markdown"},
           "filename": {"type": "string", "default": "forecast_report.md",
                        "description": "File name ending in .md (no folders)"}},
          ["text"]),
]
