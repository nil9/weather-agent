"""Weather along a route, at the time the boat passes each leg (code only, no LLM).

  - Legs: the route is cut into legs of LEG_KM; each leg's passing time is
    departure + (distance from the start to the leg's middle) / speed.
  - Weather field: hourly wind, gusts and waves at a set of points.
      real data:  Open-Meteo Forecast (wind, gusts) + Open-Meteo Marine (waves), ONE batched
                  request per API for all points, cached 60 min in data/real/route_weather/
      dummy data: the dummy forecast reused: hourly Holt-Winters wind of the OK stations,
                  inverse-distance weighted to each point (tools._idw_weights). No gusts, no
                  waves in the dummy data ("dummy data: wind only").
  - Each leg takes the nearest field point and the forecast hour of its passing time.
  - Status per leg: green / yellow / red ("too rough") from fixed, configurable limits.
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import real_data
import tools

FORECAST_URL = real_data.OPENMETEO_URL
MARINE_URL = os.environ.get("OPENMETEO_MARINE_URL", "https://marine-api.open-meteo.com/v1/marine")
ROUGH_WIND_BFT = 6            # above this: "too rough" (so Bft 7 and more)
ROUGH_WAVES_M = 1.5           # above this: "too rough"
CAUTION_BFT = 5               # Bft 5-6: yellow
CAUTION_WAVES_M = 1.0         # waves 1.0-1.5 m: yellow
LEG_KM = 3.0                  # leg length along the route
GRID_STEP_DEG = 0.1           # corridor grid for the weather-aware route
CORRIDOR_KM = 10.0
MAX_GRID_POINTS = 150
CACHE_MIN = 60
STATUS_ORDER = {"green": 0, "yellow": 1, "red": 2}
SOURCE = {"real": "Open-Meteo.com Forecast + Marine (CC BY 4.0)", "dummy": "dummy forecast (wind only)"}


class RouteWeatherError(ValueError):
    """Weather along the route can't be given (departure outside the forecast, no data)."""


@dataclass
class Field:
    """Hourly weather at a set of points: arrays [point, hour]."""
    lat: np.ndarray
    lon: np.ndarray
    times: pd.DatetimeIndex                 # hourly, UTC
    wind_ms: np.ndarray
    gust_ms: np.ndarray | None
    waves_m: np.ndarray | None
    source: str
    note: str | None = None

    def nearest(self, lat, lon) -> np.ndarray:
        """Index of the nearest field point (in km) for each (lat, lon)."""
        x, y = tools._to_km(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        fx, fy = tools._to_km(self.lon, self.lat)
        d = (np.asarray(x)[..., None] - fx) ** 2 + (np.asarray(y)[..., None] - fy) ** 2
        return d.argmin(axis=-1)

    def hour_index(self, when) -> np.ndarray:
        """Index of the nearest forecast hour; RouteWeatherError if outside the forecast."""
        when = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(when, utc=True)))
        first, last = self.times[0], self.times[-1]
        if (when < first - pd.Timedelta(minutes=30)).any() or (when > last + pd.Timedelta(minutes=30)).any():
            raise RouteWeatherError(
                f"The trip ({when.min():%a %d %b %H:%M}–{when.max():%a %d %b %H:%M} UTC) is outside the "
                f"forecast ({first:%a %d %b %H:%M}–{last:%a %d %b %H:%M} UTC). Choose another departure time.")
        hours = (when - first) / pd.Timedelta(hours=1)
        return np.clip(np.round(np.asarray(hours)).astype(int), 0, len(self.times) - 1)


# ---------------------------------------------------------------------------
# Getting the field
# ---------------------------------------------------------------------------

def _as_list(answer):
    return answer if isinstance(answer, list) else [answer]


def _cache_path(points: list[tuple[float, float]]):
    key = hashlib.sha1(json.dumps(points).encode()).hexdigest()[:16]
    return tools._safe_path(tools.DATA_DIR, f"real/route_weather/{key}.json")


def fetch_real(points: list[tuple[float, float]], now: pd.Timestamp | None = None) -> Field:
    """Open-Meteo for all points: one batched Forecast request and one batched Marine request,
    cached CACHE_MIN minutes. On failure the last good answer is used (with a note)."""
    now = now or pd.Timestamp.now(tz="UTC")
    points = [(round(a, 2), round(b, 2)) for a, b in points]
    path = _cache_path(points)
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    note = None
    if cached is None or now - pd.Timestamp(cached["fetched_at"]) >= pd.Timedelta(minutes=CACHE_MIN):
        lats = ",".join(f"{a:.2f}" for a, _ in points)
        lons = ",".join(f"{b:.2f}" for _, b in points)
        try:
            wind = real_data.http_get(FORECAST_URL, {
                "latitude": lats, "longitude": lons, "hourly": "wind_speed_10m,wind_gusts_10m",
                "wind_speed_unit": "ms", "timezone": "UTC", "forecast_days": 4}, real_data.REQUEST_TIMEOUT_S)
            marine = real_data.http_get(MARINE_URL, {
                "latitude": lats, "longitude": lons, "hourly": "wave_height",
                "timezone": "UTC", "forecast_days": 4}, real_data.REQUEST_TIMEOUT_S)
            cached = {"fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "points": points,
                      "forecast": _as_list(wind), "marine": _as_list(marine)}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cached), encoding="utf-8")
            tmp.replace(path)
        except (OSError, ValueError, KeyError) as exc:
            if cached is None:
                raise RouteWeatherError(f"Weather along the route is not available ({type(exc).__name__}).") from exc
            note = f"weather data from {pd.Timestamp(cached['fetched_at']):%Y-%m-%d %H:%M} UTC; refresh failed"
    return field_from_answers(cached["points"], cached["forecast"], cached["marine"], note)


def field_from_answers(points, forecast: list, marine: list, note: str | None = None) -> Field:
    """Open-Meteo answers (one entry per point, same order) -> Field. Pure (tested on fixtures)."""
    times = pd.DatetimeIndex(pd.to_datetime(forecast[0]["hourly"]["time"]).tz_localize("UTC"))

    def stack(answers, key):
        rows = []
        for a in answers:
            t = pd.DatetimeIndex(pd.to_datetime(a["hourly"]["time"]).tz_localize("UTC"))
            s = pd.Series(a["hourly"][key], index=t, dtype=float).reindex(times)
            rows.append(s.to_numpy())
        return np.array(rows)

    waves = stack(marine, "wave_height") if marine else None
    return Field(lat=np.array([p[0] for p in points], dtype=float), lon=np.array([p[1] for p in points], dtype=float),
                 times=times, wind_ms=stack(forecast, "wind_speed_10m"), gust_ms=stack(forecast, "wind_gusts_10m"),
                 waves_m=waves, source=SOURCE["real"], note=note)


def dummy_field(points: list[tuple[float, float]]) -> Field:
    """The dummy forecast reused: hourly wind of the stations the last agent run marked OK,
    inverse-distance weighted to each point. No gusts or waves exist in the dummy data."""
    run = tools._load_forecast_json()
    ok = [s["id"] for s in run["stations"] if s["status"] == "OK"]
    days = len(run["forecast"]["daily"])
    hourly = {s: tools._forecast_station_hourly(s, days)["wind_speed_ms"] for s in ok}
    times = pd.DatetimeIndex(hourly[ok[0]].index)
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    wind = []
    for lat, lon in points:
        w = tools._idw_weights(ok, lat, lon)
        wind.append(sum(w[s] * hourly[s].to_numpy() for s in ok))
    return Field(lat=np.array([p[0] for p in points], dtype=float), lon=np.array([p[1] for p in points], dtype=float),
                 times=times, wind_ms=np.array(wind), gust_ms=None, waves_m=None,
                 source=SOURCE["dummy"], note="dummy data: wind only (no gusts or waves)")


def get_field(points: list[tuple[float, float]]) -> Field:
    return fetch_real(points) if tools.DATA_SOURCE == "real" else dummy_field(points)


def corridor_points(path: list[list[float]], ends: list[list[float]]) -> list[tuple[float, float]]:
    """A ~0.1° grid over the route's corridor (route + ends + CORRIDOR_KM), at most MAX_GRID_POINTS."""
    lats = [p[0] for p in path + ends]
    lons = [p[1] for p in path + ends]
    pad_lat = CORRIDOR_KM / tools._KM_PER_DEG_LAT
    pad_lon = CORRIDOR_KM / tools._KM_PER_DEG_LON
    lat0, lat1 = min(lats) - pad_lat, max(lats) + pad_lat
    lon0, lon1 = min(lons) - pad_lon, max(lons) + pad_lon
    step = GRID_STEP_DEG
    while math.ceil((lat1 - lat0) / step + 1) * math.ceil((lon1 - lon0) / step + 1) > MAX_GRID_POINTS:
        step *= 1.25
    return [(round(a, 3), round(b, 3)) for a in np.arange(lat0, lat1 + 1e-9, step)
            for b in np.arange(lon0, lon1 + 1e-9, step)]


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def status(bft: int, waves: float | None, wind_limit_bft: int = ROUGH_WIND_BFT,
           waves_limit_m: float = ROUGH_WAVES_M) -> str:
    if bft > wind_limit_bft or (waves is not None and waves > waves_limit_m):
        return "red"
    if bft >= CAUTION_BFT or (waves is not None and waves >= CAUTION_WAVES_M):
        return "yellow"
    return "green"


def legs_along(path: list[list[float]], departure: pd.Timestamp, speed_kn: float, field: Field,
               wind_limit_bft: int = ROUGH_WIND_BFT, waves_limit_m: float = ROUGH_WAVES_M,
               leg_km: float = LEG_KM) -> list[dict]:
    """The route cut into legs of leg_km, each with its passing time and weather."""
    pts = [tools._to_km(lon, lat) for lat, lon in path]
    seg = [math.dist(p, q) for p, q in zip(pts, pts[1:])]
    total = sum(seg)
    if total == 0:
        bounds = [(0.0, 0.0)]
    else:
        n = max(1, math.ceil(total / leg_km))
        edges = np.linspace(0, total, n + 1)
        bounds = list(zip(edges[:-1], edges[1:]))
    cum = np.concatenate([[0.0], np.cumsum(seg)]) if seg else np.array([0.0])

    def point_at(d):
        if not seg:
            return pts[0]
        i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
        t = 0 if seg[i] == 0 else (d - cum[i]) / seg[i]
        p, q = pts[i], pts[i + 1]
        return p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])

    def coords_between(a, b):
        """The route's points from km a to km b (for drawing the leg in its colour)."""
        out = [point_at(a)]
        out += [pts[i] for i in range(1, len(pts) - 1) if a < cum[i] < b]
        out.append(point_at(b))
        return [[round(float(y) / tools._KM_PER_DEG_LAT, 5), round(float(x) / tools._KM_PER_DEG_LON, 5)]
                for x, y in out]

    km_per_h = speed_kn * 1.852
    mids = [(a + b) / 2 for a, b in bounds]
    xy = [point_at(d) for d in mids]
    lons = [x / tools._KM_PER_DEG_LON for x, _ in xy]
    lats = [y / tools._KM_PER_DEG_LAT for _, y in xy]
    times = [departure + pd.Timedelta(hours=d / km_per_h) for d in mids]
    starts = [departure + pd.Timedelta(hours=a / km_per_h) for a, _ in bounds]
    idx = field.nearest(lats, lons)
    hours = field.hour_index(times)
    legs = []
    for k, ((a, b), lat, lon, i, h) in enumerate(zip(bounds, lats, lons, idx, hours)):
        wind = float(field.wind_ms[i, h])
        gust = None if field.gust_ms is None or np.isnan(field.gust_ms[i, h]) else float(field.gust_ms[i, h])
        waves = None if field.waves_m is None or np.isnan(field.waves_m[i, h]) else float(field.waves_m[i, h])
        bft = tools._beaufort(wind)
        legs.append({
            "leg": k + 1, "from_km": round(float(a), 1), "to_km": round(float(b), 1),
            "start_utc": starts[k].strftime("%Y-%m-%dT%H:%MZ"), "time_utc": times[k].strftime("%Y-%m-%dT%H:%MZ"),
            "lat": round(float(lat), 4), "lon": round(float(lon), 4),
            "wind_ms": round(wind, 1), "wind_bft": bft,
            "gust_ms": None if gust is None else round(gust, 1),
            "gust_bft": None if gust is None else tools._beaufort(gust),
            "waves_m": None if waves is None else round(waves, 2),
            "status": status(bft, waves, wind_limit_bft, waves_limit_m),
            "forecast_hour_utc": field.times[h].strftime("%Y-%m-%dT%H:%MZ"),
            "coords": coords_between(a, b),
        })
    return legs


def summarize(legs: list[dict]) -> dict:
    """Worst leg and rough legs of a route."""
    if not legs:
        return {"worst": None, "rough_legs": [], "rough_count": 0}
    worst = max(legs, key=lambda g: (STATUS_ORDER[g["status"]], g["wind_bft"], g["waves_m"] or 0))
    rough = [g for g in legs if g["status"] == "red"]
    return {"worst": worst, "rough_legs": rough, "rough_count": len(rough),
            "max_wind_bft": max(g["wind_bft"] for g in legs),
            "max_waves_m": max((g["waves_m"] for g in legs if g["waves_m"] is not None), default=None)}
