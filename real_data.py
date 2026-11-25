"""Real observations (DWD via Bright Sky) and a model reference (Open-Meteo).

Only used when WEATHER_DATA_SOURCE=real. The dummy data in data/ is never touched,
and real data lives in its own folder, so the two can't be mixed by accident:

  data/real/current/stations.json     4 agent stations + `pool` (all usable stations)
  data/real/current/observations.csv  same columns and units as data/observations.csv
  data/real/current/meta.json         fetched_at, data_time (newest observation), hours filled
  data/real/status.json               the last refresh attempt (also failed ones)
  data/real/openmeteo/<lat>_<lon>.json  cached Open-Meteo answers per point

Usage:
  python real_data.py                  refresh now and print the station pool
  python real_data.py --save-fixtures  also save the raw API answers to tests/fixtures/

Only coordinates and dates are sent to the APIs; no personal data.
"""
import argparse
import json
import os
import re
import shutil
import time
import unicodedata
import urllib.parse
import urllib.request

import pandas as pd

import tools

BRIGHTSKY_URL = os.environ.get("BRIGHTSKY_URL", "https://api.brightsky.dev")
OPENMETEO_URL = os.environ.get("OPENMETEO_URL", "https://api.open-meteo.com/v1/forecast")
USER_AGENT = "marine-weather-agent (learning project)"

HISTORY_DAYS = 14            # hours of history per station (same as TRAIN_DAYS)
RECENT_HOURS = 3             # newest complete observation must be at most this old
MIN_COVERAGE = 0.9           # temperature, pressure and wind each in >= 90 % of the history hours
                             # (many DWD stations have no pressure/wind, or only a few days)
MAX_FILL_HOURS = 3           # gaps up to this long are filled linearly; longer gaps stay
SEARCH_RADIUS_M = 90_000     # Bright Sky /sources search radius around the sea point
POOL_RADIUS_KM = 70          # stations farther away than this are not used at all
MAX_POOL = 14                # at most this many stations are fetched
AGENT_STATIONS = 4           # the agent gets the 4 nearest pool stations to the sea point
REQUEST_TIMEOUT_S = 15       # per HTTP request
TOTAL_TIMEOUT_S = 90         # for one whole refresh
MAX_AGE_MIN = 60             # cached data younger than this is not fetched again
KMH_TO_MS = 1 / 3.6          # Bright Sky's default unit for wind is km/h

FIXTURES_DIR = tools.BASE_DIR / "tests" / "fixtures"
OBSERVATION_TYPES = {"historical", "current", "synop"}   # NOT "forecast" (MOSMIX rows)
# Bright Sky field -> our column (observations.csv)
COLUMNS = {"temperature": "temperature_c", "pressure_msl": "pressure_hpa",
           "wind_speed": "wind_speed_ms", "wind_direction": "wind_dir_deg",
           "relative_humidity": "humidity_pct"}
REQUIRED = ["temperature_c", "pressure_hpa", "wind_speed_ms"]   # what the forecast needs

ATTRIBUTION = {
    "observations": "Deutscher Wetterdienst (DWD), via Bright Sky (brightsky.dev)",
    "model": "Open-Meteo.com (CC BY 4.0)",
}


class RealDataUnavailable(RuntimeError):
    """There is no usable real data (the first fetch failed). Never replaced by dummy data."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _path(name: str):
    """A path inside data/real/, checked by the same sandbox rule as the tools."""
    return tools._safe_path(tools.DATA_DIR, f"real/{name}")


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def http_get(url: str, params: dict, timeout: float) -> dict:
    """GET a JSON API. Tests replace this function, so they never touch the network."""
    request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}",
                                     headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def sea_point() -> dict:
    """The sea forecast point. It's a fixed location (configuration), not observation
    data, so both data sources share it."""
    return json.loads(tools._safe_path(tools.DATA_DIR, "stations.json").read_text())["forecast_point"]


def station_id_from_name(name: str) -> str:
    """'Kiel-Holtenau' -> 'KIEL-HOLTENAU', 'Schönhagen (Ostseebad)' -> 'SCHOENHAGEN'."""
    name = re.sub(r"\(.*?\)", "", name)
    for a, b in {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}.items():
        name = name.replace(a, b)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").upper()


# ---------------------------------------------------------------------------
# Pure functions: station selection and conversion (tested on fixtures)
# ---------------------------------------------------------------------------

def candidate_stations(sources: list[dict], now: pd.Timestamp, center: dict) -> list[dict]:
    """DWD stations from Bright Sky /sources that report hourly observations right now,
    nearest to `center` first. One entry per station (a station has several sources)."""
    by_station = {}
    for s in sources:
        if s.get("dwd_station_id") and s["observation_type"] in OBSERVATION_TYPES:
            by_station.setdefault(s["dwd_station_id"], []).append(s)

    found = []
    for dwd_id, group in by_station.items():
        recent = [s for s in group if s["observation_type"] == "current"
                  and pd.Timestamp(s["last_record"]) >= now - pd.Timedelta(hours=RECENT_HOURS)]
        if not recent:
            continue
        # The historical source has the proper name ("Kiel-Holtenau") and exact coordinates.
        historical = [s for s in group if s["observation_type"] == "historical"]
        best = max(historical, key=lambda s: s["last_record"]) if historical else recent[0]
        dist = tools._distance_km(best["lat"], best["lon"], center["lat"], center["lon"])
        if dist <= POOL_RADIUS_KM:
            found.append({"dwd_station_id": dwd_id, "name": best["station_name"],
                          "lat": round(best["lat"], 4), "lon": round(best["lon"], 4),
                          "distance_km": round(dist, 1)})
    found.sort(key=lambda s: s["distance_km"])

    seen = set()
    for s in found:
        sid = station_id_from_name(s["name"])
        s["id"] = sid if sid not in seen else f"{sid}-{s['dwd_station_id']}"
        seen.add(s["id"])
    return found


def observations_from_weather(weather: dict) -> pd.DataFrame:
    """Bright Sky /weather answer -> hourly rows with OUR columns and units.
    Forecast rows (MOSMIX) are dropped; nothing is filled here."""
    observed = {s["id"] for s in weather["sources"] if s["observation_type"] in OBSERVATION_TYPES}
    rows = [r for r in weather["weather"] if r["source_id"] in observed]
    df = pd.DataFrame({"timestamp": pd.to_datetime([r["timestamp"] for r in rows], utc=True),
                       **{ours: [r.get(theirs) for r in rows] for theirs, ours in COLUMNS.items()}})
    df = df.set_index("timestamp").sort_index().astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df["wind_speed_ms"] = (df["wind_speed_ms"] * KMH_TO_MS).round(1)
    return df


def last_complete_hour(df: pd.DataFrame) -> pd.Timestamp | None:
    """Newest hour that has temperature, pressure AND wind (None if there is none)."""
    complete = df[REQUIRED].notna().all(axis=1)
    return complete[complete].index.max() if complete.any() else None


def coverage(df: pd.DataFrame, now: pd.Timestamp) -> dict[str, float]:
    """Share of the HISTORY_DAYS hours that have a value, per required column."""
    window = df[df.index > now - pd.Timedelta(days=HISTORY_DAYS)]
    expected = HISTORY_DAYS * 24
    return {col: round(float(window[col].notna().sum()) / expected, 3) for col in REQUIRED}


def fill_short_gaps(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
                    ) -> tuple[pd.DataFrame, int]:
    """Put the rows on a regular hourly grid [start, end] and fill gaps of at most
    MAX_FILL_HOURS hours (linear; wind direction: last value). Longer gaps stay empty,
    so check_data_quality still reports them. Returns (df, number of hours filled)."""
    grid = pd.date_range(start, end, freq="h", tz="UTC")
    df = df.reindex(grid)
    filled = pd.Series(False, index=grid)
    for col in df.columns:
        s = df[col]
        missing = s.isna()
        run_id = (missing != missing.shift()).cumsum()
        run_length = missing.groupby(run_id).transform("sum")
        inside = s.ffill().notna() & s.bfill().notna()        # a value before AND after
        short = missing & (run_length <= MAX_FILL_HOURS) & inside
        filler = s.ffill() if col == "wind_dir_deg" else s.interpolate(limit_area="inside")
        df.loc[short, col] = filler[short].round(1)
        filled |= short
    return df, int(filled.sum())


def build_snapshot(sources: dict, weather_by_id: dict, now: pd.Timestamp, sea: dict) -> dict:
    """All API answers -> the files we save (stations, observations, meta). Pure."""
    candidates = candidate_stations(sources["sources"], now, sea)[:MAX_POOL]
    frames, skipped = {}, {}
    for c in candidates:
        weather = weather_by_id.get(c["dwd_station_id"])
        if weather is None:
            skipped[c["id"]] = "could not be fetched"
            continue
        df = observations_from_weather(weather)
        last = last_complete_hour(df)
        if last is None:
            skipped[c["id"]] = "no temperature + pressure + wind observations"
        elif last < now - pd.Timedelta(hours=RECENT_HOURS):
            skipped[c["id"]] = f"newest complete observation {_iso(last)} is too old"
        elif low := {k: v for k, v in coverage(df, now).items() if v < MIN_COVERAGE}:
            skipped[c["id"]] = ("not enough history: " +
                                ", ".join(f"{k} in {v:.0%} of hours" for k, v in low.items()))
        else:
            frames[c["id"]] = df
    pool = [c for c in candidates if c["id"] in frames]
    if not pool:
        raise RealDataUnavailable("No DWD station near Kiel Bay has recent observations.")

    # All stations end at the same hour, so their forecasts start at the same hour.
    end = min(last_complete_hour(frames[c["id"]]) for c in pool)
    start = end - pd.Timedelta(days=HISTORY_DAYS) + pd.Timedelta(hours=1)
    parts, filled = [], {}
    for c in pool:
        df, filled[c["id"]] = fill_short_gaps(frames[c["id"]].loc[:end], start, end)
        df.insert(0, "station_id", c["id"])
        parts.append(df)
    obs = pd.concat(parts).rename_axis("timestamp").reset_index()
    obs["timestamp"] = obs["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def station(c):
        return {k: c[k] for k in ["id", "name", "lat", "lon", "dwd_station_id"]}

    nearest_to_sea = sorted(pool, key=lambda c: tools._distance_km(c["lat"], c["lon"], sea["lat"], sea["lon"]))
    return {
        "stations": {"stations": [station(c) for c in nearest_to_sea[:AGENT_STATIONS]],
                     "pool": [station(c) for c in pool],
                     "forecast_point": sea},
        "observations": obs[["timestamp", "station_id", *COLUMNS.values()]],
        "meta": {"fetched_at": _iso(now), "data_time": _iso(end),
                 "history_days": HISTORY_DAYS, "filled_hours": filled, "skipped": skipped,
                 "source": f"{BRIGHTSKY_URL} (DWD observations)",
                 "attribution": ATTRIBUTION["observations"]},
    }


# ---------------------------------------------------------------------------
# Fetching, cache and fallback
# ---------------------------------------------------------------------------

def fetch_raw(now: pd.Timestamp, sea: dict) -> tuple[dict, dict]:
    """Call Bright Sky: /sources once, then /weather per candidate station.
    Each request has REQUEST_TIMEOUT_S, the whole fetch TOTAL_TIMEOUT_S."""
    started = time.monotonic()

    def timeout():
        left = TOTAL_TIMEOUT_S - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError(f"fetching took longer than {TOTAL_TIMEOUT_S} s")
        return min(REQUEST_TIMEOUT_S, left)

    sources = http_get(f"{BRIGHTSKY_URL}/sources",
                       {"lat": sea["lat"], "lon": sea["lon"], "max_dist": SEARCH_RADIUS_M}, timeout())
    since = (now - pd.Timedelta(days=HISTORY_DAYS, hours=3)).floor("h")
    weather = {}
    for c in candidate_stations(sources["sources"], now, sea)[:MAX_POOL]:
        try:
            weather[c["dwd_station_id"]] = http_get(
                f"{BRIGHTSKY_URL}/weather",
                {"dwd_station_id": c["dwd_station_id"], "date": _iso(since),
                 "last_date": _iso(now), "tz": "Etc/UTC"}, timeout())
        except TimeoutError:
            raise                        # the total time is used up: the whole refresh fails
        except OSError:
            pass                         # this one station failed; build_snapshot skips it
    return sources, weather


def write_snapshot(snapshot: dict):
    """Write the new files to data/real/incoming/, then swap that folder in as
    data/real/current/. Readers never see half-old, half-new files."""
    incoming, current, previous = _path("incoming"), _path("current"), _path("previous")
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.mkdir(parents=True)
    (incoming / "stations.json").write_text(json.dumps(snapshot["stations"], indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    snapshot["observations"].to_csv(incoming / "observations.csv", index=False)
    (incoming / "meta.json").write_text(json.dumps(snapshot["meta"], indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    shutil.rmtree(previous, ignore_errors=True)
    if current.exists():
        current.rename(previous)
    incoming.rename(current)
    shutil.rmtree(previous, ignore_errors=True)


def load_meta() -> dict | None:
    path = _path("current/meta.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_status() -> dict:
    path = _path("status.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_status(status: dict):
    _path("").mkdir(parents=True, exist_ok=True)
    _path("status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def data_info(now: pd.Timestamp | None = None) -> dict:
    """What the page and forecast.json show about the real data: the newest observation
    time, and a note if the data is old or the last refresh failed."""
    meta = load_meta()
    if meta is None:
        raise RealDataUnavailable("No real data yet. Run `python real_data.py` or "
                                  "`WEATHER_DATA_SOURCE=real python agent.py`.")
    now = now or _now()
    status = _load_status()
    data_time = pd.Timestamp(meta["data_time"])
    note = None
    failed = (not status.get("ok", True)
              and pd.Timestamp(status["last_attempt"]) >= pd.Timestamp(meta["fetched_at"]))
    if failed or now - data_time > pd.Timedelta(hours=RECENT_HOURS):
        note = f"data from {data_time:%Y-%m-%d %H:%M} UTC"
        if failed:
            note += f"; refresh failed: {status['error']}"
    return {"data_source": "real", "data_time": meta["data_time"], "fetched_at": meta["fetched_at"],
            "data_note": note, "filled_hours": meta["filled_hours"], "attribution": ATTRIBUTION}


def refresh(max_age_min: float = MAX_AGE_MIN, now: pd.Timestamp | None = None,
            save_fixtures: bool = False) -> dict:
    """Fetch new real data unless the cache is younger than `max_age_min`.
    On failure the last good data stays (with a "data from <time>" note). Without any
    good data it raises RealDataUnavailable: there is NO fallback to dummy data."""
    now = now or _now()
    meta = load_meta()
    if meta and now - pd.Timestamp(meta["fetched_at"]) < pd.Timedelta(minutes=max_age_min):
        return {**data_info(now), "refreshed": False}
    try:
        sea = sea_point()
        sources, weather = fetch_raw(now, sea)
        if save_fixtures:
            save_fixture_files(sources, weather, now)
        write_snapshot(build_snapshot(sources, weather, now, sea))
        _write_status({"last_attempt": _iso(now), "ok": True})
    except (OSError, TimeoutError, ValueError, KeyError, RealDataUnavailable) as exc:
        # OSError includes urllib's URLError and socket timeouts; ValueError bad JSON.
        _write_status({"last_attempt": _iso(now), "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
        if load_meta() is None:
            raise RealDataUnavailable(f"No real data: the first fetch failed ({exc}). "
                                      "There is no fallback to dummy data.") from exc
    return {**data_info(now), "refreshed": load_meta()["fetched_at"] == _iso(now)}


# ---------------------------------------------------------------------------
# Open-Meteo: a model value at any point, for comparison only (not used by the agent)
# ---------------------------------------------------------------------------

def openmeteo_daily(raw: dict, daily_rows: list[dict]) -> list[dict]:
    """Aggregate Open-Meteo's hourly forecast over the SAME hours as our forecast:
    for each of our days from `from_utc` (or 00:00) to 23:00 UTC. Pure."""
    h = raw["hourly"]
    df = pd.DataFrame({"temp": h["temperature_2m"], "pressure": h["pressure_msl"],
                       "wind": h["wind_speed_10m"]},
                      index=pd.to_datetime(h["time"]).tz_localize("UTC"))
    out = []
    for row in daily_rows:
        start = pd.Timestamp(f"{row['date']}T{row.get('from_utc', '00:00')}", tz="UTC")
        hours = df.loc[start:pd.Timestamp(f"{row['date']}T23:00", tz="UTC")]
        if hours.empty or hours.isna().any().any():
            out.append({"date": row["date"], "available": False})
            continue
        out.append({"date": row["date"], "hours": len(hours),
                    "temp_min_c": round(float(hours["temp"].min()), 1),
                    "temp_max_c": round(float(hours["temp"].max()), 1),
                    "pressure_mean_hpa": round(float(hours["pressure"].mean()), 1),
                    "wind_max_ms": round(float(hours["wind"].max()), 1)})
    return out


def openmeteo_params(lat: float, lon: float) -> dict:
    return {"latitude": round(lat, 2), "longitude": round(lon, 2),
            "hourly": "temperature_2m,pressure_msl,wind_speed_10m",
            "wind_speed_unit": "ms", "timezone": "UTC", "forecast_days": 4}


def model_reference(lat: float, lon: float, daily_rows: list[dict],
                    now: pd.Timestamp | None = None) -> dict:
    """Open-Meteo at (lat, lon), aggregated like our forecast. Cached per point for
    MAX_AGE_MIN; on failure the last good answer with a note, else "not available"."""
    now = now or _now()
    path = _path(f"openmeteo/{lat:.2f}_{lon:.2f}.json")
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    note = None
    if cached is None or now - pd.Timestamp(cached["fetched_at"]) >= pd.Timedelta(minutes=MAX_AGE_MIN):
        try:
            raw = http_get(OPENMETEO_URL, openmeteo_params(lat, lon), REQUEST_TIMEOUT_S)
            cached = {"fetched_at": _iso(now), "raw": raw}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cached), encoding="utf-8")
            tmp.replace(path)                     # atomic: never a half-written cache file
        except (OSError, ValueError) as exc:
            if cached is None:
                return {"available": False, "reason": f"Open-Meteo not available ({type(exc).__name__})",
                        "attribution": ATTRIBUTION["model"]}
            note = f"data from {pd.Timestamp(cached['fetched_at']):%Y-%m-%d %H:%M} UTC; refresh failed"
    return {"available": True, "model": "Open-Meteo", "fetched_at": cached["fetched_at"],
            "data_note": note, "grid_point": {"lat": cached["raw"]["latitude"], "lon": cached["raw"]["longitude"]},
            "daily": openmeteo_daily(cached["raw"], daily_rows), "attribution": ATTRIBUTION["model"]}


# ---------------------------------------------------------------------------
# Fixtures for offline tests
# ---------------------------------------------------------------------------

WEATHER_FIELDS = ["timestamp", "source_id", *COLUMNS]


def save_fixture_files(sources: dict, weather: dict, now: pd.Timestamp):
    """Save the real API answers for test_real_data.py. Weather rows keep only the
    fields we read (otherwise each file is ~170 KB); the values are unchanged."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "brightsky_sources.json").write_text(json.dumps(sources), encoding="utf-8")
    for dwd_id, answer in weather.items():
        trimmed = {"sources": answer["sources"],
                   "weather": [{k: r.get(k) for k in WEATHER_FIELDS} for r in answer["weather"]]}
        (FIXTURES_DIR / f"brightsky_weather_{dwd_id}.json").write_text(json.dumps(trimmed), encoding="utf-8")
    sea = sea_point()
    om = http_get(OPENMETEO_URL, openmeteo_params(sea["lat"], sea["lon"]), REQUEST_TIMEOUT_S)
    (FIXTURES_DIR / "openmeteo_sea_point.json").write_text(json.dumps(om), encoding="utf-8")
    (FIXTURES_DIR / "fixture_info.json").write_text(json.dumps(
        {"recorded_at": _iso(now), "sea_point": sea,
         "note": "Real API answers (Bright Sky / DWD, Open-Meteo). Weather rows trimmed to the fields we use."},
        indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Fetch real DWD observations for Kiel Bay.")
    parser.add_argument("--save-fixtures", action="store_true",
                        help="also save the raw API answers to tests/fixtures/")
    args = parser.parse_args()

    info = refresh(max_age_min=0, save_fixtures=args.save_fixtures)
    print(f"Refreshed: {info['refreshed']}   observations until {info['data_time']}"
          + (f"   NOTE: {info['data_note']}" if info["data_note"] else ""))
    stations = json.loads(_path("current/stations.json").read_text(encoding="utf-8"))
    meta = load_meta()
    sea = stations["forecast_point"]
    agent_ids = {s["id"] for s in stations["stations"]}
    print(f"\n| Station | DWD id | km to sea point | Agent | Hours filled | Quality |")
    print("|---|---|---|---|---|---|")
    old_source = tools.DATA_SOURCE
    tools.DATA_SOURCE = "real"               # so the quality check reads the real files
    try:
        for s in stations["pool"]:
            issues = tools._quality_issues(s["id"])
            dist = tools._distance_km(s["lat"], s["lon"], sea["lat"], sea["lon"])
            print(f"| {s['id']} ({s['name']}) | {s['dwd_station_id']} | {dist:.1f} | "
                  f"{'yes' if s['id'] in agent_ids else ''} | {meta['filled_hours'][s['id']]} | "
                  f"{'SUSPECT: ' + '; '.join(issues) if issues else 'OK'} |")
    finally:
        tools.DATA_SOURCE = old_source
    for sid, reason in meta["skipped"].items():
        print(f"  skipped {sid}: {reason}")


if __name__ == "__main__":
    main()
