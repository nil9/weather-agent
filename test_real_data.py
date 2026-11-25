"""Offline tests for real_data.py and the dummy/real separation (no network, no LLM).

The API answers come from tests/fixtures/ (real Bright Sky / Open-Meteo answers, saved with
`python real_data.py --save-fixtures`). real_data.http_get is replaced by a fake that serves
them, and all files are written to a temporary folder, never to data/ or reports/.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import real_data
import tools

FIX = real_data.FIXTURES_DIR
INFO = json.loads((FIX / "fixture_info.json").read_text())
NOW = pd.Timestamp(INFO["recorded_at"])          # "now" when the fixtures were recorded
SEA = INFO["sea_point"]

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


def load(name):
    return json.loads((FIX / name).read_text())


class FakeHttp:
    """Serves the fixtures instead of the network. `fail` makes every call raise."""
    def __init__(self, fail: Exception | None = None):
        self.calls, self.fail = 0, fail

    def __call__(self, url, params, timeout):
        self.calls += 1
        if self.fail:
            raise self.fail
        if url.endswith("/sources"):
            return load("brightsky_sources.json")
        if url.endswith("/weather"):
            path = FIX / f"brightsky_weather_{params['dwd_station_id']}.json"
            if not path.exists():
                raise OSError("no fixture")
            return json.loads(path.read_text())
        if url == real_data.OPENMETEO_URL:
            return load("openmeteo_sea_point.json")
        raise AssertionError(f"unexpected URL {url}")


class TempFolders:
    """Point tools at an empty temporary data/ and reports/ (with the sea point config)."""
    def __enter__(self):
        self.saved = (tools.DATA_DIR, tools.REPORTS_DIR, tools.DATA_SOURCE, real_data.http_get)
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "data").mkdir()
        (self.tmp / "reports").mkdir()
        (self.tmp / "data" / "stations.json").write_text(json.dumps({"stations": [], "forecast_point": SEA}))
        tools.DATA_DIR, tools.REPORTS_DIR = self.tmp / "data", self.tmp / "reports"
        return self

    def __exit__(self, *exc):
        tools.DATA_DIR, tools.REPORTS_DIR, tools.DATA_SOURCE, real_data.http_get = self.saved
        shutil.rmtree(self.tmp)


# --- 1. Conversion -----------------------------------------------------------
print("=== conversion (Kiel-Holtenau fixture) ===")
raw = load("brightsky_weather_02564.json")
df = real_data.observations_from_weather(raw)
check("our column names", list(df.columns) == list(real_data.COLUMNS.values()), list(df.columns))
check("timestamps are UTC", str(df.index.tz) == "UTC")
forecast_ids = {s["id"] for s in raw["sources"] if s["observation_type"] == "forecast"}
check("forecast (MOSMIX) rows dropped", len(df) == sum(r["source_id"] not in forecast_ids for r in raw["weather"]))
row = next(r for r in raw["weather"] if r["source_id"] not in forecast_ids and r["wind_speed"])
ours = df.loc[pd.Timestamp(row["timestamp"]), "wind_speed_ms"]
check("wind km/h -> m/s", ours == round(row["wind_speed"] / 3.6, 1), f"{row['wind_speed']} km/h -> {ours}")
check("pressure kept in hPa", 950 < df["pressure_hpa"].mean() < 1060)

end = df.index[-1]
start = end - pd.Timedelta(days=real_data.HISTORY_DAYS) + pd.Timedelta(hours=1)
holes = df.drop(df.index[[100, 101]]).drop(df.index[200:205])       # a 2 h gap and a 5 h gap
filled, n = real_data.fill_short_gaps(holes, start, end)
t = df.index[100]
expected = round((df.loc[df.index[99], "temperature_c"] * 2 + df.loc[df.index[102], "temperature_c"]) / 3, 1)
check("2 h gap filled (linear)", n == 2 and filled.loc[t, "temperature_c"] == expected,
      f"filled {n} h, value {filled.loc[t, 'temperature_c']} vs {expected}")
check("5 h gap left empty", filled.loc[df.index[200]:df.index[204]].isna().all().all())
check("regular hourly grid", len(filled) == real_data.HISTORY_DAYS * 24)

# --- 2. Selection --------------------------------------------------------------
print("\n=== station selection ===")
sources = load("brightsky_sources.json")
candidates = real_data.candidate_stations(sources["sources"], NOW, SEA)
dists = [c["distance_km"] for c in candidates]
check("candidates sorted by distance", dists == sorted(dists))
check("precipitation-only station (Koehn, no 'current' source) not a candidate",
      "04666" not in {c["dwd_station_id"] for c in candidates})
weather = {p.stem.split("_")[-1]: json.loads(p.read_text()) for p in FIX.glob("brightsky_weather_*.json")}
snap = real_data.build_snapshot(sources, weather, NOW, SEA)
agent_ids = [s["id"] for s in snap["stations"]["stations"]]
pool = snap["stations"]["pool"]
check("agent gets the 4 nearest usable stations to the sea point",
      agent_ids == ["KIEL-HOLTENAU", "SCHOENHAGEN", "FEHMARN", "DOERNICK"], agent_ids)
check("station without pressure history skipped (Hohwacht)",
      "pressure_hpa" in snap["meta"]["skipped"].get("HOHWACHT", ""), snap["meta"]["skipped"].get("HOHWACHT"))
check("station with only recent hours skipped (Leuchtturm Kiel)", "LEUCHTTURM-KIEL" in snap["meta"]["skipped"])
fehmarn_side = {"lat": 54.45, "lon": 10.85}
near = sorted(pool, key=lambda s: tools._distance_km(s["lat"], s["lon"], fehmarn_side["lat"], fehmarn_side["lon"]))[:4]
check("another point gets ITS 4 nearest (Fehmarn first)", near[0]["id"] == "FEHMARN", [s["id"] for s in near])
obs = snap["observations"]
check("same columns as data/observations.csv",
      list(obs.columns) == list(pd.read_csv(Path("data/observations.csv"), nrows=1).columns))
check("all stations end at the same hour", obs.groupby("station_id")["timestamp"].max().nunique() == 1)

# --- 3. Cache and fallback ------------------------------------------------------
print("\n=== cache and fallback ===")
with TempFolders() as tmp:
    http = real_data.http_get = FakeHttp()
    info = real_data.refresh(now=NOW)
    check("first refresh fetches and saves", info["refreshed"] and http.calls > 1, info)
    check("no note for fresh data", info["data_note"] is None, info["data_note"])
    calls = http.calls
    info = real_data.refresh(now=NOW + pd.Timedelta(minutes=30))
    check("cache younger than 60 min: no request", http.calls == calls and not info["refreshed"])

    real_data.http_get = FakeHttp(fail=TimeoutError("timed out"))
    info = real_data.refresh(now=NOW + pd.Timedelta(hours=2))
    check("failed refresh keeps the last good data", info["data_time"] == snap["meta"]["data_time"])
    check("... with a visible 'data from <time>' note",
          info["data_note"] and info["data_note"].startswith("data from") and "refresh failed" in info["data_note"],
          info["data_note"])

with TempFolders():
    real_data.http_get = FakeHttp(fail=OSError("network down"))
    try:
        real_data.refresh(now=NOW)
        check("no data at all -> error", False, "no exception")
    except real_data.RealDataUnavailable as exc:
        check("no data at all -> error, no dummy fallback",
              not tools._safe_path(tools.DATA_DIR, "real/current").exists(), str(exc))

# --- 4. No mixing of dummy and real ---------------------------------------------
print("\n=== no mixing ===")
for made_from, selected in [("dummy", "real"), ("real", "dummy")]:
    with TempFolders():
        # Each source has its own file; this simulates a copied file with the wrong content.
        (tools.REPORTS_DIR / f"forecast_{selected}.json").write_text(json.dumps(
            {"data_source": made_from, "generated_at": "x", "stations": [], "forecast": {"daily": []}}))
        tools.DATA_SOURCE = selected
        for name, call in [("_load_forecast_json", lambda: tools._load_forecast_json()),
                           ("forecast_at", lambda: tools.forecast_at(54.55, 10.45))]:
            try:
                call()
                check(f"{made_from} content in forecast_{selected}.json: {name} refuses", False, "no error")
            except tools.DataSourceMismatch as exc:
                check(f"{made_from} content in forecast_{selected}.json: {name} refuses",
                      made_from in str(exc), str(exc))
with TempFolders():
    (tools.REPORTS_DIR / "forecast_dummy.json").write_text(json.dumps({"generated_at": "x", "forecast": None}))
    tools.DATA_SOURCE = "dummy"
    check("forecast file without data_source counts as dummy",
          tools._load_forecast_json()["generated_at"] == "x")

# --- 5. Open-Meteo ---------------------------------------------------------------
print("\n=== Open-Meteo comparison ===")
om = load("openmeteo_sea_point.json")
day0 = NOW.strftime("%Y-%m-%d")
day1 = (NOW + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
from_hour = (NOW.floor("h") + pd.Timedelta(hours=1)).strftime("%H:%M")
rows = real_data.openmeteo_daily(om, [{"date": day0, "from_utc": from_hour}, {"date": day1}])
h = pd.DataFrame(om["hourly"]).set_index("time")
same_hours = h.loc[f"{day0}T{from_hour}":f"{day0}T23:00"]
check("partial first day uses the same hours as our forecast",
      rows[0]["hours"] == len(same_hours) and rows[0]["wind_max_ms"] == round(same_hours["wind_speed_10m"].max(), 1),
      rows[0])
check("full day = 24 hours", rows[1]["hours"] == 24)
with TempFolders():
    real_data.http_get = FakeHttp(fail=OSError("down"))
    ref = real_data.model_reference(54.55, 10.45, [{"date": day1}], now=NOW)
    check("Open-Meteo down, no cache -> not available", ref["available"] is False, ref)
    real_data.http_get = FakeHttp()
    real_data.model_reference(54.55, 10.45, [{"date": day1}], now=NOW)
    real_data.http_get = FakeHttp(fail=OSError("down"))
    ref = real_data.model_reference(54.55, 10.45, [{"date": day1}], now=NOW + pd.Timedelta(hours=2))
    check("Open-Meteo down, old cache -> values with a note",
          ref["available"] and ref["data_note"] and ref["data_note"].startswith("data from"), ref.get("data_note"))

# --- 6. sail_safety: higher wind of our estimate and Open-Meteo (real data only) ---------
print("\n=== sail_safety with Open-Meteo ===")
import chat                                                    # noqa: E402  (after the fixtures)


def forecast_data(source, ours, model=None, model_available=True, trend="steady"):
    """A minimal forecast dict for one day; `reference` preset so nothing is fetched."""
    data = {"sea_point": {"id": "SEA", "name": "Sea", "lat": 54.55, "lon": 10.45},
            "forecast": {"daily": [{"date": day1, "wind_max_ms": ours}],
                         "pressure_trend": {"label": trend, "change_hpa": 0.0}}}
    if source == "real":
        data["data_source"] = "real"
        data["reference"] = ({"available": True, "daily": [{"date": day1, "wind_max_ms": model}]}
                             if model_available else {"available": False, "reason": "down"})
    return data


r = tools._sail_safety(forecast_data("real", ours=5.0, model=9.9), day1)
check("Open-Meteo higher -> verdict from Open-Meteo (Bft 5 = caution)",
      r["verdict"] == "caution" and r["wind_max_ms"] == 9.9 and r["wind_source"] == "Open-Meteo", r)
check("... and the reasons say which source", "Open-Meteo" in r["reasons"][0] and "5.0" in r["reasons"][0])
r = tools._sail_safety(forecast_data("real", ours=12.0, model=6.0), day1)
check("our estimate higher -> verdict from our estimate (Bft 6 = caution)",
      r["verdict"] == "caution" and r["wind_max_ms"] == 12.0 and r["wind_source"] == "our estimate", r)
r = tools._sail_safety(forecast_data("real", ours=4.0, model=4.0), day1)
check("equal -> 'both (equal)', safe", r["verdict"] == "safe" and r["wind_source"] == "both (equal)", r)
r = tools._sail_safety(forecast_data("real", ours=5.0, model_available=False), day1)
check("Open-Meteo not available -> our estimate, and it says so",
      r["wind_max_ms"] == 5.0 and r["wind_open_meteo_ms"] is None and "not available" in r["reasons"][0], r)
r = tools._sail_safety(forecast_data("real", ours=9.0, model=14.5, trend="falling"), day1)
check("Open-Meteo Bft 7 -> not safe (falling pressure can't go higher)", r["verdict"] == "not safe", r)

dummy = tools._sail_safety(forecast_data("dummy", ours=5.0), day1)
check("dummy: unchanged (our estimate only, no source fields)",
      dummy["verdict"] == "safe" and dummy["wind_max_ms"] == 5.0 and "wind_source" not in dummy
      and len(dummy["reasons"]) == 2, dummy)
check("dummy: Open-Meteo is never fetched", "reference" not in forecast_data("dummy", ours=5.0))

safety = [{"date": day1, "verdict": "caution", "wind_source": "Open-Meteo"}]
q = "Is it safe to sail tomorrow?"
said = lambda answer: chat.check_answer(answer, q, [{"name": "sail_safety"}], [json.dumps({"x": 9.9})], safety)
check("chat rejects an answer that doesn't name the source",
      any("Open-Meteo" in p for p in said("Tomorrow it is caution because of the wind.")))
check("chat accepts an answer that names the source",
      said("Tomorrow is caution: the Open-Meteo model gives the higher wind, 9.9 m/s.") == [])
ours_safety = [{"date": day1, "verdict": "safe", "wind_source": "our estimate"}]
check("chat: 'our estimate' named via the stations",
      chat.check_answer("It is safe; our estimate from the stations is higher.", q, [{"name": "sail_safety"}],
                        [], ours_safety) == [])
check("chat (dummy): no source required",
      chat.check_answer("It is caution tomorrow.", q, [{"name": "sail_safety"}], [],
                        [{"date": day1, "verdict": "caution"}]) == [])

with TempFolders():
    tools.DATA_SOURCE = "real"
    (tools.REPORTS_DIR / "forecast_real.json").write_text(json.dumps(
        {"data_source": "real", "generated_at": "x", "stations": [], "sea_point": SEA,
         "forecast": {"daily": [{"date": day1, "wind_max_ms": 5.0}],
                      "pressure_trend": {"label": "steady", "change_hpa": 0.0}}}))
    real_data.http_get = http = FakeHttp()
    r = tools._sail_safety(tools.load_forecast_data(), day1)
    om_wind = real_data.openmeteo_daily(om, [{"date": day1}])[0]["wind_max_ms"]
    check("real forecast: Open-Meteo fetched on first use, from the fixture",
          http.calls == 1 and r["wind_open_meteo_ms"] == om_wind, r)

# --- 7. Separate forecast files per data source ------------------------------------
print("\n=== separate forecast files ===")
with TempFolders():
    for source in ["dummy", "real"]:
        (tools.REPORTS_DIR / f"forecast_{source}.json").write_text(json.dumps(
            {"data_source": source, "generated_at": source, "forecast": None}))
    for source in ["dummy", "real"]:
        tools.DATA_SOURCE = source
        check(f"{source} mode reads forecast_{source}.json",
              tools.forecast_file() == f"forecast_{source}.json"
              and tools._load_forecast_json()["generated_at"] == source)
    tools.DATA_SOURCE = "real"
    check("real reports go to reports/real/", tools._safe_path(tools.REPORTS_DIR, tools.REPORT_FOLDERS["real"] + "r.md").parent.name == "real")

print(f"\n{'all passed' if not failures else f'{failures} FAILED'}")
