"""Offline tests for route_weather.py and the weather-aware route (no network, no LLM).

Weather: tests/fixtures/route_weather_corridor.json, real batched Open-Meteo answers (Forecast +
Marine) for the corridor of the test route Kiel Bay -> west of Fehmarn. The routing data files
in data/ are used as in test_routing.py.
"""
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import real_data
import route_weather as rw
import clock
import routing
import tools

FIX = json.loads((Path(__file__).parent / "tests" / "fixtures" / "route_weather_corridor.json").read_text())
A, B = (54.45, 10.45), (54.45, 10.95)          # Kiel Bay -> west of Fehmarn (around the firing areas)
failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


def fixture_field() -> rw.Field:
    return rw.field_from_answers(FIX["points"], FIX["forecast"], FIX["marine"])


FIELD = fixture_field()
DEPARTURE = FIELD.times[0] + pd.Timedelta(hours=3)
clock.freeze(DEPARTURE)     # "now" (the default departure) is inside the saved forecast, on any day

# --- 1. The field from the batched answers --------------------------------------------
print("=== field from batched Open-Meteo answers ===")
check(f"one entry per point: {FIELD.wind_ms.shape} (points x hours)",
      FIELD.wind_ms.shape == (len(FIX["points"]), len(FIX["forecast"][0]["hourly"]["time"])))
check("gusts and waves present (real)", FIELD.gust_ms is not None and FIELD.waves_m is not None)
check("hourly UTC times", str(FIELD.times.tz) == "UTC" and (FIELD.times[1] - FIELD.times[0]) == pd.Timedelta(hours=1))

# --- 2. Legs: passing times, lookup, Beaufort ---------------------------------------------
print("\n=== legs ===")
original_get = rw.get_field
rw.get_field = lambda points: FIELD                    # no network: the fixture field everywhere
try:
    r = routing.route(*A, *B, weather="show", departure=DEPARTURE)
finally:
    rw.get_field = original_get
legs = r["weather"]["legs"]
check(f"{len(legs)} legs of about {rw.LEG_KM} km cover the route ({r['distance_km']} km)",
      legs[0]["from_km"] == 0 and abs(legs[-1]["to_km"] - r["distance_km"]) < 0.2)
speed_kmh = r["speed_kn"] * routing.KM_PER_NM
ok_times = all(abs((pd.Timestamp(g["time_utc"]) - DEPARTURE) / pd.Timedelta(hours=1)
                   - (g["from_km"] + g["to_km"]) / 2 / speed_kmh) < 1.5 / 60 for g in legs)
check("passing time = departure + distance / speed (shown to the minute)", ok_times)
check("each leg uses the nearest forecast hour",
      all(abs(pd.Timestamp(g["forecast_hour_utc"]) - pd.Timestamp(g["time_utc"])) <= pd.Timedelta(minutes=30)
          for g in legs))
check("Beaufort is exact (tools._beaufort of the wind)", all(g["wind_bft"] == tools._beaufort(g["wind_ms"]) for g in legs))
idx = FIELD.nearest([legs[0]["lat"]], [legs[0]["lon"]])[0]
h = FIELD.hour_index([legs[0]["time_utc"]])[0]
check("the leg's values come from the nearest point and hour",
      legs[0]["wind_ms"] == round(float(FIELD.wind_ms[idx, h]), 1) and legs[0]["waves_m"] == round(float(FIELD.waves_m[idx, h]), 2))
check(f"worst leg: Bft {r['weather']['worst']['wind_bft']}, waves {r['weather']['worst']['waves_m']} m "
      f"({r['weather']['worst']['status']})", r["weather"]["worst"] in legs)

print("\n=== status at the limits (wind > Bft 6, waves > 1.5 m = too rough) ===")
for args, expected in [((4, 0.9), "green"), ((5, None), "yellow"), ((4, 1.0), "yellow"), ((6, 1.5), "yellow"),
                       ((7, None), "red"), ((4, 1.6), "red"), ((5, 0.5, 4), "red")]:
    check(f"status{args} = {expected}", rw.status(*args) == expected, rw.status(*args))

# --- 3. Dummy mode --------------------------------------------------------------------
print("\n=== dummy mode ===")
saved = tools.DATA_SOURCE
tools.DATA_SOURCE = "dummy"
try:
    d = rw.dummy_field([(54.5, 10.6), (54.45, 10.9)])
finally:
    tools.DATA_SOURCE = saved
check("dummy: wind from the dummy forecast, no gusts or waves",
      d.gust_ms is None and d.waves_m is None and np.isfinite(d.wind_ms).all() and "wind only" in d.note)
check("dummy: the waves rule is skipped", rw.status(5, None) == "yellow" and rw.status(4, None) == "green")

# --- 4. A rough leg changes the weather-aware route --------------------------------------
print("\n=== a rough leg changes the weather-aware route ===")
calm = fixture_field()
calm.wind_ms[:] = 5.0                                   # Bft 3 everywhere ...
calm.waves_m[:] = 0.3
# ... except one place on the shortest route: Bft 8 and 2 m waves at every hour.
short = routing.route(*A, *B, weather="off")
mid = short["path"][len(short["path"]) // 2]
p = calm.nearest([mid[0]], [mid[1]])[0]
calm.wind_ms[p, :] = 19.0
calm.waves_m[p, :] = 2.0
rw.get_field = lambda points: calm
try:
    shortest = routing.route(*A, *B, weather="show", departure=DEPARTURE)
    aware = routing.route(*A, *B, weather="aware", departure=DEPARTURE)
    e = routing.explain(*A, *B, weather="aware", departure=DEPARTURE)
finally:
    rw.get_field = original_get
check(f"the shortest route has rough legs ({shortest['weather']['rough_count']})", shortest["weather"]["rough_count"] > 0)
check(f"the weather-aware route has fewer ({aware['weather']['rough_count']})",
      aware["weather"]["rough_count"] < shortest["weather"]["rough_count"])
check(f"... on a different, longer path ({aware['distance_km']} vs {shortest['distance_km']} km)",
      aware["path"] != shortest["path"] and aware["distance_km"] > shortest["distance_km"])
check("... and keeps every other rule (no land, depth >= 3 m, no closed zone)",
      aware["min_depth_m"] >= 3.0 and aware["zones_crossed"] == [] and aware["min_zone_distance_km"] >= 0.5)
check(f"the comparison is returned ({aware['weather']['shortest']['distance_km']} km, "
      f"{aware['weather']['shortest']['rough_count']} rough legs)",
      aware["weather"]["shortest"]["rough_count"] == shortest["weather"]["rough_count"])
wrow = e["detour"][-1]
check(f"the breakdown has the weather step: {wrow['step']} {wrow['extra_text']} ({wrow.get('reason', '')[:80]}…)",
      wrow["step"].startswith("+ weather") and "rough leg" in wrow.get("reason", ""))

rw.get_field = lambda points: fixture_field()
try:
    same = routing.route(*A, *B, weather="aware", departure=DEPARTURE, wind_limit_bft=12, waves_limit_m=9)
finally:
    rw.get_field = original_get
check("nothing rough: the weather-aware route is the shortest route",
      same["path"] == short["path"] and same["weather"]["rough_count"] == 0)

# --- 5. Errors, cache ------------------------------------------------------------------
print("\n=== departure outside the forecast, cache ===")
rw.get_field = lambda points: FIELD
try:
    after = routing.route(*A, *B, weather="show", departure=FIELD.times[-1] + pd.Timedelta(days=2))
    check("departure after the forecast: a route, every leg in no_forecast_legs (no weather)",
          not after["weather"]["legs"] and after["weather"]["no_forecast_legs"]
          and all(g["status"] == "none" for g in after["weather"]["no_forecast_legs"]))
    try:
        routing.route(*A, *B, weather="show", departure=FIELD.times[0] - pd.Timedelta(days=1))
        check("departure before the forecast: error", False, "no error")
    except routing.RouteError as exc:
        check(f"departure before the forecast: clear error ({str(exc)[:70]}…)", "outside the forecast" in str(exc))
    try:
        rw.legs_along(after["path"], FIELD.times[-1] + pd.Timedelta(days=2), 6, FIELD)
        check("legs_along (all legs needed): error past the forecast", False, "no error")
    except rw.RouteWeatherError as exc:
        check("legs_along (all legs needed): error past the forecast", "outside the forecast" in str(exc))
finally:
    rw.get_field = original_get


class FakeHttp:
    def __init__(self):
        self.calls = 0

    def __call__(self, url, params, timeout):
        self.calls += 1
        n = len(params["latitude"].split(","))
        return (FIX["forecast"] if url == rw.FORECAST_URL else FIX["marine"])[:n]


saved_dir, saved_http = tools.DATA_DIR, real_data.http_get
tmp = Path(tempfile.mkdtemp())
tools.DATA_DIR = tmp
real_data.http_get = http = FakeHttp()
try:
    pts = [tuple(p) for p in FIX["points"]]
    now = pd.Timestamp(FIX["fetched_at"])
    first = rw.fetch_real(pts, now=now)
    check(f"one batched request per API for {len(pts)} points ({http.calls} requests)", http.calls == 2)
    again = rw.fetch_real(pts, now=now + pd.Timedelta(minutes=30))
    check("cached: no second request within 60 min", http.calls == 2 and np.allclose(first.wind_ms, again.wind_ms))
finally:
    tools.DATA_DIR, real_data.http_get = saved_dir, saved_http
    shutil.rmtree(tmp)

print(f"\n{'all passed' if not failures else f'{failures} FAILED'}")
