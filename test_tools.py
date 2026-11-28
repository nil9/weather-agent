"""Call each tool once and print the JSON it returns (no LLM involved)."""
import json

import clock
import tools

clock.freeze_for_tests()    # "now" is fixed: the tests give the same result on any day


def show(title, result):
    print(f"\n=== {title} ===")
    print(json.dumps(json.loads(result), indent=2))  # pretty-print the JSON string


show("list_stations()", tools.list_stations())
show("get_observations('WEST', 24)", tools.get_observations("WEST", 24))
show("check_data_quality('NORTH')", tools.check_data_quality("NORTH"))
show("check_data_quality('EAST')", tools.check_data_quality("EAST"))
show("forecast_station('WEST', 3)", tools.forecast_station("WEST", 3))
estimate = tools.estimate_at_sea_point(["NORTH", "SOUTH", "WEST"], 3)
show("estimate_at_sea_point(['NORTH', 'SOUTH', 'WEST'], 3)", estimate)

# write_report only accepts a report that contains the numbers of the last estimate.
show("write_report(placeholder)  (numbers missing -> error)",
     tools.write_report("# Test report\n\nForecast goes here.\n", "test_report.md"))
table = "\n".join(
    f"| {d['date']} | {d['temp_min_c']} | {d['temp_max_c']} | {d['pressure_mean_hpa']} | {d['wind_max_ms']} |"
    for d in json.loads(estimate)["daily"])
show("write_report(full table)", tools.write_report(f"# Test report\n\n{table}\n", "test_report.md"))

# Safety checks: these must return errors, not crash or write files.
show("write_report escape attempt", tools.write_report("bad", "../data/evil.md"))
show("unknown station", tools.get_observations("MOON"))

# Guardrails: SUSPECT stations are refused / dropped by the tools themselves.
show("forecast_station('EAST')  (SUSPECT -> error)", tools.forecast_station("EAST"))
show("estimate_at_sea_point(['EAST'])  (nothing reliable -> error)",
     tools.estimate_at_sea_point(["EAST"]))

print(f"\n{len(tools.TOOL_SPECS)} tool specs:", [t["function"]["name"] for t in tools.TOOL_SPECS])

# ---------------------------------------------------------------------------
# Chat tools (read reports/forecast.json; run `python agent.py` first)
# ---------------------------------------------------------------------------
if not tools.forecast_dates():
    print("\n(skipping chat tools: no reports/forecast.json yet)")
else:
    show("get_warnings()", tools.get_warnings())
    for day in tools.forecast_dates():
        show(f"sail_safety('{day}')", tools.sail_safety(day))
    show("sail_safety('2030-01-01')  (unknown day -> error)", tools.sail_safety("2030-01-01"))

    # The escalation rule, with a faked FALLING pressure trend: every verdict goes up one level.
    real_load = tools._load_forecast_json
    def falling_trend():
        data = real_load()
        data["forecast"]["pressure_trend"]["label"] = "falling"
        return data
    tools._load_forecast_json = falling_trend
    print("\n=== sail_safety with a faked falling pressure trend ===")
    for day in tools.forecast_dates():
        r = json.loads(tools.sail_safety(day))
        print(f"{day}: Bft {r['beaufort']} -> {r['verdict']}   ({'; '.join(r['reasons'])})")
    tools._load_forecast_json = real_load

    print("\nBeaufort checks:", {ms: tools._beaufort(ms) for ms in [0.4, 5.4, 7.9, 8.0, 10.8, 13.9, 33.0]})

# ---------------------------------------------------------------------------
# Forecast at a clicked point (map), computed in code
# ---------------------------------------------------------------------------
import time

if tools.forecast_dates():
    print("\n=== forecast_at (map clicks) ===")
    sea = tools._load_stations()["forecast_point"]
    ok = [s["id"] for s in tools._load_forecast_json()["stations"] if s["status"] == "OK"]
    agent_path = json.loads(tools.estimate_at_sea_point(ok))
    click_path = tools.forecast_at(sea["lat"], sea["lon"])["forecast"]
    same = (agent_path["daily"] == click_path["daily"]
            and agent_path["pressure_trend"] == click_path["pressure_trend"])
    print(f"default sea point, agent vs. click: {'IDENTICAL' if same else 'DIFFERENT (bug!)'}")

    start = time.time(); tools.forecast_at(54.50, 10.30); cached_ms = (time.time() - start) * 1000
    print(f"second point from cache: {cached_ms:.0f} ms")

    # ~200 m off the (polygon) coast at SOUTH: water, so forecast_at accepts it
    near = {s["id"]: s["weight"] for s in tools.forecast_at(54.4337, 10.3900)["stations"]}
    print("just off SOUTH, weights:", near)
    # Exactly on a station: stations sit on land, so test the weighting helper directly
    on = tools._idw_weights(ok, 54.425, 10.378)
    print("exactly on SOUTH (_idw_weights):", {s: round(w, 3) for s, w in on.items()})
    for lat, lon, label in [(55.40, 10.40, "60 km north"), (54.45, 11.06, "Fehmarn (only EAST nearby)")]:
        try:
            tools.forecast_at(lat, lon)
            print(f"{label}: accepted (bug!)")
        except tools.OutsideForecastArea as exc:
            print(f"{label}: {exc}")

# ---------------------------------------------------------------------------
# Land/sea check (Natural Earth 1:10m land, 500 m coast tolerance)
# ---------------------------------------------------------------------------
print("\n=== is_on_land ===")
LAND_CASES = [
    # (name, lat, lon, expected_on_land)
    ("default sea point", 54.55, 10.45, False),
    ("Kiel city (Schreventeich)", 54.33, 10.10, True),
    ("open bay", 54.65, 10.60, False),
    ("Fehmarn interior", 54.47, 11.15, True),
    ("chat test point near WEST", 54.57, 10.12, False),
    # Tolerance, measured from the polygon coastline near SOUTH (Schoenberger Strand):
    ("200 m off the coast", 54.43373, 10.38997, False),
    ("400 m inland (within tolerance)", 54.42950, 10.38416, False),
    ("600 m inland (beyond tolerance)", 54.42809, 10.38223, True),
]
land_failures = 0
for name, lat, lon, expected in LAND_CASES:
    got = tools.is_on_land(lat, lon)
    land_failures += got != expected
    print(f"{'PASS' if got == expected else 'FAIL'}  {name}: {'land' if got else 'water'}")

try:
    tools.forecast_at(54.33, 10.10)
    print("FAIL  forecast_at(Kiel) was accepted")
    land_failures += 1
except tools.PointOnLand as exc:
    print(f"PASS  forecast_at(Kiel): {exc}")

start = time.time()
for _ in range(1000):
    tools.is_on_land(54.33, 10.10)
print(f"1000 cached checks: {(time.time() - start) * 1000:.1f} ms")
print(f"land/sea: {'all passed' if not land_failures else f'{land_failures} FAILED'}")
