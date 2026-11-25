"""Call each tool once and print the JSON it returns (no LLM involved)."""
import json

import tools


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

    near = {s["id"]: s["weight"] for s in tools.forecast_at(54.43, 10.38)["stations"]}
    print("next to SOUTH, weights:", near)
    on = {s["id"]: s["weight"] for s in tools.forecast_at(54.425, 10.378)["stations"]}
    print("exactly on SOUTH, weights:", on)
    for lat, lon, label in [(55.40, 10.40, "60 km north"), (54.45, 11.06, "Fehmarn (only EAST nearby)")]:
        try:
            tools.forecast_at(lat, lon)
            print(f"{label}: accepted (bug!)")
        except tools.OutsideForecastArea as exc:
            print(f"{label}: {exc}")
