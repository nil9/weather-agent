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
