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
show("estimate_at_sea_point(['NORTH', 'SOUTH', 'WEST'], 3)",
     tools.estimate_at_sea_point(["NORTH", "SOUTH", "WEST"], 3))
show("write_report(...)", tools.write_report("# Test report\n\nHello from test_tools.py\n", "test_report.md"))

# Safety checks: these must return errors, not crash or write files.
show("write_report escape attempt", tools.write_report("bad", "../data/evil.md"))
show("unknown station", tools.get_observations("MOON"))

print(f"\n{len(tools.TOOL_SPECS)} tool specs:", [t["function"]["name"] for t in tools.TOOL_SPECS])
