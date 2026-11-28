"""Tests for the voyage-planning feature (rulebook, rule checker, vessel profiles, candidates,
passage plan). Code only, offline: the route weather uses the saved Open-Meteo fixture.

Run:  python test_voyage.py
"""
import json
from pathlib import Path

import rule_checker

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


# --- Phase A: rulebook ----------------------------------------------------------------
print("=== Phase A: rulebook (rules/passage_rules.yaml) ===")
rules = rule_checker.load_rules()
ids = [r["id"] for r in rules]
check(f"{len(rules)} rules load and validate (unique ids, known types, sources)",
      len(rules) >= 20 and len(set(ids)) == len(ids) and all(r["source"] for r in rules))
by_type = {t: [r["id"] for r in rules if r["type"] == t] for t in rule_checker.RULE_TYPES}
check(f"types used: { {t: len(v) for t, v in by_type.items()} }", all(by_type[t] for t in rule_checker.RULE_TYPES))
for rid in ["R-UKC-01", "R-UKC-02", "R-ZONE-01", "R-ZONE-02", "R-WIND-01", "R-BUF-01", "R-BUF-02",
            "R-COLREG-09", "R-WX-01", "R-WX-02", "R-WX-03", "R-AREA-01"]:
    check(f"modelled rule {rid}: {rule_checker.rule(rid)['factor']}", rule_checker.rule(rid)["type"] != "not_modelled")
for rid in ["R-ENC-01", "R-NTM-01", "R-AIS-01", "R-ICE-01", "R-CUR-01", "R-TSS-01", "R-PILOT-02", "R-REP-01"]:
    r = rule_checker.rule(rid)
    check(f"not modelled, with a reason: {rid} ({r['factor']})", r["type"] == "not_modelled" and len(r["reason"]) > 20)
check("COLREG Rule 9 is cited for narrow passages", "Rule 9" in rule_checker.rule("R-COLREG-09")["source"])
check("UKC rule is fed by the existing settings (draft, margin)",
      rule_checker.rule("R-UKC-01")["settings"] == ["draft_m", "margin_m"])

# --- Phase B: rule checker ------------------------------------------------------------
print("\n=== Phase B: rule checker (rule_checker.check_route) ===")
import time

import pandas as pd

import clock
import route_weather as rw
import routing

FIX = json.loads((Path(__file__).parent / "tests" / "fixtures" / "route_weather_corridor.json").read_text())
FIELD = rw.field_from_answers(FIX["points"], FIX["forecast"], FIX["marine"])
DEPARTURE = FIELD.times[0] + pd.Timedelta(hours=3)
clock.freeze(DEPARTURE)     # "now" (the default departure) is inside the saved forecast, on any day
rw.get_field = lambda points: FIELD                     # offline: the saved Open-Meteo answers everywhere
routing.warm_up()


def checked(a, b, **kw):
    extra = {k: kw.pop(k) for k in ("air_draught_m", "gust_limit_bft", "block_coefficient", "dwt") if k in kw}
    started = time.time()
    r = routing.route(*a, *b, weather="show", departure=DEPARTURE, **kw)
    rc = rule_checker.check_route(r, **extra)
    return r, rc, time.time() - started


def status(rc, rid):
    return next(row for row in rc["rows"] if row["id"] == rid)


KIEL, FEHMARN_W = (54.45, 10.45), (54.45, 10.95)
r, rc, sec = checked(KIEL, FEHMARN_W)
check(f"one row per rule ({len(rc['rows'])}), route + rule check in {sec:.2f} s (< 3 s)",
      len(rc["rows"]) == len(rules) and sec < 3)
check("every row has id, factor, status, source", all(row["id"] and row["factor"] and row["source"]
                                                      and row["status"] in rule_checker.STATUSES for row in rc["rows"]))
order = [rule_checker.STATUS_ORDER[row["status"]] for row in rc["rows"]]
check("failed / warned rules first, not modelled last", order == sorted(order))
for rid in ["R-UKC-01", "R-ZONE-01", "R-ZONE-02", "R-WIND-01", "R-BUF-01", "R-BUF-02", "R-COLREG-09", "R-WX-04"]:
    row = status(rc, rid)
    check(f"Kiel Bay -> west of Fehmarn: {rid} pass ({row['value']}; limit {row['limit']})", row["status"] == "pass")
check(f"not modelled rules reported as such ({rc['counts']['not_modelled']})",
      rc["counts"]["not_modelled"] >= 11 and status(rc, "R-TSS-01")["status"] == "not_modelled")
w = status(rc, "R-WX-01")
top = max(r["weather"]["legs"], key=lambda g: g["wind_bft"])["wind_bft"]
check(f"R-WX-01 uses the route weather: {w['status']} ({w['value']}; {w['limit']})",
      f"Beaufort {top}" in w["value"] and w["status"] == ("fail" if top > 6 else "warn" if top >= 5 else "pass"))
check(f"R-WX-02 / R-WX-03 checked with real gusts / waves: {status(rc, 'R-WX-02')['value']}; {status(rc, 'R-WX-03')['value']}",
      status(rc, "R-WX-02")["status"] != "not_modelled" and status(rc, "R-WX-03")["status"] != "not_modelled")
calm = rule_checker.check_route(r, gust_limit_bft=12)
check("a higher gust limit changes R-WX-02 (the setting feeds the rule)", status(calm, "R-WX-02")["status"] == "pass")
sq = status(rc, "R-UKC-02")
check(f"squat estimated (Barrass): {sq['value']}", "squat" in sq["value"] and sq["status"] in ("pass", "warn"))

r, rc, _ = checked((54.40, 11.00), (54.52, 11.60))             # Heiligenhafen -> Rødsand (Fehmarnsund)
row = status(rc, "R-COLREG-09")
check(f"Fehmarnsund: R-COLREG-09 warns ({row['value']})", row["status"] == "warn" and "Fehmarnsund" in row["value"])
check("... and R-BUF-01 / R-BUF-02 still pass (reduced clearance only inside the passage)",
      status(rc, "R-BUF-01")["status"] == "pass" and status(rc, "R-BUF-02")["status"] == "pass")
for air, want in [(14, "pass"), (22, "warn"), (25, "fail")]:
    row = status(rule_checker.check_route(r, air_draught_m=air), "R-AIR-01")
    check(f"air draught {air} m under the Fehmarnsund Bridge (about 23 m): {row['status']}", row["status"] == want, row)
row = status(rule_checker.check_route(r), "R-CONT-01")
check(f"ports of refuge per leg: {row['value']}", row["status"] in ("pass", "warn") and "nm to" in row["value"])

r, rc, _ = checked(KIEL, FEHMARN_W, zones="penalty", penalty_factor=0)
row = status(rc, "R-ZONE-02")
check(f"penalty mode, factor 0: R-ZONE-02 warns with the km inside ({row['value']})",
      row["status"] == "warn" and "km inside" in row["value"])
r, rc, _ = checked((54.55, 10.40), (54.60, 11.25), draft_m=11.0, margin_m=1.0)   # Kiel Bay -> Fehmarnbelt, deep water
check(f"draft 11 m: R-PILOT-01 warns ({status(rc, 'R-PILOT-01')['value']})", status(rc, "R-PILOT-01")["status"] == "warn")
check("UKC: min depth >= draft + margin -> R-UKC-01 pass", status(rc, "R-UKC-01")["status"] == "pass")

saved = rw.get_field
import tools
rw.get_field = lambda points: rw.dummy_field(points)
try:
    tools_source, tools.DATA_SOURCE = tools.DATA_SOURCE, "dummy"
    r, rc, _ = checked(KIEL, FEHMARN_W)
finally:
    tools.DATA_SOURCE = tools_source
    rw.get_field = saved
off = routing.route(*KIEL, *FEHMARN_W, weather="off")
rc_off = rule_checker.check_route(off, weather_error="The trip is outside the forecast")
check("departure outside the forecast: R-WX-04 warns with the reason, the other weather rules are not modelled",
      status(rc_off, "R-WX-04")["status"] == "warn" and "outside the forecast" in status(rc_off, "R-WX-04")["value"]
      and status(rc_off, "R-WX-01")["status"] == "not_modelled")
check("dummy data: gusts and waves are 'not modelled' (wind only), wind is checked",
      status(rc, "R-WX-02")["status"] == "not_modelled" and status(rc, "R-WX-03")["status"] == "not_modelled"
      and status(rc, "R-WX-01")["status"] != "not_modelled")

# --- Phase C: vessel profiles -------------------------------------------------------------
print("\n=== Phase C: vessel profiles (rules/vessel_profiles.yaml) ===")
profiles = rule_checker.load_profiles()
check(f"profiles: {sorted(profiles['profiles'])}, default {profiles['default']}",
      set(profiles["profiles"]) == {"small_sailboat", "motor_yacht", "coaster", "large_ship"})
sail = rule_checker.profile_settings("small_sailboat")
check("the default profile (small sailboat) equals the old default settings (nothing changes)",
      (sail["draft_m"], sail["margin_m"], sail["speed_kn"], sail["wind_limit_bft"], sail["waves_limit_m"])
      == (routing.DRAFT_M, routing.SAFETY_MARGIN_M, routing.DEFAULT_SPEED_KN, rw.ROUGH_WIND_BFT, rw.ROUGH_WAVES_M)
      and profiles["default"] == "small_sailboat")
big = rule_checker.profile_settings("large_ship")
check(f"UKC policy -> margin in code: large ship 15 % of {big['draft_m']} m, min 2 m = {big['margin_m']} m; "
      f"coaster {rule_checker.profile_settings('coaster')['margin_m']} m",
      big["margin_m"] == 2.0 and rule_checker.profile_settings("coaster")["margin_m"] == 1.0
      and rule_checker.ukc_margin({"type": "percent_of_draft", "percent": 15, "min_m": 2.0}, 20) == 3.0)
try:
    rule_checker.profile_settings("submarine")
    check("unknown profile: error", False)
except KeyError:
    check("unknown profile: error", True)
import voyage

ROUTE_KEYS = ("draft_m", "margin_m", "speed_kn", "wind_limit_bft", "waves_limit_m")


def profile_route(name, a, b, **extra):
    ps = rule_checker.profile_settings(name)
    settings = {**{k: ps[k] for k in ROUTE_KEYS}, "air_draught_m": ps["air_draught_m"],
                "gust_limit_bft": ps["gust_limit_bft"], "profile": name, "weather": "show", "departure": DEPARTURE,
                **extra}
    return voyage.checked_route(*a, *b, settings)


r = profile_route("large_ship", (54.55, 10.40), (54.60, 11.25))
rc = r["rule_check"]
check(f"large ship, Kiel Bay -> Fehmarnbelt: {r['distance_km']} km, min depth {r['min_depth_m']} m >= 14.5 m",
      r["min_depth_m"] >= 14.5 and status(rc, "R-UKC-01")["status"] == "pass")
check(f"... R-PILOT-01 warns (draft 12.5 m >= 11 m) and R-ROUTE-T warns (75,000 dwt)",
      status(rc, "R-PILOT-01")["status"] == "warn" and status(rc, "R-ROUTE-T")["status"] == "warn")
check(f"... squat uses the profile's block coefficient ({status(rc, 'R-UKC-02')['value']})",
      "Cb 0.82" in status(rc, "R-UKC-02")["value"])
small = profile_route("small_sailboat", (54.55, 10.40), (54.60, 11.25))["rule_check"]
check("small sailboat on the same trip: no pilot / Route T warning",
      status(small, "R-PILOT-01")["status"] == "pass" and status(small, "R-ROUTE-T")["status"] == "pass")
full = profile_route("coaster", (54.40, 11.00), (54.52, 11.60))
check(f"coaster ({full['required_depth_m']} m needed) can't use the Fehmarnsund: around Fehmarn "
      f"({full['distance_km']} km, via {full['via']})", "Fehmarnsund" not in full["via"])
# Draft adjusted to 2.0 m and 6 kn (needs 2.0 + squat 0.27 + 1.0 = 3.27 m): through the Fehmarnsund,
# under the bridge (about 23 m).
coaster = profile_route("coaster", (54.40, 11.00), (54.52, 11.60), draft_m=2.0, margin_m=1.0, speed_kn=6)
row = status(coaster["rule_check"], "R-AIR-01")
check(f"coaster, draft adjusted to 2.0 m, air draught 28 m through the Fehmarnsund: R-AIR-01 fails ({row['value']})",
      row["status"] == "fail" and "Fehmarnsund Bridge" in row["value"])
adjusted = profile_route("coaster", (54.40, 11.00), (54.52, 11.60), draft_m=2.0, margin_m=1.0, speed_kn=6,
                         air_draught_m=20)
check("the user can still adjust a profile value (air draught 20 m -> R-AIR-01 passes)",
      status(adjusted["rule_check"], "R-AIR-01")["status"] == "pass")
try:
    profile_route("large_ship", (54.45, 10.45), (54.45, 10.95))
    check("large ship in shallow water: no route", False, "a route was found")
except routing.RouteError as exc:
    check(f"large ship in shallow water: no route, explained ({str(exc)[:70]}…)", "too shallow" in str(exc), str(exc))

# --- Phase D: candidates, recommendation, passage plan ---------------------------------
print("\n=== Phase D: candidates, recommendation, passage plan ===")
settings = {"air_draught_m": 14, "gust_limit_bft": 8, "profile": "small_sailboat", "departure": DEPARTURE}
started = time.time()
events = list(voyage.candidates(54.40, 11.00, 54.52, 11.60, settings))
seconds = time.time() - started
progress = [e for e in events if e["type"] == "candidate"]
result = events[-1]
check(f"3 candidates computed with progress events ({[(e['index'], e['kind']) for e in progress]}) in {seconds:.1f} s (< 10 s)",
      [e["index"] for e in progress] == [1, 2, 3] and result["type"] == "result" and seconds < 10)
kinds = [c["kind"] for c in result["candidates"]]
dups = [e for e in progress if e.get("duplicate_of")]
check(f"duplicates removed: kept {kinds}, dropped {[(e['kind'], e['duplicate_of']) for e in dups]}",
      len(kinds) == len(set(kinds)) and len(kinds) + len(dups) + sum(bool(e.get("error")) for e in progress) == 3
      and all(not any(c["path"] == o["path"] for o in result["candidates"] if o is not c) for c in result["candidates"]))
rec = result["recommended"]
check(f"recommended: {rec} (ranking {result['ranking']})",
      rec is None or not next(c for c in result["candidates"] if c["kind"] == rec)["fails"])
check(f"the reasons compare the candidates: {result['reasons']}", len(result["reasons"]) == len(kinds) - 1)
safest = next((c for c in result["candidates"] if c["kind"] == "safest"), None)
check("the safest candidate avoids the Fehmarnsund Rule 9 warning (full buffers)",
      safest is not None and "R-COLREG-09" not in safest["warnings"])
fake = lambda kind, fails, warns, wind, km: {"kind": kind, "label": kind, "fails": fails, "warnings": warns,
                                             "worst_wind_bft": wind, "worst_waves_m": 0.5, "distance_km": km, "path": [kind]}
ranked = voyage.recommend([fake("a", ["R-WX-01"], [], 3, 10), fake("b", [], ["x", "y"], 5, 30),
                           fake("c", [], ["x"], 6, 40), fake("d", [], ["x"], 5, 50)])
check(f"score: a hard fail is never recommended; then fewer warnings, then lower wind ({ranked['ranking']})",
      ranked["recommended"] == "d" and ranked["ranking"] == ["d", "c", "b", "a"])
none_ok = voyage.recommend([fake("a", ["R-WX-01"], [], 3, 10), fake("b", ["R-WX-04"], [], 3, 20)])
check("every candidate fails: none recommended, the least bad is named",
      none_ok["recommended"] is None and none_ok["least_bad"] == "a")

plan = voyage.passage_plan(54.40, 11.00, 54.52, 11.60, rec or result["least_bad"], settings)
etas = [pd.Timestamp(w["eta_utc"]) for w in plan["waypoints"]]
check(f"passage plan: {len(plan['waypoints'])} waypoints with increasing ETAs, {len(plan['legs'])} legs",
      len(plan["waypoints"]) >= 2 and etas == sorted(etas) and etas[0] == pd.Timestamp(plan["departure_utc"]))
check("... weather, min UKC and a port of refuge per leg",
      all({"wind_bft", "min_ukc_m", "port_of_refuge"} <= set(g) and g["port_of_refuge"] in rule_checker.HARBOURS
          for g in plan["legs"]) and min(g["min_ukc_m"] for g in plan["legs"]) >= plan["min_ukc_m"] - 0.05)
check(f"... the rule-check summary ({plan['rule_summary']['counts']})", "counts" in plan["rule_summary"])
import xml.etree.ElementTree as ET
doc = voyage.gpx(plan)
root = ET.fromstring(doc)
ns = {"g": "http://www.topografix.com/GPX/1/1"}
pts = root.findall("g:rte/g:rtept", ns)
check(f"GPX 1.1: one route point per waypoint ({len(pts)}), with ETA times and 'NOT FOR NAVIGATION'",
      len(pts) == len(plan["waypoints"]) and pts[0].find("g:time", ns) is not None and "NOT FOR NAVIGATION" in doc)

# --- Fixes after testing -------------------------------------------------------------------
print("\n=== Fixes after testing: bridges, squat in the router, forecast end, candidate rows, errors ===")
from shapely.geometry import LineString as _Line
lines = rule_checker.bridge_lines()
air = rule_checker.rule("R-AIR-01")["parameters"]
check(f"bridge lines from OSM for all {len(air['bridges'])} rulebook bridges ({sorted(lines)})",
      set(lines) == {b["name"] for b in air["bridges"]})
row = status(rule_checker.check_route(routing.route(54.40, 11.00, 54.52, 11.60)), "R-AIR-01")
check(f"hand clearance marked in the rule check: {row['value']}",
      "Fehmarnsund Bridge" in row["value"] and "approximate - verify on chart" in row["value"])
old = lines["Storstrøm Bridge (old, 1937)"][0]
c = old.centroid
across = _Line([(c.x - 0.5, c.y + 0.3), (c.x + 0.5, c.y - 0.3)])          # a line crossing the old bridge
passed = rule_checker.bridges_passed(across, air)
check(f"a bridge without a known clearance is passed with source 'unknown' ({passed})",
      any(b["name"] == "Storstrøm Bridge (old, 1937)" and b["source"] == "unknown" for b in passed))

my = {**{k: rule_checker.profile_settings("motor_yacht")[k] for k in ROUTE_KEYS}, "profile": "motor_yacht"}
rs = voyage.router_settings(my)
check(f"motor yacht: router squat = 0.45 x 15^2 / 100 = {rs.get('squat_m')} m", rs.get("squat_m") == 1.01)
check("small sailboat: no squat in the router (old routes unchanged)",
      "squat_m" not in voyage.router_settings({"profile": "small_sailboat", "speed_kn": 6}))
r = voyage.checked_route(54.55, 10.40, 54.60, 11.25, {**my, "weather": "show", "departure": DEPARTURE})
check(f"... required depth = draft + squat + margin = {r['required_depth_m']} m; R-UKC-01: {status(r['rule_check'], 'R-UKC-01')['limit']}",
      r["required_depth_m"] == 3.51 and "squat 1.01 m" in status(r["rule_check"], "R-UKC-01")["limit"])

late = FIELD.times[-1] - pd.Timedelta(hours=2)
r = voyage.checked_route(54.55, 10.40, 54.60, 11.25, {"weather": "show", "departure": late})
w, row = r["weather"], status(r["rule_check"], "R-WX-04")
check(f"trip past the forecast end: {len(w['legs'])} legs with weather, {len(w['no_forecast_legs'])} without",
      w["legs"] and w["no_forecast_legs"] and all(g["status"] == "none" and g["wind_bft"] is None
                                                  for g in w["no_forecast_legs"]))
check(f"... R-WX-04 warns (not fails) and lists the legs: {row['value']}",
      row["status"] == "warn" and f"legs {w['no_forecast_legs'][0]['leg']}–" in row["value"])
plan = voyage.passage_plan(54.55, 10.40, 54.60, 11.25, "shortest", {"weather": "show", "departure": late})
check(f"... the passage plan lists the legs without forecast ({plan['no_forecast_legs']})",
      plan["no_forecast_legs"] == [g["leg"] for g in w["no_forecast_legs"]]
      and plan["legs"][-1]["status"] == "none" and len(plan["legs"]) == len(w["legs"]) + len(w["no_forecast_legs"]))
r = voyage.checked_route(54.55, 10.40, 54.60, 11.25, {"weather": "show", "departure": FIELD.times[-1] + pd.Timedelta(days=1)})
check("departure after the forecast: a route, every leg without forecast, wind 'not modelled'",
      not r["weather"]["legs"] and status(r["rule_check"], "R-WX-04")["status"] == "warn"
      and status(r["rule_check"], "R-WX-01")["status"] == "not_modelled")

result = list(voyage.candidates(54.40, 11.00, 54.52, 11.60, settings))[-1]
check(f"candidate rows: always all three in order ({[(x['kind'], x.get('duplicate_label') or x.get('error') or x['distance_km']) for x in result['rows']]})",
      [x["kind"] for x in result["rows"]] == ["shortest", "balanced", "safest"]
      and all(x.get("duplicate_label") or x.get("error") or x.get("distance_km") for x in result["rows"]))
real_checked = voyage.checked_route


def failing_safest(*args):
    if args[4].get("passages") is False:
        raise routing.RouteError("Start: no route with full buffers", "start", [54.4, 11.0], "Too close to land", "land_buffer")
    return real_checked(*args)


voyage.checked_route = failing_safest
try:
    result = list(voyage.candidates(54.40, 11.00, 54.52, 11.60, settings))[-1]
finally:
    voyage.checked_route = real_checked
safe_row = result["rows"][2]
check(f"a candidate that can't be calculated keeps its row and says why ({safe_row})",
      safe_row["kind"] == "safest" and safe_row["short"] == "Too close to land" and "full buffers" in safe_row["error"])
dup_rows = [x for x in result["rows"] if x.get("duplicate_of")]
check("a duplicate names the route it equals ('same route as Shortest')",
      all(x["duplicate_label"] in ("Shortest", "Balanced", "Safest") for x in dup_rows))

ls = {k: rule_checker.profile_settings("large_ship")[k] for k in ROUTE_KEYS}
try:
    voyage.checked_route(54.45, 10.30, 54.60, 10.20, {**ls, "profile": "large_ship", "weather": "off"})
    check("large ship, shallow start: error", False, "a route was found")
except routing.RouteError as exc:
    pr = exc.problem()
    check(f"structured error for the page: {pr['which']}, {pr['kind']}, at {pr['at']}, '{pr['short']}'",
          pr["which"] == "start" and pr["kind"] == "shallow" and pr["at"] == [54.45, 10.3]
          and pr["short"].startswith("Too shallow for Large ship") and "m here, needs" in pr["short"])
import app
q = app.route_query(from_lat=54.45, from_lon=10.30, to_lat=54.60, to_lon=10.20, speed_kn=14, draft_m=12.5,
                    margin_m=2.0, zones="block", land_buffer_m=300, zone_buffer_m=500, passage_land_buffer_m=50,
                    passage_zone_buffer_m=100, penalty_factor=10, wind_farms="block", turbine_radius_m=500,
                    weather="off", departure=None, wind_limit_bft=9, waves_limit_m=6, gust_limit_bft=11,
                    air_draught_m=50, profile="large_ship")
resp = app.sea_route(q)
body = json.loads(resp.body)
check(f"/api/route: 400 with 'detail' and 'problem' ({body['problem']['short']})",
      resp.status_code == 400 and body["detail"].startswith("Start:") and body["problem"]["which"] == "start")
check(f"the API accepts the large ship's 12.5 m draft (limit {routing.MAX_DRAFT_M:g} m)", routing.MAX_DRAFT_M >= 12.5)

print("\n=== Dummy time: the app's default departure follows the dummy calendar ===")
import chat
frozen = clock._frozen
clock.freeze(None)                                     # the app's clock (not frozen)
try:
    first = tools.forecast_dates()[0]                  # dummy data: a fixed 3-day period
    info = app.current_time()
    expected = pd.Timestamp(first, tz="UTC") + pd.Timedelta(hours=clock.DUMMY_START_HOURS)
    check(f"dummy data: /api/clock = {info['now_utc']}, '{info['label']}'",
          info["dummy"] and pd.Timestamp(info["now_utc"]) == expected
          and info["label"] == f"Dummy time: {expected:%a %d %b %H:%M} UTC (synthetic scenario)")
    check("the same clock as the chat's 'today' and the router's default departure",
          chat.today_date() == first and routing._departure(None) == expected)
    tools.DATA_SOURCE = "real"
    real = app.current_time()
    check("real data: the real now, no dummy label",
          not real["dummy"] and real["label"] is None
          and abs(pd.Timestamp(real["now_utc"]) - pd.Timestamp.now(tz="UTC")) < pd.Timedelta(minutes=2))
finally:
    tools.DATA_SOURCE = "dummy"
    clock._frozen = frozen

print(f"\n{'all passed' if not failures else f'{failures} FAILED'}")
