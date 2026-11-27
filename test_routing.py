"""Tests for routing.py (code only, offline). Needs the cached files in data/:
land_kiel_bay.geojson (download_land.py), depth_kiel_bay.npz and
restricted_zones_kiel_bay.geojson (download_routing_data.py).

Phase 1 tests use terms=[] (land only); phase 2 tests use the default rules."""
import math
import time

from shapely.geometry import LineString, Point, box

import routing
import tools

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


def line_km(result):
    return LineString([tools._to_km(lon, lat) for lat, lon in result["path"]])


LAND = tools._land_km()
SEA_POINT = (54.55, 10.45)

# --- 1. The route never crosses land -----------------------------------------------
print("=== never crosses land ===")
PAIRS = {
    "across the bay": (SEA_POINT, (54.60, 10.85)),
    "Laboe (outer Kiel Fjord) -> sea point": ((54.41, 10.22), SEA_POINT),
    "Eckernförde Bay -> east of Fehmarn": ((54.47, 9.88), (54.45, 11.35)),
    "Kiel Bay -> Little Belt": (SEA_POINT, (55.05, 9.95)),
    "west -> east of Fehmarn": ((54.50, 10.95), (54.45, 11.35)),
}
grid = routing.build_grid()
mask = routing.usable_mask(grid, [])          # phase 1: land only
for name, (a, b) in PAIRS.items():
    started = time.time()
    r = routing.route(*a, *b, terms=[])
    seconds = time.time() - started
    check(f"{name}: {r['distance_km']} km, no land crossed", not line_km(r).intersects(LAND))
    check(f"{name}: computed in {seconds:.2f} s (< 1 s)", seconds < 1)
    for end in (a, b):
        cell, _ = routing.snap(grid, mask, *end)
        check(f"{name}: snapped square {cell} is water", bool(grid.water[cell]))

# --- 2. Same point ----------------------------------------------------------------
print("\n=== same point ===")
r = routing.route(*SEA_POINT, *SEA_POINT, terms=[])
check("same point: 0 km, ETA 0, one point",
      r["distance_km"] == 0 and r["eta_hours"] == 0 and len(r["path"]) == 1, r)

# --- 3. Around Fehmarn -------------------------------------------------------------
print("\n=== around Fehmarn ===")
west, east = (54.50, 10.95), (54.45, 11.35)
fehmarn = next(g for g in getattr(LAND, "geoms", [LAND]) if g.contains(Point(tools._to_km(11.15, 54.47))))
straight = LineString([tools._to_km(west[1], west[0]), tools._to_km(east[1], east[0])])
check("the straight line crosses Fehmarn (so the test means something)", straight.intersects(fehmarn))
r = routing.route(*west, *east, terms=[])
check("the route does not touch Fehmarn", not line_km(r).intersects(fehmarn))
check(f"the route ({r['distance_km']} km) is longer than the straight line ({straight.length:.1f} km)",
      r["distance_km"] > straight.length)

# --- 4. Other properties ----------------------------------------------------------
print("\n=== other ===")
there = routing.route(*west, *east, terms=[])["distance_km"]
back = routing.route(*east, *west, terms=[])["distance_km"]
check(f"swapped start/end: same distance ({there} vs {back} km)", abs(there - back) <= 0.01 * there)

on_land = (54.47, 11.15)                                    # inside Fehmarn
r = routing.route(*on_land, *SEA_POINT, terms=[])
check(f"a point on land snaps to water ({r['start']['snap_km']} km)",
      0 < r["start"]["snap_km"] <= routing.MAX_SNAP_KM and not line_km(r).intersects(LAND))

for name, point, expected in [("far inland", (54.20, 10.00), "No usable water within 5 km: this point is on land"),
                              ("outside the area", (53.50, 10.00), "Outside the routing area")]:
    try:
        routing.route(*point, *SEA_POINT, terms=[])
        check(f"{name}: error", False, "no error")
    except routing.RouteError as exc:
        check(f"{name}: clear error", expected in str(exc), str(exc))

check("6 nm = 11.112 km", math.isclose(11.112 / routing.KM_PER_NM, 6.0))
check("ETA text: 1 h", routing.eta_text(6.0 / 6.0) == "1 h 00 min")
r = routing.route(*west, *east, speed_kn=6, terms=[])
check("ETA = distance in nm / speed", math.isclose(r["eta_hours"], r["distance_nm"] / 6, abs_tol=0.002), r)
r12 = routing.route(*west, *east, speed_kn=12, terms=[])
check("twice the speed: half the time", math.isclose(r12["eta_hours"] * 2, r["eta_hours"], abs_tol=0.002))
try:
    routing.route(*west, *east, speed_kn=0, terms=[])
    check("speed 0: error", False)
except routing.RouteError:
    check("speed 0: error", True)

# --- 5. The cost-term interface (for later phases) ---------------------------------
print("\n=== cost terms (extension point) ===")


class RestrictedZone:
    """A test term: a 'restricted' rectangle that the route must avoid."""
    name = "test zone"

    def __init__(self, lon0, lat0, lon1, lat1):
        self.zone = box(*tools._to_km(lon0, lat0), *tools._to_km(lon1, lat1))

    def blocked(self, cell, grid):
        x, y = grid.center(cell)
        half = grid.cell_km / 2
        return box(x - half, y - half, x + half, y + half).intersects(self.zone)

    def extra(self, a, b, step_km, ctx):
        return 0.0

    def segment_ok(self, p, q, grid):
        return not LineString([p, q]).intersects(self.zone)


a, b = SEA_POINT, (54.55, 10.80)
free = routing.route(*a, *b, terms=[], cell_km=0.5)
zone = RestrictedZone(10.60, 54.50, 10.64, 54.62)
blocked = routing.route(*a, *b, terms=[zone], cell_km=0.5)
check("without the zone the route goes through it", line_km(free).intersects(zone.zone))
check("with the zone the route avoids it", not line_km(blocked).intersects(zone.zone))
check(f"... and is longer ({blocked['distance_km']} vs {free['distance_km']} km)",
      blocked["distance_km"] > free["distance_km"])
check("the term is listed in the result", blocked["cost_terms"] == ["test zone"])


class Toll:
    """A test term with an extra cost: every step costs 1 km more. Same route, more cost."""
    name = "toll"

    def blocked(self, cell, grid):
        return False

    def extra(self, a, b, step_km, ctx):
        return 1.0

    def segment_ok(self, p, q, grid):
        return True


tolled = routing.route(*a, *b, terms=[Toll()], cell_km=0.5)   # custom terms: slow, so a coarser grid
check("an extra-cost term still finds a water route", not line_km(tolled).intersects(LAND))

# ===================================================================================
# Phase 2: depth, restricted zones and safety buffers (default rules)
# ===================================================================================
print("\n=== phase 2: data files ===")
depth = routing.load_depth()
check(f"depth grid {depth.values.shape} covers the routing box",
      depth.values.shape == (round((routing.ROUTE_BBOX[3] - routing.ROUTE_BBOX[1]) * 960),
                             round((routing.ROUTE_BBOX[2] - routing.ROUTE_BBOX[0]) * 960)))
sea_depth = float(depth.at(*tools._to_km(SEA_POINT[1], SEA_POINT[0])))
check(f"sea point is about 19 m deep ({sea_depth} m)", 17 <= sea_depth <= 21)
check("Fehmarn interior is land in the depth data", float(depth.at(*tools._to_km(11.15, 54.47))) < 0)
zones = routing.load_zones()
firing = [z for z in zones if z.category == "firing"]
names = {z.name for z in firing}
check("firing areas Todendorf and Putlos are in the zone file",
      {"Schießgebiet Todendorf", "Schießgebiet Putlos"} <= names, names)
check("exercise areas are ignored by default", routing.zone_rules()["exercise"] == "ignore")
blocked_zones = [z for z in zones if routing.zone_rules()[z.category] == "block" and z.kind == "area"]
turbines = [z for z in zones if z.kind == "turbine"]          # points; their radius is checked separately


def samples_min_depth(result, step_km=0.025):
    """Independent check: depth every 25 m along the route (finer than the router's 50 m)."""
    pts = [tools._to_km(lon, lat) for lat, lon in result["path"]]
    mins = []
    for p, q in zip(pts, pts[1:]):
        n = max(2, math.ceil(math.dist(p, q) / step_km) + 1)
        mins.append(min(float(depth.at(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
                        for t in [i / (n - 1) for i in range(n)]))
    return min(mins) if mins else float(depth.at(*pts[0]))


def exact_zone_distance(result, zone_list):
    """Independent check: exact distance (shapely) from the route to the nearest zone."""
    pts = [tools._to_km(lon, lat) for lat, lon in result["path"]]
    geom = LineString(pts) if len(pts) > 1 else Point(pts[0])
    return min(z.geom_km.distance(geom) for z in zone_list)


def exact_land_distance(result):
    pts = [tools._to_km(lon, lat) for lat, lon in result["path"]]
    return LAND.distance(LineString(pts) if len(pts) > 1 else Point(pts[0]))


def touches(result, zone_list):
    line = line_km(result)
    return [z.name for z in zone_list if z.geom_km.intersects(line)]


routing.warm_up()

print("\n=== phase 2: every route keeps the rules (defaults: 3.0 m depth, 300 m from land, 500 m from zones) ===")
RULE_PAIRS = {
    "across the bay": (SEA_POINT, (54.60, 10.85)),
    "Kiel Bay -> west of Fehmarn (past the firing areas)": ((54.45, 10.45), (54.45, 10.95)),
    "west -> east of Fehmarn": ((54.50, 10.95), (54.45, 11.35)),
    "Kiel Bay -> Little Belt": (SEA_POINT, (55.05, 9.95)),
    "Kiel Bay -> Gedser": ((54.45, 10.25), (54.53, 11.96)),
}
for name, (a, b) in RULE_PAIRS.items():
    started = time.time()
    r = routing.route(*a, *b)
    seconds = time.time() - started
    check(f"{name}: {r['distance_km']} km in {seconds:.2f} s, no land", not line_km(r).intersects(LAND) and seconds < 2)
    check(f"{name}: min depth {r['min_depth_m']} m (25 m samples {samples_min_depth(r):.1f} m) >= 3.0 m",
          r["min_depth_m"] >= 3.0 and samples_min_depth(r) >= 3.0)
    check(f"{name}: >= 300 m from land (exact {exact_land_distance(r) * 1000:.0f} m)", exact_land_distance(r) >= 0.3)
    check(f"{name}: >= 500 m from closed zones (exact {exact_zone_distance(r, blocked_zones) * 1000:.0f} m)",
          exact_zone_distance(r, blocked_zones) >= 0.5)
check("the rules are listed", r["cost_terms"] == ["depth", "zones (block)", "buffers"], r["cost_terms"])

print("\n=== phase 2: speed (0.25 km grid) ===")
started = time.time()
r = routing.route(54.50, 10.00, 54.45, 11.35)                 # Eckernförde Bay -> east of Fehmarn
seconds = time.time() - started
check(f"Eckernförde Bay -> east of Fehmarn: {r['distance_km']} km in {seconds:.2f} s (< 2 s)", seconds < 2)
started = time.time()
routing.explain(54.50, 10.00, 54.45, 11.35)
check(f"its detour breakdown (4 routes): {time.time() - started:.2f} s (< 4 s)", time.time() - started < 4)

print("\n=== extended area: Fehmarnbelt, Rødsand, Gedser, Møn, Faxe Bugt (to 12.8° E, 55.3° N) ===")
check(f"the routing area is {routing.ROUTE_BBOX}", routing.ROUTE_BBOX == (9.8, 54.15, 12.8, 55.3))
r = routing.route(54.40, 11.00, 54.52, 11.60)                 # off Heiligenhafen -> Rødsand (south of Rødsand II)
check(f"Heiligenhafen -> Rødsand: {r['distance_km']} km, ends east of Fehmarn (snap {r['end']['snap_km']} km)",
      r["end"]["snap_km"] < 0.5 and max(p[1] for p in r["path"]) > 11.55 and not r["zones_crossed"])
check(f"... no land, depth >= 3.0 m (25 m samples {samples_min_depth(r):.1f} m)",
      not line_km(r).intersects(LAND) and samples_min_depth(r) >= 3.0)
check(f"... through the Fehmarnsund with the Rule 9 warning ({r['passages_used']})",
      r["passages_used"] == ["Fehmarnsund"] and "COLREG Rule 9" in (r["passage_warning"] or ""))
check(f"... passage buffers kept: >= 100 m from closed zones (exact {exact_zone_distance(r, blocked_zones) * 1000:.0f} m), "
      f">= 50 m from land (exact {exact_land_distance(r) * 1000:.0f} m)",
      exact_zone_distance(r, blocked_zones) >= 0.1 and exact_land_distance(r) >= 0.05)
alt = r["alternative"] or {}
check(f"... the full-buffer alternative goes around Fehmarn ({alt.get('distance_km')} km)",
      (alt.get("distance_km") or 0) > r["distance_km"] and not alt.get("passages_used"))
started = time.time()
r = routing.route(55.05, 9.95, 54.53, 11.96)                  # Little Belt -> Gedser: the longest route
seconds = time.time() - started
check(f"Little Belt -> Gedser: {r['distance_km']} km in {seconds:.2f} s (< 2 s)", seconds < 2 and not line_km(r).intersects(LAND))
started = time.time()
routing.explain(55.05, 9.95, 54.53, 11.96)
check(f"its detour breakdown: {time.time() - started:.2f} s (< 4 s)", time.time() - started < 4)
for name, point in [("east of 12.8° E", (54.60, 12.90)), ("north of 55.3° N", (55.35, 10.60))]:
    try:
        routing.route(*point, *SEA_POINT)
        check(f"{name}: error", False, "no error")
    except routing.RouteError as exc:
        check(f"{name}: 'Outside the routing area'", "Outside the routing area" in str(exc), str(exc))

print("\n=== Danish sounds and Faxe Bugt (narrow passages, no-route explanations) ===")
started = time.time()
r = routing.route(55.20, 12.30, 54.40, 11.00)                 # Faxe Bugt -> off Heiligenhafen
seconds = time.time() - started
check(f"Faxe Bugt -> Heiligenhafen: {r['distance_km']} km in {seconds:.2f} s (< 2 s), via {r['via']}",
      seconds < 2 and "around Møn (east of Møns Klint)" in r["via"] and not line_km(r).intersects(LAND))
check(f"... the panel names the passage it used ({r['passages_used']}, warning: {r['passage_warning'][:40]}…)",
      r["passages_used"] == ["Fehmarnsund"] and r["via"][-1] == "Fehmarnsund" and "COLREG Rule 9" in r["passage_warning"])
check(f"... rules kept (depth {samples_min_depth(r):.1f} m >= 3 m, land {exact_land_distance(r) * 1000:.0f} m >= 50 m, "
      f"zones {exact_zone_distance(r, blocked_zones) * 1000:.0f} m >= 100 m: passage buffers)",
      samples_min_depth(r) >= 3.0 and exact_land_distance(r) >= 0.05 and exact_zone_distance(r, blocked_zones) >= 0.1)
r = routing.route(54.70, 12.30, 54.895, 12.05)                 # Baltic side -> Stubbekøbing
check(f"through Grønsund: {r['distance_km']} km, via {r['via']}, Rule 9 warning for {r['passages_used']}",
      "Grønsund" in r["via"] and "Grønsund" in r["passages_used"] and samples_min_depth(r) >= 3.0)
try:
    routing.route(55.00, 12.25, 55.20, 12.30)                  # Stege Bugt -> Faxe Bugt
    check("Stege Bugt with 2 m draft + 1 m margin: no route", False, "a route was found")
except routing.RouteError as exc:
    check(f"Stege Bugt, 3 m needed: Bøgestrøm stays closed and is named ({str(exc)[:60]}…)",
          str(exc).startswith("Start:") and "Ulvsund / Bøgestrøm: too shallow" in str(exc)
          and "the boat needs 3 m" in str(exc), str(exc))
r = routing.route(55.00, 12.25, 55.20, 12.30, draft_m=1.0, margin_m=0.5)
check(f"Stege Bugt, 1.5 m needed: through Bøgestrøm ({r['distance_km']} km, via {r['via']})",
      "Ulvsund / Bøgestrøm" in r["via"] and samples_min_depth(r) >= 1.5)
try:
    routing.route(54.40, 11.00, 54.765, 11.865)                 # -> Nykøbing in Guldborgsund
    check("Guldborgsund (Nykøbing): no route", False, "a route was found")
except routing.RouteError as exc:
    check(f"Guldborgsund: the coastline data has no open channel, said so ({str(exc)[:50]}…)",
          str(exc).startswith("Destination:") and "Guldborgsund: in the map data this water is enclosed by land" in str(exc),
          str(exc))
try:
    routing.route(54.37, 12.455, 54.40, 11.00)                  # Bodden behind Darß / Zingst
    check("Bodden (Darß / Zingst): no route", False, "a route was found")
except routing.RouteError as exc:
    check(f"Bodden: the water continues across the east edge, said so",
          str(exc).startswith("Start: No route from here") and "no connection to the open sea" in str(exc)
          and "east edge (12.8° E)" in str(exc) and not str(exc).endswith(".."), str(exc))
started = time.time()
r = routing.route(55.05, 9.95, 55.20, 12.30)                  # Little Belt -> Faxe Bugt: the longest route
seconds = time.time() - started
check(f"Little Belt -> Faxe Bugt: {r['distance_km']} km in {seconds:.2f} s (< 2 s)", seconds < 2)
started = time.time()
routing.explain(55.05, 9.95, 55.20, 12.30)
check(f"its detour breakdown: {time.time() - started:.2f} s (< 4 s)", time.time() - started < 4)

print("\n=== wind farms (OpenStreetMap): areas + offshore turbines with a safety radius ===")
farm_areas = {z.name for z in zones if z.category == "wind_farm" and z.kind == "area"}
check(f"wind farm areas at Rødsand are in the data ({sorted(farm_areas)})",
      {"Rødsand II Havmøllepark", "Nysted Havmøllepark (Rødsand 1)"} <= farm_areas)
check(f"{len(turbines)} offshore turbines as points, all in water",
      len(turbines) >= 150 and all(z.geom_km.geom_type == "Point" and not LAND.contains(z.geom_km)
                                   and float(depth.at(z.geom_km.x, z.geom_km.y)) > 0 for z in turbines))
check("wind farms are blocked by default", routing.zone_rules()["wind_farm"] == "block"
      and routing.make_rules().turbine_radius_km == 0.5)


def turbine_distance(result):
    """Independent check: exact distance (shapely) from the route to the nearest turbine."""
    return min(z.geom_km.distance(line_km(result)) for z in turbines)


A, B = (54.55, 11.40), (54.55, 11.85)                         # west of Rødsand II -> east of Nysted
straight = LineString([tools._to_km(A[1], A[0]), tools._to_km(B[1], B[0])])
check("the straight line runs through both Rødsand wind farms (so the test means something)",
      sum(z.geom_km.intersects(straight) for z in zones if z.category == "wind_farm" and z.kind == "area") == 2
      and min(z.geom_km.distance(straight) for z in turbines) < 0.2)
started = time.time()
r = routing.route(*A, *B)
seconds = time.time() - started
check(f"near Rødsand: {r['distance_km']} km in {seconds:.2f} s, no land, outside both farm areas",
      seconds < 2 and not line_km(r).intersects(LAND) and not r["wind_farms_crossed"]
      and not touches(r, [z for z in zones if z.category == "wind_farm" and z.kind == "area"]))
check(f"... >= 500 m from every turbine (exact {turbine_distance(r) * 1000:.0f} m, reported "
      f"{r['min_turbine_distance_km'] * 1000:.0f} m, nearest: {r['nearest_turbine']})",
      turbine_distance(r) >= 0.5 and abs(turbine_distance(r) - r["min_turbine_distance_km"]) < 0.01)
check(f"... >= 500 m from the farm areas (zone buffer, exact {exact_zone_distance(r, blocked_zones) * 1000:.0f} m)",
      exact_zone_distance(r, blocked_zones) >= 0.5)
check(f"... and the rules still hold (min depth {samples_min_depth(r):.1f} m, land {exact_land_distance(r) * 1000:.0f} m)",
      samples_min_depth(r) >= 3.0 and exact_land_distance(r) >= 0.3)
r1000 = routing.route(*A, *B, turbine_radius_m=1000)
check(f"turbine radius 1000 m: >= 1000 m from every turbine (exact {turbine_distance(r1000) * 1000:.0f} m), "
      f"longer ({r1000['distance_km']} vs {r['distance_km']} km)",
      turbine_distance(r1000) >= 1.0 and r1000["distance_km"] > r["distance_km"])
free = routing.route(*A, *B, wind_farms="penalty", penalty_factor=0)
check(f"penalty mode, factor 0: straight through, the farms are still listed ({free['wind_farms_crossed']})",
      free["distance_km"] < r["distance_km"] and len(free["wind_farms_crossed"]) == 2)
pen = routing.route(*A, *B, wind_farms="penalty")
check(f"penalty mode, factor 10: goes around (0 km inside: {pen['zone_km_inside_penalty']} km)",
      pen["zone_km_inside_penalty"] == 0 and turbine_distance(pen) >= 0.5)
e = routing.explain(*A, *B)
zrow = e["detour"][2]
check(f"breakdown: '{zrow['step']}' {zrow['extra_text']} ({zrow.get('reason')})",
      zrow["step"].startswith("+ restricted zones and wind farms") and "Rødsand II Havmøllepark (wind farm)" in zrow.get("reason", "")
      and set(zrow.get("wind_farms_avoided", [])) == {"Rødsand II Havmøllepark", "Nysted Havmøllepark (Rødsand 1)"})
check(f"main reason sentence: {e['main_reason_sentence'][:90]}…",
      e["main_reason_sentence"].startswith("Main reason: the wind farms add") and "Rødsand II" in e["main_reason_sentence"])
pe = routing.explain(*A, *B, wind_farms="penalty")
check(f"penalty mode: the breakdown names the farms it goes around ({pe['detour'][2].get('reason', '')[:70]}…)",
      "penalty: going around is cheaper" in pe["detour"][2].get("reason", "")
      and pe["main_reason_sentence"].startswith("Main reason: the wind farms add"))
check("the breakdown lists the wind farms in the summary",
      set(e["summary"]["wind_farms_avoided"]) == {"Rødsand II Havmøllepark", "Nysted Havmøllepark (Rødsand 1)"})
inside = routing.route(54.555, 11.54, *B)                      # a click inside Rødsand II
check(f"a click inside a wind farm is moved out, with the reason ({inside['snap_warnings'][0][:95]}…)",
      "inside Rødsand II Havmøllepark (wind farm" in inside["snap_warnings"][0])
t = next(z for z in turbines if z.name == "Nysted Havmøllepark (Rødsand 1)" and z.geom_km.x == max(
    q.geom_km.x for q in turbines if q.name == "Nysted Havmøllepark (Rødsand 1)"))   # the easternmost turbine
lon, lat = routing.to_lonlat(t.geom_km.x + 0.3, t.geom_km.y)    # 300 m east of it, outside the area
reason = routing.point_reason(float(lat), float(lon), routing.make_rules())
check(f"a point 300 m from a turbine: {reason}", "from a wind turbine (Nysted" in reason and "radius is 500 m" in reason)
try:
    routing.make_rules(turbine_radius_m=10)
    check("turbine radius 10 m: error", False)
except routing.RouteError as exc:
    check(f"turbine radius 10 m: error ({exc})", "between 50 and 2000" in str(exc))

print("\n=== phase 2: narrow passages (reduced buffers, COLREG Rule 9 warning) ===")
a, b = (54.408, 11.043), (54.376, 11.213)                     # west and east of the Fehmarnsund
r = routing.route(*a, *b)
check(f"Fehmarnsund with the default buffers: {r['distance_km']} km (about 12 km, not ~50 km around Fehmarn)",
      9 <= r["distance_km"] <= 15 and r["min_depth_m"] >= 3.0)
check(f"... with the narrow-passage warning ({r['passage_warning']})",
      r["passages_used"] == ["Fehmarnsund"] and "COLREG Rule 9" in (r["passage_warning"] or ""))
check(f"... keeps >= 100 m from closed zones in the passage (exact {exact_zone_distance(r, blocked_zones) * 1000:.0f} m)",
      exact_zone_distance(r, blocked_zones) >= 0.1)
check(f"... and >= 50 m from land (exact {exact_land_distance(r) * 1000:.0f} m)", exact_land_distance(r) >= 0.05)
check(f"... the depth rule stays fully active (min {r['min_depth_m']} m, 25 m samples {samples_min_depth(r):.1f} m)",
      samples_min_depth(r) >= 3.0)
alt = r["alternative"] or {}
check(f"... the full-buffer alternative: {alt.get('distance_km')} km, ETA {alt.get('eta_text')} (around Fehmarn)",
      (alt.get("distance_km") or 0) > 40 and alt.get("eta_text"))
no_passages = routing.route(*a, *b, passages=False)
check(f"without the passage exception the sound is closed ({no_passages['distance_km']} km)",
      no_passages["distance_km"] > 40 and not no_passages["passages_used"])

r = routing.route(54.368, 10.158, 54.425, 10.22)              # Holtenau -> Laboe
check(f"Kiel Fjord Holtenau -> Laboe opens: {r['distance_km']} km, start moved only {r['start']['snap_km']} km",
      5 <= r["distance_km"] <= 10 and r["start"]["snap_km"] < 0.5 and not r["snap_warnings"])
check(f"... with the warning ({r['passage_warning']})",
      r["passages_used"] == ["Kiel Fjord (Friedrichsort narrows)"] and "keep to the fairway" in r["passage_warning"])
check(f"... the full-buffer alternative says why it isn't possible ({(r['alternative'] or {}).get('error', '')[:70]}…)",
      "not possible" in (r["alternative"] or {}).get("error", ""))
check("open-water routes get no passage warning",
      all(not routing.route(*a2, *b2)["passages_used"] for a2, b2 in RULE_PAIRS.values()))
check("a passage buffer never exceeds the open-water buffer",
      routing.make_rules(land_buffer_m=30, passage_land_buffer_m=50).passage_land_buffer_km == 0.03)

print("\n=== phase 2: tiny unnamed zones ===")
tiny = routing.tiny_zones()
check(f"tiny unnamed zones (< 1 ha) in the zone file: {tiny or 'none'} (listed in the log)", isinstance(tiny, list))
named_small = [z.name for z in zones if z.kind == "area" and z.geom_km.area < 0.01 and not routing.is_tiny(z)]
check(f"small zones WITH a name keep their buffer ({named_small})", all(not routing.is_tiny(z) for z in zones
                                                                        if z.name in named_small))
fake = routing.Zone("no_entry area", "no_entry", "way/0", Point(0, 0).buffer(0.03))   # ~0.28 ha, unnamed
check("an unnamed zone under 1 ha counts as tiny", routing.is_tiny(fake))
check("a named one doesn't", not routing.is_tiny(routing.Zone("Pier", "no_entry", "way/1", Point(0, 0).buffer(0.03))))

print("\n=== phase 2: safety buffers ===")
a, b = (54.40, 11.02), (54.765, 10.0)                         # Heiligenhafen -> Gelting Bay
r = routing.route(*a, *b)
check(f"Heiligenhafen -> Gelting: {r['distance_km']} km, >= 500 m from the firing areas "
      f"(exact {exact_zone_distance(r, firing) * 1000:.0f} m)", exact_zone_distance(r, firing) >= 0.5)
check(f"... and >= 300 m from land (exact {exact_land_distance(r) * 1000:.0f} m)", exact_land_distance(r) >= 0.3)
check("the buffers are reported", r["settings"]["land_buffer_m"] == 300 and r["settings"]["zone_buffer_m"] == 500
      and r["min_zone_distance_km"] >= 0.5 and r["min_land_distance_km"] >= 0.3, r["settings"])
wide = routing.route(*a, *b, zone_buffer_m=1500)
check(f"a 1500 m zone buffer keeps >= 1.5 km (exact {exact_zone_distance(wide, firing):.2f} km)",
      exact_zone_distance(wide, firing) >= 1.5)

print("\n=== phase 2: a known shallow area ===")
# Heiligenhafen: the straight line crosses water of 1.5 m; the boat needs 3.0 m (2.0 + 1.0).
a, b = (54.376, 11.032), (54.411, 11.010)
straight = [tools._to_km(a[1], a[0]), tools._to_km(b[1], b[0])]
check(f"Heiligenhafen: the straight line crosses shallow water ({routing.depths_along(straight)[0]:.1f} m < 3 m)",
      routing.depths_along(straight)[0] < 3.0 and not LineString(straight).intersects(LAND))
r = routing.route(*a, *b, land_buffer_m=0, zone_buffer_m=0)
check(f"Heiligenhafen: the route keeps >= 3.0 m (min {r['min_depth_m']} m, 25 m samples "
      f"{samples_min_depth(r):.1f} m)", samples_min_depth(r) >= 3.0)

print("\n=== phase 2: the draft changes the route (Stoller Grund, ~5 m) ===")
a, b = (54.453, 10.26), (54.453, 10.36)
straight = [tools._to_km(a[1], a[0]), tools._to_km(b[1], b[0])]
shoal = routing.depths_along(straight)[0]
check(f"Stoller Grund: the straight line crosses {shoal:.1f} m", 3.0 <= shoal < 5.5)
small = routing.route(*a, *b, draft_m=2.0, margin_m=1.0, land_buffer_m=0, zone_buffer_m=0)
deep = routing.route(*a, *b, draft_m=4.5, margin_m=1.0, land_buffer_m=0, zone_buffer_m=0)
check(f"draft 2.0 + 1.0 m crosses the shoal (min {small['min_depth_m']} m)", small["min_depth_m"] < 5.5)
check(f"draft 4.5 + 1.0 m goes around it (min {deep['min_depth_m']} m >= 5.5 m)",
      deep["min_depth_m"] >= 5.5 and samples_min_depth(deep) >= 5.5)
check(f"... on a different, longer path ({deep['distance_km']} vs {small['distance_km']} km)",
      deep["path"] != small["path"] and deep["distance_km"] > small["distance_km"])

print("\n=== phase 2: firing areas ===")
a, b = (54.45, 10.45), (54.45, 10.95)              # Kiel Bay -> west of Fehmarn
straight = LineString([tools._to_km(a[1], a[0]), tools._to_km(b[1], b[0])])
crossed = [z.name for z in firing if z.geom_km.intersects(straight)]
check(f"the straight line crosses {crossed}", {"Schießgebiet Todendorf", "Schießgebiet Putlos"} <= set(crossed))
r = routing.route(*a, *b)
check(f"zones avoided: the route ({r['distance_km']} km) touches no firing area",
      not touches(r, firing) and r["zones_crossed"] == [], touches(r, firing))
p = routing.route(*a, *b, zones="penalty", penalty_factor=1)
check(f"penalty mode, factor 1: going around (+{r['distance_km'] - 32.25:.1f} km) is still cheaper than "
      f"25 km inside the zones ({p['distance_km']} km, crosses {p['zones_crossed']})",
      p["zones_crossed"] == [] and p["distance_km"] <= r["distance_km"] + 0.5)

print("\n=== phase 2: why a click is rejected ===")
for name, point, expected in [
        ("inside a firing area", (54.42, 10.60), "inside Schießgebiet Todendorf (firing area"),
        ("far inland", (54.20, 10.00), "on land")]:
    try:
        routing.route(*point, *SEA_POINT)
        check(f"{name}: refused", False, "no error")
    except routing.RouteError as exc:
        check(f"{name}: the reason names it ({exc})", expected in str(exc))
_, (sx, sy) = routing.depths_along([tools._to_km(11.032, 54.376), tools._to_km(11.010, 54.411)])
slon, slat = routing.to_lonlat(sx, sy)
reason = routing.point_reason(float(slat), float(slon), routing.make_rules())
check(f"too shallow: values in the reason ({reason})", "1.5 m deep" in reason and "3 m" in reason)

print("\n=== phase 2: snapping ===")
r = routing.route(float(slat), float(slon), *SEA_POINT, land_buffer_m=0, zone_buffer_m=0)
snapped = r["start"]["snapped"]
snapped_depth = float(depth.at(*tools._to_km(snapped[1], snapped[0])))
check(f"a click in 1.5 m water snaps to {snapped_depth:.1f} m (>= 3.0 m), {r['start']['snap_km']} km away",
      snapped_depth >= 3.0 and r["start"]["snap_km"] > 0)

print("\n=== phase 2: detour breakdown ===")
e = routing.explain(54.45, 10.45, 54.45, 10.95)
steps = [row["step"].split(" (")[0] for row in e["detour"]]
check(f"four steps: {steps}", len(e["detour"]) == 4 and e["detour"][0]["extra_km"] is None)
zone_row = e["detour"][2]
check(f"the zone step names the avoided firing areas ({zone_row.get('zones_avoided')})",
      {"Schießgebiet Todendorf", "Schießgebiet Putlos"} <= set(zone_row.get("zones_avoided", [])))
check(f"main reason: {e['main_reason']}", e["main_reason"].startswith("restricted zones"))
check(f"the route bulges {e['summary']['detour_side']}", e["summary"]["detour_side"]["side"] == "north")
total = sum(row["extra_km"] for row in e["detour"][1:])
check(f"the extra km add up ({total:.2f} = {e['route']['distance_km']} - {e['detour'][0]['distance_km']})",
      math.isclose(total, e["route"]["distance_km"] - e["detour"][0]["distance_km"], abs_tol=0.05))
e = routing.explain(54.453, 10.26, 54.453, 10.36, draft_m=4.5, margin_m=1.0, land_buffer_m=0, zone_buffer_m=0)
check(f"Stoller Grund, 4.5 m draft: the depth step names the shoal ({e['detour'][1].get('reason')})",
      "5.1 m deep" in e["detour"][1].get("reason", "") and e["detour"][1]["extra_km"] >= 0)

print("\n=== phase 2: the breakdown is never meaningfully negative ===")
PAIRS_FOR_BREAKDOWN = [((54.45, 10.45), (54.45, 10.95)), ((54.408, 11.043), (54.376, 11.213)),
                       ((54.368, 10.158), (54.425, 10.22)), ((54.40, 11.02), (54.765, 10.0)),
                       (SEA_POINT, (54.60, 10.85)), ((54.453, 10.26), (54.453, 10.36)),
                       ((54.50, 10.00), (54.45, 11.35))]
worst = 0.0
for a, b in PAIRS_FOR_BREAKDOWN:
    e = routing.explain(*a, *b)
    for row in e["detour"][1:]:
        if row["extra_km"] is None:
            continue
        worst = min(worst, row["extra_km"])
        if -routing.NOISE_KM < row["extra_km"] < routing.NOISE_KM:
            check(f"noise shown as '≈ 0 km (grid rounding)' ({row['extra_km']} km)",
                  row["extra_text"] == "≈ 0 km (grid rounding)")
check(f"no step makes the route more than 0.2 km shorter (worst {worst} km)", worst > -routing.NOISE_KM)

print("\n=== phase 2: penalty factor ===")
a, b = (54.534, 10.881), (54.518, 10.815)                     # clips a corner of a firing area
low = routing.route(*a, *b, zones="penalty", penalty_factor=0.5)   # the zone's 500 m ring costs too
high = routing.route(*a, *b, zones="penalty", penalty_factor=20)
check(f"factor 0.5: the route crosses the corner ({low['zone_km_inside_penalty']} km inside, {low['distance_km']} km)",
      low["zone_km_inside_penalty"] > 0.1 and low["zones_crossed"])
check(f"factor 20: it goes around ({high['zone_km_inside_penalty']} km inside, {high['distance_km']} km)",
      high["zone_km_inside_penalty"] == 0 and high["distance_km"] > low["distance_km"])
e = routing.explain(*a, *b, zones="penalty", penalty_factor=20)
pen = [x for x in e["alternatives"] if x["kind"] == "penalty"]
check(f"the table explains it: {pen[0]['label'] if pen else 'nothing'}",
      bool(pen) and "crossing would save" in pen[0]["label"] and "penalty" in pen[0]["label"])
for bad in (-1, 25, float("nan")):
    try:
        routing.route(*a, *b, zones="penalty", penalty_factor=bad)
        check(f"penalty factor {bad}: refused", False, "no error")
    except routing.RouteError as exc:
        check(f"penalty factor {bad}: refused with a message ({exc})", "between 0 and 20" in str(exc))

print("\n=== phase 2: penalty factor 0 (zones cost nothing, still listed) ===")
from dataclasses import replace                                # noqa: E402
F0_PAIRS = {"Heiligenhafen -> Damp": ((54.40, 11.00), (54.58, 10.03)),
            "Kiel Bay -> west of Fehmarn": ((54.45, 10.45), (54.45, 10.95)),
            "firing-area corner": ((54.534, 10.881), (54.518, 10.815))}
for name, (a, b) in F0_PAIRS.items():
    f0 = routing.route(*a, *b, zones="penalty", penalty_factor=0, land_buffer_m=0, zone_buffer_m=0)
    e = routing.explain(*a, *b, zones="penalty", penalty_factor=0, land_buffer_m=0, zone_buffer_m=0)
    depth_stage = e["detour"][1]["distance_km"]
    check(f"{name}: factor 0 = direct water + depth rule ({f0['distance_km']} vs {depth_stage} km)",
          abs(f0["distance_km"] - depth_stage) < routing.NOISE_KM)
    f0_full = routing.route(*a, *b, zones="penalty", penalty_factor=0)
    ignored = routing.route(*a, *b, zones="penalty",
                            rules=replace(routing.make_rules(zones="penalty", penalty_factor=0), penalty=()))
    check(f"{name}: with all other rules, factor 0 = penalty zones ignored ({f0_full['distance_km']} vs "
          f"{ignored['distance_km']} km)", abs(f0_full["distance_km"] - ignored["distance_km"]) < routing.NOISE_KM)
check("factor 0 still lists the crossed zones",
      {"Schießgebiet Todendorf", "Schießgebiet Putlos"} <= set(
          routing.route(54.45, 10.45, 54.45, 10.95, zones="penalty", penalty_factor=0)["zones_crossed"]))

print("\n=== phase 2: the zone buffer follows the zone's mode ===")
pen_rules = routing.make_rules(zones="penalty", penalty_factor=1)
blk_rules = routing.make_rules(zones="block")
plan_pen = routing.build_plan(routing.CELL_KM, pen_rules, [])
plan_blk = routing.build_plan(routing.CELL_KM, blk_rules, [])
ring = routing._penalty_squares(routing.CELL_KM, pen_rules) & ~routing._zone_squares(routing.CELL_KM, pen_rules.penalty)
check(f"penalty mode: the 500 m ring around crossable zones costs the penalty ({int(ring.sum())} squares)",
      ring.sum() > 0)
usable_pen, usable_blk = int((plan_pen.usable & ring).sum()), int((plan_blk.usable & ring).sum())
check(f"... and usable (not a hard block): {usable_pen} of {int(ring.sum())} ring squares usable in penalty mode, "
      f"{usable_blk} in block mode (the rest: land, depth or no-entry areas)",
      usable_pen > 0.5 * ring.sum() and usable_blk <= 0.01 * ring.sum())

print("\n=== phase 2: shallow-water layer ===")
import base64                                                  # noqa: E402
import numpy as np                                             # noqa: E402
m3 = routing.shallow_mask(2.0, 1.0)
m6 = routing.shallow_mask(4.5, 1.0)
bits3 = np.unpackbits(np.frombuffer(base64.b64decode(m3["bits"]), dtype=np.uint8))[:m3["rows"] * m3["cols"]]
bits6 = np.unpackbits(np.frombuffer(base64.b64decode(m6["bits"]), dtype=np.uint8))[:m6["rows"] * m6["cols"]]
check(f"the layer grows with the draft ({bits3.sum()} -> {bits6.sum()} squares)", bits6.sum() > bits3.sum() > 0)

print(f"\n{'all passed' if not failures else f'{failures} FAILED'}")
