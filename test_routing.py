"""Tests for routing.py (code only, offline: needs only data/land_kiel_bay.geojson)."""
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
mask = routing.usable_mask(grid, [])
for name, (a, b) in PAIRS.items():
    started = time.time()
    r = routing.route(*a, *b)
    seconds = time.time() - started
    check(f"{name}: {r['distance_km']} km, no land crossed", not line_km(r).intersects(LAND))
    check(f"{name}: computed in {seconds:.2f} s (< 1 s)", seconds < 1)
    for end in (a, b):
        cell, _ = routing.snap(grid, mask, *end)
        check(f"{name}: snapped square {cell} is water", bool(grid.water[cell]))

# --- 2. Same point ----------------------------------------------------------------
print("\n=== same point ===")
r = routing.route(*SEA_POINT, *SEA_POINT)
check("same point: 0 km, ETA 0, one point",
      r["distance_km"] == 0 and r["eta_hours"] == 0 and len(r["path"]) == 1, r)

# --- 3. Around Fehmarn -------------------------------------------------------------
print("\n=== around Fehmarn ===")
west, east = (54.50, 10.95), (54.45, 11.35)
fehmarn = next(g for g in getattr(LAND, "geoms", [LAND]) if g.contains(Point(tools._to_km(11.15, 54.47))))
straight = LineString([tools._to_km(west[1], west[0]), tools._to_km(east[1], east[0])])
check("the straight line crosses Fehmarn (so the test means something)", straight.intersects(fehmarn))
r = routing.route(*west, *east)
check("the route does not touch Fehmarn", not line_km(r).intersects(fehmarn))
check(f"the route ({r['distance_km']} km) is longer than the straight line ({straight.length:.1f} km)",
      r["distance_km"] > straight.length)

# --- 4. Other properties ----------------------------------------------------------
print("\n=== other ===")
there = routing.route(*west, *east)["distance_km"]
back = routing.route(*east, *west)["distance_km"]
check(f"swapped start/end: same distance ({there} vs {back} km)", abs(there - back) <= 0.01 * there)

on_land = (54.47, 11.15)                                    # inside Fehmarn
r = routing.route(*on_land, *SEA_POINT)
check(f"a point on land snaps to water ({r['start']['snap_km']} km)",
      0 < r["start"]["snap_km"] <= routing.MAX_SNAP_KM and not line_km(r).intersects(LAND))

for name, point, expected in [("far inland", (54.20, 10.00), "No water within"),
                              ("outside the area", (53.50, 10.00), "Outside the routing area")]:
    try:
        routing.route(*point, *SEA_POINT)
        check(f"{name}: error", False, "no error")
    except routing.RouteError as exc:
        check(f"{name}: clear error", expected in str(exc), str(exc))

check("6 nm = 11.112 km", math.isclose(11.112 / routing.KM_PER_NM, 6.0))
check("ETA text: 1 h", routing.eta_text(6.0 / 6.0) == "1 h 00 min")
r = routing.route(*west, *east, speed_kn=6)
check("ETA = distance in nm / speed", math.isclose(r["eta_hours"], r["distance_nm"] / 6, abs_tol=0.002), r)
r12 = routing.route(*west, *east, speed_kn=12)
check("twice the speed: half the time", math.isclose(r12["eta_hours"] * 2, r["eta_hours"], abs_tol=0.002))
try:
    routing.route(*west, *east, speed_kn=0)
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
free = routing.route(*a, *b)
zone = RestrictedZone(10.60, 54.50, 10.64, 54.62)
blocked = routing.route(*a, *b, terms=[zone])
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


tolled = routing.route(*a, *b, terms=[Toll()])
check("an extra-cost term still finds a water route", not line_km(tolled).intersects(LAND))

print(f"\n{'all passed' if not failures else f'{failures} FAILED'}")
