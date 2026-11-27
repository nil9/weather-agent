"""Sea routing: the shortest WATER route between two points (code only, no LLM).

Phase 1: land.  Phase 2: water depth, restricted zones, safety buffers, detour explanation.

How it works:
  1. Grid: 0.25 km squares over Kiel Bay to Gedser (ROUTE_BBOX), in the same local km grid as the
     land/sea check (tools._to_km). A square is water only if it doesn't touch the land
     polygons at all (data/land_kiel_bay.geojson, unshrunk).
  2. Rules (all vectorised with numpy, see Rules):
       - depth: draft + margin at every square centre, and along every step between two
         centres (sampled every 50 m in the ~100 m EMODnet depth grid)
       - zones: squares touching a blocked zone are unusable; penalty zones cost extra
       - buffers: square centres and steps keep land_buffer from land and zone_buffer from
         blocked zones (distance rasters from a distance transform of the depth grid)
     A step between two usable squares is allowed only if it passes every rule along its
     whole length, so the route can't cut a corner across land, shallows or zones.
  3. Only the largest connected usable area is used (scipy connected components), and each
     clicked point snaps to its nearest usable square (at most MAX_SNAP_KM away). A rejected
     click gets the exact reason (land, depth with values, zone name, buffer).
  4. Search: Dijkstra on the step graph (scipy, C speed). Edge cost = step length + penalties,
     so the result is the cheapest route (the shortest one if no penalty zone is involved).
  5. Smooth: the grid zigzag is straightened ("string pulling") where the straight line
     passes the same rules (sampled along it) and stays off the land polygons.
  6. Distance in km and nm, ETA, minimum depth and distances to land / zones along the route.
  explain() routes the same trip 4 times (land only, + depth, + zones, + buffers), so the
  extra km of each rule show which one caused the detour. The chat uses it too.

Cost terms: custom terms with blocked(cell) / extra(a, b, ...) / segment_ok(p, q) still work
(the tests use them), but they are evaluated square by square in Python, so they are slow.
Later phases (weather at the arrival time, traffic density) become vectorised rules like
the zone penalty: an extra cost per step, always >= 0.
"""
import base64
import functools
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from datetime import datetime

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

import tools

ROUTE_BBOX = (9.8, 54.15, 12.8, 55.3)    # lon_min, lat_min, lon_max, lat_max (Kiel Bay to Gedser, Møn, Faxe Bugt)
CELL_KM = 0.25                           # grid resolution (0.25 km opens the Fehmarnsund)
MAX_SNAP_KM = 5.0                        # a click farther than this from usable water is rejected
SNAP_WARN_KM = 0.5                       # a click moved farther than this gets a visible warning
KM_PER_NM = 1.852
DEFAULT_SPEED_KN = 6.0
STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]   # (row, col)
BASE_STEPS = [(0, 1), (1, 0), (1, 1), (1, -1)]    # the other 4 are the same steps reversed

# --- rule settings (defaults; the Route tab can change them) -------------------
DRAFT_M = 2.0                            # the boat's draft
SAFETY_MARGIN_M = 1.0                    # extra water under the keel (water level, data errors)
LAND_BUFFER_M = 300                      # keep at least this far from land
ZONE_BUFFER_M = 500                      # keep at least this far from blocked zones
SAMPLE_KM = 0.02                         # rules are checked every 20 m along steps and lines
                                         # (finer than the ~67 m depth pixels, so none is skipped)
RASTER_TOL_KM = 0.07                     # distance rasters have ~100 m pixels: this is added to
                                         # a buffer, so the real distance is at least the buffer
# What each zone category means for the route: "block", "penalty" or "ignore".
# "firing" and "restricted" follow the user's choice `zones`, "wind_farm" its own choice
# `wind_farms` (block / penalty); see zone_rules().
ZONE_RULES = {"no_entry": "block", "firing": "block", "restricted": "block", "exercise": "ignore",
              "wind_farm": "block"}
TURBINE_RADIUS_M = 500                   # keep this far from every offshore wind turbine
TURBINE_RADIUS_RANGE = (50, 2000)        # the Route tab may set the radius in this range
ZONE_PENALTY = 10.0                      # penalty mode (default): each km inside a zone costs 10 km extra
PENALTY_RANGE = (0.0, 20.0)              # the Route tab may set the factor in this range
                                         # (0 = penalty zones cost nothing, but are still listed)
NOISE_KM = 0.2                           # breakdown steps within +-0.2 km are grid / straightening noise
TINY_ZONE_KM2 = 0.01                     # unnamed zones smaller than 1 ha: blocked, but no buffer
# Named narrow passages (lon/lat polygons). Inside them the buffers are reduced (a fairway is
# narrower than 2 x 300 m); the depth rule stays fully active. Routes that use the reduced
# clearance get a COLREG Rule 9 warning and a full-buffer alternative.
NARROW_PASSAGES = [
    {"name": "Fehmarnsund", "polygon": [(11.07, 54.36), (11.26, 54.36), (11.26, 54.42), (11.07, 54.42)]},
    {"name": "Kiel Fjord (Friedrichsort narrows)",
     "polygon": [(10.13, 54.33), (10.23, 54.33), (10.23, 54.425), (10.13, 54.425)]},
    # Danish sounds between Sjælland, Møn, Falster and Lolland (boxes drawn from the depth data)
    {"name": "Guldborgsund", "polygon": [(11.72, 54.63), (11.95, 54.63), (11.95, 54.91), (11.72, 54.91)]},
    {"name": "Storstrøm / Masnedsund", "polygon": [(11.78, 54.90), (12.06, 54.90), (12.06, 55.04), (11.78, 55.04)]},
                                                     # (includes the Farø narrows towards Grønsund / Ulvsund)
    {"name": "Ulvsund / Bøgestrøm", "polygon": [(12.02, 54.96), (12.36, 54.96), (12.36, 55.12), (12.02, 55.12)]},
    {"name": "Grønsund", "polygon": [(12.00, 54.82), (12.25, 54.82), (12.25, 54.955), (12.00, 54.955)]},
]
# Named sea areas used only to describe a route ("Via: around Møn → Fehmarnsund"); no rules.
ROUTE_LANDMARKS = [
    {"name": "around Møn (east of Møns Klint)", "polygon": [(12.45, 54.88), (12.8, 54.88), (12.8, 55.1), (12.45, 55.1)]},
]
# Harbours and sea areas, only to NAME a route's start and destination ("Heiligenhafen → Faxe Bugt").
# A point gets the nearest name within PLACE_KM, else its coordinates. (lat, lon), approximate.
PLACES = {
    "Kiel": (54.33, 10.15), "Laboe": (54.405, 10.225), "Kiel Lighthouse": (54.50, 10.27),
    "Eckernförde": (54.47, 9.85), "Damp": (54.585, 10.02), "Schleimünde": (54.67, 10.04),
    "Gelting Bay": (54.77, 9.98), "Marstal": (54.855, 10.52), "Ærøskøbing": (54.89, 10.41),
    "Bagenkop": (54.75, 10.68), "Rudkøbing": (54.94, 10.71), "Spodsbjerg": (54.93, 10.83),
    "Heiligenhafen": (54.375, 10.98), "Großenbrode": (54.37, 11.09), "Orth (Fehmarn)": (54.45, 11.05),
    "Burgtiefe (Fehmarn)": (54.41, 11.20), "Puttgarden": (54.50, 11.23), "Rødby": (54.655, 11.35),
    "Nakskov Fjord": (54.83, 11.08), "Rødsand": (54.56, 11.60), "Nysted": (54.665, 11.73),
    "Gedser": (54.575, 11.93), "Kühlungsborn": (54.16, 11.75), "Warnemünde": (54.18, 12.09),
    "Klintholm (Møn)": (54.95, 12.47), "Stege": (54.985, 12.29), "Faxe Bugt": (55.20, 12.30),
    "Rødvig": (55.25, 12.38), "Vordingborg": (55.005, 11.91), "Femø": (54.97, 11.53),
    "Omø": (55.16, 11.14), "Karrebæksminde": (55.18, 11.65), "Bandholm": (54.84, 11.49),
    "Stubbekøbing": (54.89, 12.04),
}
PLACE_KM = 8.0


def place_name(lat: float, lon: float) -> str:
    """'Heiligenhafen' if a known place is within PLACE_KM, else '54.40° N, 11.00° E'."""
    x, y = tools._to_km(lon, lat)
    name, (plat, plon) = min(PLACES.items(), key=lambda kv: math.dist((x, y), tools._to_km(kv[1][1], kv[1][0])))
    if math.dist((x, y), tools._to_km(plon, plat)) <= PLACE_KM:
        return name
    return f"{lat:.2f}° N, {lon:.2f}° E"


PASSAGE_LAND_BUFFER_M = 50
PASSAGE_ZONE_BUFFER_M = 100
PASSAGE_WARNING = "narrow passage (COLREG Rule 9): reduced clearance, keep to the fairway"
log = logging.getLogger("routing")
CATEGORY_LABEL = {"no_entry": "no-entry area", "firing": "firing area", "restricted": "restricted military area",
                  "exercise": "exercise area", "wind_farm": "wind farm"}
DEPTH_FILE = "depth_kiel_bay.npz"
ZONES_FILE = "restricted_zones_kiel_bay.geojson"
WIND_FILE = "wind_farms_kiel_bay.geojson"      # offshore wind farm areas + turbines (points)
SOURCES = {"land": "Natural Earth 1:10m (public domain)",
           "depth": "EMODnet Bathymetry DTM (CC BY 4.0)",
           "zones": "© OpenStreetMap contributors (ODbL)",
           "wind_farms": "© OpenStreetMap contributors (ODbL)"}
NOT_FOR_NAVIGATION = "Not for navigation."


class RouteError(ValueError):
    """The route can't be computed (outside the area, no usable water nearby, no connection)."""


class RoutingDataMissing(FileNotFoundError):
    """data/ lacks the depth or zone file."""


@dataclass(frozen=True)
class RouteContext:
    speed_kn: float
    departure: datetime | None = None    # for time-dependent rules later (weather)


@dataclass(frozen=True)
class Rules:
    """Everything that decides which squares and steps a route may use."""
    required_depth_m: float | None = None          # None: no depth rule
    block: tuple[str, ...] = ()                    # zone categories that are closed
    penalty: tuple[str, ...] = ()                  # zone categories that cost extra
    land_buffer_km: float = 0.0                    # open water
    zone_buffer_km: float = 0.0
    passages: bool = False                         # reduced buffers in NARROW_PASSAGES
    passage_land_buffer_km: float = 0.0
    passage_zone_buffer_km: float = 0.0
    penalty_factor: float = ZONE_PENALTY
    turbine_radius_km: float = TURBINE_RADIUS_M / 1000   # a turbine is a circle of this radius


LAND_ONLY = Rules()


# ---------------------------------------------------------------------------
# Grid and data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    x0: float                 # km of the south-west corner (local km grid)
    y0: float
    cell_km: float
    water: np.ndarray         # bool, shape (rows = south -> north, cols = west -> east)

    def center(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return self.x0 + (col + 0.5) * self.cell_km, self.y0 + (row + 0.5) * self.cell_km

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.water.shape
        return np.meshgrid(self.x0 + (np.arange(cols) + 0.5) * self.cell_km,
                           self.y0 + (np.arange(rows) + 0.5) * self.cell_km)

    def squares(self) -> np.ndarray:
        rows, cols = self.water.shape
        gx, gy = np.meshgrid(self.x0 + np.arange(cols) * self.cell_km, self.y0 + np.arange(rows) * self.cell_km)
        return shapely.box(gx, gy, gx + self.cell_km, gy + self.cell_km)

    def lonlat_bounds(self) -> list[list[float]]:
        rows, cols = self.water.shape
        lon0, lat0 = to_lonlat(self.x0, self.y0)
        lon1, lat1 = to_lonlat(self.x0 + cols * self.cell_km, self.y0 + rows * self.cell_km)
        return [[float(lat0), float(lon0)], [float(lat1), float(lon1)]]


def to_lonlat(x_km, y_km):
    """Inverse of tools._to_km (works on numbers and numpy arrays)."""
    return x_km / tools._KM_PER_DEG_LON, y_km / tools._KM_PER_DEG_LAT


@functools.lru_cache(maxsize=4)
def build_grid(cell_km: float = CELL_KM) -> Grid:
    """Water squares of ROUTE_BBOX. Built once per resolution."""
    land = tools._land_km()
    shapely.prepare(land)
    lon0, lat0, lon1, lat1 = ROUTE_BBOX
    x0, y0 = tools._to_km(lon0, lat0)
    x1, y1 = tools._to_km(lon1, lat1)
    cols, rows = int((x1 - x0) // cell_km), int((y1 - y0) // cell_km)
    empty = Grid(x0, y0, cell_km, np.zeros((rows, cols), dtype=bool))
    water = ~shapely.intersects(land, empty.squares())          # touches land at all -> not water
    return Grid(x0, y0, cell_km, water)


@functools.lru_cache(maxsize=1)
def _land_exact():
    return prep(tools._land_km())


@dataclass(frozen=True)
class Raster:
    """A value per pixel of the depth grid (row 0 = north), looked up at local-km points."""
    values: np.ndarray
    lon0: float
    lat1: float
    ppd: int

    def at(self, x_km, y_km) -> np.ndarray:
        lon, lat = to_lonlat(np.asarray(x_km, dtype=float), np.asarray(y_km, dtype=float))
        rows, cols = self.values.shape
        col = np.clip(((lon - self.lon0) * self.ppd).astype(int), 0, cols - 1)
        row = np.clip(((self.lat1 - lat) * self.ppd).astype(int), 0, rows - 1)
        return self.values[row, col]


@functools.lru_cache(maxsize=1)
def load_depth() -> Raster:
    """Water depth in m (positive), <= 0 on land. EMODnet Bathymetry, ~100 m pixels."""
    path = tools._safe_path(tools.DATA_DIR, DEPTH_FILE)
    if not path.exists():
        raise RoutingDataMissing("Depth data missing: run `python download_routing_data.py` once.")
    data = np.load(path)
    return Raster(values=-(data["elev_dm"].astype(np.float32) / 10), lon0=float(data["lon0"]),
                  lat1=float(data["lat1"]), ppd=int(data["pixels_per_degree"]))


def _pixel_xy() -> tuple[np.ndarray, np.ndarray]:
    d = load_depth()
    rows, cols = d.values.shape
    lon = d.lon0 + (np.arange(cols) + 0.5) / d.ppd
    lat = d.lat1 - (np.arange(rows) + 0.5) / d.ppd
    return np.meshgrid(lon * tools._KM_PER_DEG_LON, lat * tools._KM_PER_DEG_LAT)


def _mark_outline(inside: np.ndarray, geom) -> np.ndarray:
    """Also mark every pixel the outline of `geom` passes through, so features thinner than
    a pixel (a narrow zone, a mole) are not lost when only pixel centres are tested."""
    d = load_depth()
    boundary = shapely.segmentize(geom.boundary, 0.02)            # a point every 20 m
    xy = shapely.get_coordinates(boundary)
    if not len(xy):
        return inside
    lon, lat = to_lonlat(xy[:, 0], xy[:, 1])
    rows, cols = inside.shape
    col = np.clip(((lon - d.lon0) * d.ppd).astype(int), 0, cols - 1)
    row = np.clip(((d.lat1 - lat) * d.ppd).astype(int), 0, rows - 1)
    inside = inside.copy()
    inside[row, col] = True
    return inside


def _distance_raster(inside: np.ndarray) -> Raster:
    """km from each pixel to the nearest pixel where `inside` is true (0 inside)."""
    d = load_depth()
    dy = tools._KM_PER_DEG_LAT / d.ppd
    dx = tools._KM_PER_DEG_LON / d.ppd
    dist = distance_transform_edt(~inside, sampling=(dy, dx)).astype(np.float32)
    return Raster(dist, d.lon0, d.lat1, d.ppd)


@functools.lru_cache(maxsize=1)
def land_distance() -> Raster:
    """km to the nearest land: the land polygons OR land in the depth model (small islets)."""
    land = tools._land_km()
    shapely.prepare(land)
    inside = shapely.contains_xy(land, *_pixel_xy()) | (load_depth().values <= 0)
    return _distance_raster(_mark_outline(inside, land))


# --- restricted zones ----------------------------------------------------------

@dataclass(frozen=True)
class Zone:
    name: str
    category: str
    osm: str
    geom_km: object           # shapely geometry in the local km grid (a Point for a turbine)
    kind: str = "area"        # "area" or "turbine" (wind turbines: a circle of the turbine radius)


def _zone_files() -> list[Path]:
    paths = []
    for name, hint in [(ZONES_FILE, "Restricted-zone"), (WIND_FILE, "Wind-farm")]:
        path = tools._safe_path(tools.DATA_DIR, name)
        if not path.exists():
            flag = " --wind-only" if name == WIND_FILE else ""
            raise RoutingDataMissing(f"{hint} data missing: run `python download_routing_data.py{flag}` once.")
        paths.append(path)
    return paths


@functools.lru_cache(maxsize=1)
def load_zones() -> tuple[Zone, ...]:
    """Restricted zones and wind farms (areas and turbines) from data/."""
    zones = []
    for path in _zone_files():
        for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
            geom = shapely.transform(shape(f["geometry"]),
                                     lambda xy: xy * [tools._KM_PER_DEG_LON, tools._KM_PER_DEG_LAT])
            p = f["properties"]
            zones.append(Zone(p["name"], p["category"], p["osm"], geom, p.get("kind", "area")))
    tiny = [z for z in zones if is_tiny(z)]
    if tiny:
        log.warning("Tiny unnamed zones (< 1 ha): blocked, but without a buffer: %s",
                    ", ".join(f"{z.osm} ({z.category}, {z.geom_km.area * 100:.2f} ha)" for z in tiny))
    else:
        log.warning("Tiny unnamed zones (< 1 ha): none in the zone file.")
    return tuple(zones)


def is_tiny(zone: "Zone") -> bool:
    """An unnamed zone smaller than 1 ha (the download names unnamed zones '<category> area')."""
    return zone.kind == "area" and zone.geom_km.area < TINY_ZONE_KM2 and zone.name == f"{zone.category} area"


def no_buffer(zone: "Zone") -> bool:
    """Zones that get no zone buffer: tiny unnamed zones, and turbines (their radius IS their buffer)."""
    return zone.kind == "turbine" or is_tiny(zone)


def zone_geom(zone: "Zone", turbine_km: float = TURBINE_RADIUS_M / 1000):
    """The area a zone closes: a turbine is a circle of the turbine radius."""
    return zone.geom_km.buffer(turbine_km, 16) if zone.kind == "turbine" else zone.geom_km


def tiny_zones() -> list[dict]:
    return [{"osm": z.osm, "category": z.category, "area_ha": round(z.geom_km.area * 100, 2)}
            for z in load_zones() if is_tiny(z)]


def zones_geojson() -> dict:
    """The zone and wind-farm files as they are (lon/lat), for drawing on the map. Turbines are
    points; the page draws their radius."""
    features = []
    for path in _zone_files():
        features += json.loads(path.read_text(encoding="utf-8"))["features"]
    return {"type": "FeatureCollection", "features": features,
            "properties": {"turbine_radius_m": TURBINE_RADIUS_M}}


@functools.lru_cache(maxsize=16)
def _zone_union(categories: tuple[str, ...], include_tiny: bool = True,
                turbine_km: float = TURBINE_RADIUS_M / 1000):
    """Union of the zones of these categories. include_tiny=False: only zones that get a buffer
    (no tiny zones, no turbines), so turbine_km doesn't matter then."""
    geoms = [zone_geom(z, turbine_km) for z in load_zones()
             if z.category in categories and (include_tiny or not no_buffer(z))]
    if not geoms:
        return None
    union = unary_union(geoms)
    shapely.prepare(union)
    return union


@functools.lru_cache(maxsize=16)
def zone_distance(categories: tuple[str, ...], include_tiny: bool = True,
                  turbine_km: float = TURBINE_RADIUS_M / 1000) -> Raster:
    """km to the nearest zone of these categories (0 inside). include_tiny=False: for buffers."""
    union = _zone_union(categories, include_tiny, turbine_km)
    if union is None:
        inside = np.zeros(load_depth().values.shape, dtype=bool)
    else:
        inside = _mark_outline(shapely.contains_xy(union, *_pixel_xy()), union)
    return _distance_raster(inside)


@functools.lru_cache(maxsize=16)
def _zone_squares(cell_km: float, categories: tuple[str, ...],
                  turbine_km: float = TURBINE_RADIUS_M / 1000) -> np.ndarray:
    """Grid squares that touch a zone of these categories (exact, turbines as circles)."""
    union = _zone_union(categories, True, turbine_km)
    grid = build_grid(cell_km)
    if union is None:
        return np.zeros(grid.water.shape, dtype=bool)
    return shapely.intersects(union, grid.squares())


# --- values at square centres and minimum values along steps (cached per grid) ----

@functools.lru_cache(maxsize=1)
def _passage_union():
    from shapely.geometry import Polygon
    polys = [shapely.transform(Polygon(p["polygon"]), lambda xy: xy * [tools._KM_PER_DEG_LON, tools._KM_PER_DEG_LAT])
             for p in NARROW_PASSAGES]
    union = unary_union(polys)
    shapely.prepare(union)
    return union


@functools.lru_cache(maxsize=1)
def passage_raster() -> Raster:
    """1 inside a named narrow passage, 0 elsewhere (pixel grid)."""
    d = load_depth()
    inside = shapely.contains_xy(_passage_union(), *_pixel_xy()).astype(np.float32)
    return Raster(inside, d.lon0, d.lat1, d.ppd)


@functools.lru_cache(maxsize=8)
def _passage_exact_zone(cell_km: float, categories: tuple[str, ...]):
    """Inside narrow passages, the EXACT distance (shapely) from square centres and from each
    step to the closed zones without tiny ones; np.inf elsewhere. No raster tolerance is
    needed there, so the small passage buffers are not doubled."""
    grid = build_grid(cell_km)
    union = _zone_union(categories, False)
    in_passage = centre_values(cell_km, "passage") > 0
    rows, cols = in_passage.shape
    centre = np.full((rows, cols), np.inf)
    steps = {}
    if union is None:
        return centre, {d: np.full((rows, cols), np.inf) for d in BASE_STEPS}
    cx, cy = grid.centres()
    centre[in_passage] = shapely.distance(union, shapely.points(cx[in_passage], cy[in_passage]))
    for dr, dc in BASE_STEPS:
        d = np.full((rows, cols), np.inf)
        either = in_passage | _shifted(in_passage, dr, dc)
        r, c = np.nonzero(either)
        x0, y0 = cx[r, c], cy[r, c]
        lines = shapely.linestrings(np.stack([np.column_stack([x0, y0]),
                                              np.column_stack([x0 + dc * cell_km, y0 + dr * cell_km])], axis=1))
        d[r, c] = shapely.distance(union, lines)
        steps[(dr, dc)] = d
    return centre, steps


def _raster(key):
    if key == "depth":
        return load_depth()
    if key == "land":
        return land_distance()
    if key == "passage":
        return passage_raster()
    return zone_distance(*key[1:])          # ("zone", categories[, include_tiny[, turbine_km]])


@functools.lru_cache(maxsize=32)
def centre_values(cell_km: float, key) -> np.ndarray:
    return _raster(key).at(*build_grid(cell_km).centres())


@functools.lru_cache(maxsize=32)
def step_minimum(cell_km: float, key) -> dict[tuple[int, int], np.ndarray]:
    """For every step between two square centres (4 base directions), the minimum of the
    raster along it, sampled every SAMPLE_KM. Independent of the settings, so it is computed
    once per grid; a rule check is then one comparison per step."""
    grid = build_grid(cell_km)
    raster = _raster(key)
    cx, cy = grid.centres()
    rows, cols = grid.water.shape
    result = {}
    for dr, dc in BASE_STEPS:
        n = math.ceil(cell_km * math.hypot(dr, dc) / SAMPLE_KM) + 1
        values = np.full((rows, cols), np.inf, dtype=np.float32)
        for t in np.linspace(0, 1, n):
            np.minimum(values, raster.at(cx + t * dc * cell_km, cy + t * dr * cell_km), out=values)
        result[(dr, dc)] = values
    return result


def _sample(points):
    xs, ys = [], []
    for p, q in zip(points, points[1:]):
        n = max(2, math.ceil(math.dist(p, q) / SAMPLE_KM) + 1)
        xs.append(np.linspace(p[0], q[0], n))
        ys.append(np.linspace(p[1], q[1], n))
    if not xs:
        return np.array([points[0][0]]), np.array([points[0][1]])
    return np.concatenate(xs), np.concatenate(ys)


def _min_along(raster: Raster, points) -> tuple[float, tuple[float, float]]:
    x, y = _sample(points)
    v = raster.at(x, y)
    i = int(np.argmin(v))
    return float(v[i]), (float(x[i]), float(y[i]))


def depths_along(points) -> tuple[float, tuple[float, float]]:
    """Minimum depth along a polyline (sampled every SAMPLE_KM) and where it is."""
    return _min_along(load_depth(), points)


# ---------------------------------------------------------------------------
# Settings -> rules
# ---------------------------------------------------------------------------

def zone_rules(zones: str = "block", wind_farms: str = "block") -> dict[str, str]:
    """The rule per category. `zones` ("block" / "penalty") applies to firing and other
    restricted military areas, `wind_farms` to wind farm areas and turbines; no-entry areas
    are always blocked, exercise areas ignored."""
    if zones not in ("block", "penalty"):
        raise RouteError("zones must be 'block' or 'penalty'.")
    if wind_farms not in ("block", "penalty"):
        raise RouteError("wind_farms must be 'block' or 'penalty'.")
    return {**ZONE_RULES, "firing": zones, "restricted": zones, "wind_farm": wind_farms}


def make_rules(draft_m: float = DRAFT_M, margin_m: float = SAFETY_MARGIN_M, zones: str = "block",
               land_buffer_m: float = LAND_BUFFER_M, zone_buffer_m: float = ZONE_BUFFER_M,
               passages: bool = True, passage_land_buffer_m: float = PASSAGE_LAND_BUFFER_M,
               passage_zone_buffer_m: float = PASSAGE_ZONE_BUFFER_M,
               penalty_factor: float = ZONE_PENALTY, wind_farms: str = "block",
               turbine_radius_m: float = TURBINE_RADIUS_M) -> Rules:
    if draft_m <= 0 or margin_m < 0:
        raise RouteError("Draft must be more than 0 m and the margin at least 0 m.")
    if min(land_buffer_m, zone_buffer_m, passage_land_buffer_m, passage_zone_buffer_m) < 0:
        raise RouteError("Buffers can't be negative.")
    if not (isinstance(penalty_factor, (int, float)) and math.isfinite(penalty_factor)
            and PENALTY_RANGE[0] <= penalty_factor <= PENALTY_RANGE[1]):
        raise RouteError(f"The penalty factor must be between {PENALTY_RANGE[0]:g} and {PENALTY_RANGE[1]:g}.")
    if not (isinstance(turbine_radius_m, (int, float)) and math.isfinite(turbine_radius_m)
            and TURBINE_RADIUS_RANGE[0] <= turbine_radius_m <= TURBINE_RADIUS_RANGE[1]):
        raise RouteError(f"The turbine radius must be between {TURBINE_RADIUS_RANGE[0]} and "
                         f"{TURBINE_RADIUS_RANGE[1]} m.")
    rules = zone_rules(zones, wind_farms)
    return Rules(required_depth_m=round(draft_m + margin_m, 2),
                 block=tuple(sorted(c for c, r in rules.items() if r == "block")),
                 penalty=tuple(sorted(c for c, r in rules.items() if r == "penalty")),
                 land_buffer_km=land_buffer_m / 1000, zone_buffer_km=zone_buffer_m / 1000,
                 passages=passages,
                 # a passage never gets a LARGER buffer than open water
                 passage_land_buffer_km=min(passage_land_buffer_m, land_buffer_m) / 1000,
                 passage_zone_buffer_km=min(passage_zone_buffer_m, zone_buffer_m) / 1000,
                 penalty_factor=float(penalty_factor), turbine_radius_km=turbine_radius_m / 1000)


def _threshold(buffer_km: float) -> float:
    """Distance a centre / step must keep: the buffer plus the raster tolerance, or with
    no buffer just 'not inside' (more than 0)."""
    return buffer_km + RASTER_TOL_KM if buffer_km > 0 else 1e-6


def _thresholds(rules: Rules, in_passage) -> tuple:
    """(land, zone) thresholds per point: reduced inside narrow passages (if enabled)."""
    land_open, zone_open = _threshold(rules.land_buffer_km), _threshold(rules.zone_buffer_km)
    if not rules.passages:
        return land_open, zone_open
    return (np.where(in_passage, _threshold(rules.passage_land_buffer_km), land_open),
            np.where(in_passage, _threshold(rules.passage_zone_buffer_km), zone_open))


# ---------------------------------------------------------------------------
# Usable squares, allowed steps, graph
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    grid: Grid
    rules: Rules
    usable: np.ndarray                     # usable squares in the largest connected area
    graph: object                          # scipy csr matrix (directed, cost per step)


def _shifted(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r, c] = mask[r + dr, c + dc] (False outside the grid)."""
    rows, cols = mask.shape
    out = np.zeros_like(mask)
    rs, re_ = max(0, -dr), min(rows, rows - dr)
    cs, ce = max(0, -dc), min(cols, cols - dc)
    out[rs:re_, cs:ce] = mask[rs + dr:re_ + dr, cs + dc:ce + dc]
    return out


def _usable_squares(grid: Grid, rules: Rules, terms: list) -> np.ndarray:
    ck = grid.cell_km
    usable = grid.water.copy()
    land_thr, zone_thr = _thresholds(rules, centre_values(ck, "passage") > 0)
    if rules.required_depth_m is not None:
        usable &= centre_values(ck, "depth") >= rules.required_depth_m
    if rules.land_buffer_km > 0:
        usable &= centre_values(ck, "land") >= land_thr
    if rules.block:
        usable &= ~_zone_squares(ck, rules.block, rules.turbine_radius_km)   # inside a closed zone / turbine circle
        if rules.zone_buffer_km > 0:                                 # buffer: not around tiny zones
            raster_ok = centre_values(ck, ("zone", rules.block, False)) >= zone_thr
            if rules.passages:                                       # passages: exact distance instead
                in_p = centre_values(ck, "passage") > 0
                exact, _ = _passage_exact_zone(ck, rules.block)
                raster_ok = np.where(in_p, exact >= rules.passage_zone_buffer_km, raster_ok)
            usable &= raster_ok
    for term in terms:                                    # custom terms (slow, square by square)
        if hasattr(term, "blocked_mask"):
            usable &= ~term.blocked_mask(grid)
        else:
            for row, col in zip(*np.nonzero(usable)):
                if term.blocked((row, col), grid):
                    usable[row, col] = False
    return usable


@functools.lru_cache(maxsize=8)
def _plan_cached(cell_km: float, rules: Rules) -> Plan:
    return _plan(build_grid(cell_km), rules, [])


def build_plan(cell_km: float, rules: Rules, terms: list) -> Plan:
    """Usable squares + step graph for these rules (cached when there are no custom terms)."""
    if not terms:
        return _plan_cached(cell_km, rules)
    return _plan(build_grid(cell_km), rules, terms)


def _penalty_squares(ck: float, rules: Rules) -> np.ndarray | None:
    """Squares that cost the penalty: inside a penalty zone, or within its buffer. The buffer
    follows the zone's mode: around a blocked zone it blocks, around a penalty zone it costs
    the same penalty, so a crossable zone never keeps a hard 500 m ring."""
    if not rules.penalty or rules.penalty_factor == 0:
        return None
    squares = _zone_squares(ck, rules.penalty, rules.turbine_radius_km)
    if rules.zone_buffer_km > 0:
        land_thr, zone_thr = _thresholds(rules, centre_values(ck, "passage") > 0)
        squares = squares | (centre_values(ck, ("zone", rules.penalty, False)) < zone_thr)
    return squares


def _plan(grid: Grid, rules: Rules, terms: list) -> Plan:
    ck = grid.cell_km
    usable = _usable_squares(grid, rules, terms)
    rows, cols = usable.shape
    penalty_squares = _penalty_squares(ck, rules)
    land_steps = step_minimum(ck, "land")
    depth_steps = step_minimum(ck, "depth") if rules.required_depth_m is not None else None
    inside_steps = step_minimum(ck, ("zone", rules.block, True, rules.turbine_radius_km)) if rules.block else None
    buffer_steps = (step_minimum(ck, ("zone", rules.block, False))
                    if rules.block and rules.zone_buffer_km > 0 else None)
    # A step touching a narrow passage uses the passage's (smaller) buffers.
    in_passage = centre_values(ck, "passage") > 0
    exact_steps = (_passage_exact_zone(ck, rules.block)[1]
                   if rules.block and rules.zone_buffer_km > 0 and rules.passages else None)
    land = tools._land_km()
    shapely.prepare(land)

    index = np.arange(rows * cols).reshape(rows, cols)
    sources, targets, weights = [], [], []
    for dr, dc in BASE_STEPS:
        land_thr, zone_thr = _thresholds(rules, in_passage | _shifted(in_passage, dr, dc))
        ok = usable & _shifted(usable, dr, dc) & (land_steps[(dr, dc)] >= land_thr)
        if depth_steps is not None:
            ok &= depth_steps[(dr, dc)] >= rules.required_depth_m
        if inside_steps is not None:
            ok &= inside_steps[(dr, dc)] > 1e-6
        if buffer_steps is not None:
            raster_ok = buffer_steps[(dr, dc)] >= zone_thr
            if rules.passages:                                       # passages: exact distance instead
                touching = in_passage | _shifted(in_passage, dr, dc)
                raster_ok = np.where(touching, exact_steps[(dr, dc)] >= rules.passage_zone_buffer_km, raster_ok)
            ok &= raster_ok
        r, c = np.nonzero(ok)
        if dr and dc and len(r):
            # A diagonal step must also stay off the exact land polygons (no corner cutting).
            x0 = grid.x0 + (c + 0.5) * ck
            y0 = grid.y0 + (r + 0.5) * ck
            coords = np.stack([np.column_stack([x0, y0]), np.column_stack([x0 + dc * ck, y0 + dr * ck])], axis=1)
            clear = ~shapely.intersects(land, shapely.linestrings(coords))
            r, c = r[clear], c[clear]
        if terms and len(r):
            seg_terms = [t for t in terms if hasattr(t, "segment_ok")]
            keep = np.array([all(t.segment_ok(grid.center((a, b)), grid.center((a + dr, b + dc)), grid)
                                 for t in seg_terms) for a, b in zip(r, c)], dtype=bool)
            r, c = r[keep], c[keep]
        step_km = ck * math.hypot(dr, dc)
        a, b = index[r, c], index[r + dr, c + dc]
        w_ab = np.full(len(a), step_km)
        w_ba = np.full(len(a), step_km)
        if penalty_squares is not None:
            w_ab += rules.penalty_factor * step_km * penalty_squares[r + dr, c + dc]
            w_ba += rules.penalty_factor * step_km * penalty_squares[r, c]
        ctx = RouteContext(DEFAULT_SPEED_KN)
        for t in terms:                                   # custom extra costs (slow)
            if hasattr(t, "extra"):
                w_ab += np.array([t.extra((i, j), (i + dr, j + dc), step_km, ctx) for i, j in zip(r, c)])
                w_ba += np.array([t.extra((i + dr, j + dc), (i, j), step_km, ctx) for i, j in zip(r, c)])
        sources += [a, b]
        targets += [b, a]
        weights += [w_ab, w_ba]
    n = rows * cols
    graph = csr_matrix((np.concatenate(weights), (np.concatenate(sources), np.concatenate(targets))),
                       shape=(n, n))
    # Only the largest connected area (the open sea): pockets cut off by the rules are dropped.
    _, labels = connected_components(graph, directed=False)
    labels = labels.reshape(rows, cols)
    if usable.any():
        sea = labels == np.bincount(labels[usable]).argmax()
        usable = usable & sea
    return Plan(grid, rules, usable, graph)


def usable_mask(grid: Grid, terms: list) -> np.ndarray:
    """Usable squares (land only + custom terms) in the largest connected area."""
    return build_plan(grid.cell_km, LAND_ONLY, terms).usable


# ---------------------------------------------------------------------------
# Snapping, and why a click is rejected
# ---------------------------------------------------------------------------

def _in_box(lat: float, lon: float):
    lon0, lat0, lon1, lat1 = ROUTE_BBOX
    if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
        raise RouteError(f"Outside the routing area ({lat0}–{lat1}° N, {lon0}–{lon1}° E).")


def point_reason(lat: float, lon: float, rules: Rules, connection: bool = True) -> str | None:
    """Why this exact point can't be used with these rules (in plain words, with values).
    If the point itself is fine, its water is cut off: connection=True names what blocks the
    connection to the open sea (connection_reason); connection=False returns None then."""
    x, y = tools._to_km(lon, lat)
    p = Point(x, y)
    if _land_exact().contains(p):
        return "this point is on land"
    if rules.required_depth_m is not None:
        d = float(load_depth().at(x, y))
        if d < rules.required_depth_m:
            return (f"the water here is {max(d, 0):.1f} m deep, the boat needs at least "
                    f"{rules.required_depth_m:g} m (draft + safety margin)")
    for z in load_zones():
        if z.category not in rules.block:
            continue
        if z.kind == "turbine" and z.geom_km.distance(p) < rules.turbine_radius_km:
            return (f"this point is {z.geom_km.distance(p) * 1000:.0f} m from a wind turbine ({z.name}); "
                    f"the turbine radius is {rules.turbine_radius_km * 1000:.0f} m")
        if z.kind == "area" and z.geom_km.contains(p):
            return f"this point is inside {z.name} ({CATEGORY_LABEL[z.category]}, closed with the current settings)"
    if rules.block and rules.zone_buffer_km > 0:
        dist, z = min(((z.geom_km.distance(p), z) for z in load_zones()
                       if z.category in rules.block and not no_buffer(z)), key=lambda t: t[0])
        if dist < rules.zone_buffer_km + RASTER_TOL_KM:
            return (f"this point is {dist * 1000:.0f} m from {z.name} ({CATEGORY_LABEL[z.category]}); "
                    f"the zone buffer is {rules.zone_buffer_km * 1000:.0f} m")
    if rules.land_buffer_km > 0:
        dist = float(land_distance().at(x, y))
        in_passage = rules.passages and float(passage_raster().at(x, y)) > 0
        buffer_km = rules.passage_land_buffer_km if in_passage else rules.land_buffer_km
        if dist < buffer_km + RASTER_TOL_KM:
            return (f"this point is about {dist * 1000:.0f} m from land; the land buffer is {buffer_km * 1000:.0f} m"
                    + (" (narrow passage)" if in_passage else ""))
    if not connection:
        return None
    return (connection_reason(lat, lon, rules, build_grid().cell_km)
            or "the water here is cut off from the open sea with the current settings")


def snap(grid: Grid, mask: np.ndarray, lat: float, lon: float,
         rules: Rules = LAND_ONLY) -> tuple[tuple[int, int], float]:
    """Nearest usable square to (lat, lon) -> (cell, distance in km)."""
    _in_box(lat, lon)
    x, y = tools._to_km(lon, lat)
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise RouteError("No usable water at all with these settings.")
    # A click IN water that has no water connection to the open sea (a lagoon, or water that
    # connects only outside the routing area) is not moved across land to another sea.
    here = (int((y - grid.y0) // grid.cell_km), int((x - grid.x0) // grid.cell_km))
    if (0 <= here[0] < grid.water.shape[0] and 0 <= here[1] < grid.water.shape[1] and grid.water[here]
            and not build_plan(grid.cell_km, LAND_ONLY, []).usable[here]):
        local = point_reason(lat, lon, rules, connection=False)
        connection = connection_reason(lat, lon, rules, grid.cell_km)
        raise RouteError("No route from here: " + (f"{local}; " if local else "")
                         + (connection or "this water has no connection to the open sea").rstrip(".") + ".")
    cx = grid.x0 + (cols + 0.5) * grid.cell_km
    cy = grid.y0 + (rows + 0.5) * grid.cell_km
    dist = np.hypot(cx - x, cy - y)
    best = int(np.argmin(dist))
    if dist[best] > MAX_SNAP_KM:
        local = point_reason(lat, lon, rules, connection=False)
        connection = connection_reason(lat, lon, rules, grid.cell_km)
        text = f"No usable water within {MAX_SNAP_KM:g} km"
        if local:
            text += f": {local}."
        if connection:
            text += (" " if local else ". ") + "No route from here: " + connection
        elif not local:
            text += ": the water here is cut off from the open sea with the current settings."
        raise RouteError(text)
    return (int(rows[best]), int(cols[best])), float(dist[best])


# ---------------------------------------------------------------------------
# Why there is no route: what blocks the water connection to the open sea
# ---------------------------------------------------------------------------

MAX_BLOCKAGES = 3            # name at most this many blocked connections
BLOCK_COST = 1000.0          # search cost per km through a square that breaks a rule


def _edges_touched(area: np.ndarray) -> list[str]:
    """Which edges of the routing area this set of squares reaches (rows run south -> north)."""
    lon0, lat0, lon1, lat1 = ROUTE_BBOX
    return [label for label, hit in [(f"east edge ({lon1:g}° E)", area[:, -1].any()),
                                     (f"west edge ({lon0:g}° E)", area[:, 0].any()),
                                     (f"north edge ({lat1:g}° N)", area[-1, :].any()),
                                     (f"south edge ({lat0:g}° N)", area[0, :].any())] if hit]


def _place_name(cells: np.ndarray, grid: Grid) -> str:
    """A named narrow passage that holds most of these squares, else the position."""
    x = grid.x0 + (cells[:, 1] + 0.5) * grid.cell_km
    y = grid.y0 + (cells[:, 0] + 0.5) * grid.cell_km
    from shapely.geometry import Polygon
    best, count = None, 0
    for p in NARROW_PASSAGES:
        poly = shapely.transform(Polygon(p["polygon"]), lambda xy: xy * [tools._KM_PER_DEG_LON, tools._KM_PER_DEG_LAT])
        n = int(shapely.contains_xy(poly, x, y).sum())
        if n > count:
            best, count = p["name"], n
    if best and count >= len(cells) / 2:
        return best
    lon, lat = to_lonlat(float(x.mean()), float(y.mean()))
    return f"the water near {lat:.2f}° N, {lon:.2f}° E"


def _describe_block(cells: np.ndarray, grid: Grid, rules: Rules) -> str:
    """'Ulvsund / Bøgestrøm: too shallow (down to 1.8 m; the boat needs 3 m)' with the values
    of the squares that break a rule on the least-blocked way through."""
    ck = grid.cell_km
    r, c = cells[:, 0], cells[:, 1]
    in_p = (centre_values(ck, "passage")[r, c] > 0) if rules.passages else np.zeros(len(r), dtype=bool)
    land_thr, zone_thr = _thresholds(rules, in_p)
    parts = []
    if rules.required_depth_m is not None:
        depth = centre_values(ck, "depth")[r, c]
        shallow = depth < rules.required_depth_m
        if shallow.any():
            parts.append(f"too shallow (down to {max(float(depth[shallow].min()), 0):.1f} m on the deepest way "
                         f"through; the boat needs {rules.required_depth_m:g} m)")
    if rules.land_buffer_km > 0:
        land = centre_values(ck, "land")[r, c]
        narrow = land < land_thr
        if narrow.any():
            buffer_m = (rules.passage_land_buffer_km if in_p[narrow].all() else rules.land_buffer_km) * 1000
            parts.append(f"too narrow for the land buffer (the water comes within "
                         f"{float(land[narrow].min()) * 1000:.0f} m of land; it needs {buffer_m:.0f} m buffer "
                         f"+ {RASTER_TOL_KM * 1000:.0f} m tolerance for the ~100 m data grid)")
    if rules.block:
        x = grid.x0 + (c + 0.5) * ck
        y = grid.y0 + (r + 0.5) * ck
        pts = shapely.points(x, y)
        names = sorted({z.name for z in load_zones() if z.category in rules.block and float(min(
            shapely.distance(zone_geom(z, rules.turbine_radius_km), pts))) < (
            rules.zone_buffer_km if not no_buffer(z) else 0) + RASTER_TOL_KM})
        if names:
            parts.append("closed zone" + ("s" if len(names) > 1 else "") + " " + ", ".join(names))
    if not parts:
        parts.append("a gap the grid steps can't pass with these buffers")
    return f"{_place_name(cells, grid)}: {'; '.join(parts)}"


def connection_reason(lat: float, lon: float, rules: Rules, cell_km: float = CELL_KM) -> str | None:
    """Why the water at this point is cut off from the open sea with these rules, naming the
    blocking places (narrow passages, shallows, zones) or the routing-area edge. None if the
    point is not cut off (or not near water at all).

    How: on the plain water grid (land only) search from the point to the nearest square that
    IS usable with the rules, but make every square that breaks a rule very expensive, so the
    search takes the least-blocked connection. The rule-breaking squares on it are the
    bottleneck. Then close that stretch (+1 km) and look for the next connection."""
    from scipy.ndimage import binary_dilation
    from scipy.sparse import diags
    plan = build_plan(cell_km, rules, [])
    water_plan = build_plan(cell_km, LAND_ONLY, [])
    grid = plan.grid
    rows, cols = grid.water.shape
    x, y = tools._to_km(lon, lat)
    wr, wc = np.nonzero(grid.water)
    if not len(wr):
        return None
    d = np.hypot(grid.x0 + (wc + 0.5) * cell_km - x, grid.y0 + (wr + 0.5) * cell_km - y)
    i = int(np.argmin(d))
    if d[i] > MAX_SNAP_KM:
        return None
    start = (int(wr[i]), int(wc[i]))
    if plan.usable[start]:
        return None

    # Water that doesn't reach the open sea even without any rules: land or the area's edge.
    if not water_plan.usable[start]:
        _, labels = connected_components(water_plan.graph, directed=False)
        labels = labels.reshape(rows, cols)
        body = labels == labels[start]
        edges = _edges_touched(body)
        if edges:
            return ("inside the routing area this water has no connection to the open sea; "
                    f"it continues across the routing area's {' and '.join(edges)}, and a way around outside "
                    "the area can't be checked.")
        place = _place_name(np.array([start]), grid)
        where = f"{place}: " if not place.startswith("the water near") else ""
        return (f"{where}in the map data this water is enclosed by land: the coastline (Natural Earth 1:10m) and "
                f"the {cell_km:g} km grid leave no open channel to the sea here (channels narrower than about "
                f"{cell_km * 1000:.0f} m are lost).")

    ok = _usable_squares(grid, rules, []).ravel()          # each square on its own, before connectivity
    weight = 1.0 + BLOCK_COST * (~ok)
    keep = np.ones(rows * cols)
    s = start[0] * cols + start[1]
    targets = np.flatnonzero(plan.usable)
    found, reachable = [], None
    for _ in range(MAX_BLOCKAGES):
        graph = (diags(keep) @ water_plan.graph @ diags(keep * weight)).tocsr()
        graph.eliminate_zeros()
        dist, pred = dijkstra(graph, directed=True, indices=s, return_predecessors=True)
        reachable = np.isfinite(dist).reshape(rows, cols)
        reach_t = targets[np.isfinite(dist[targets])]
        if not len(reach_t):
            break
        node = int(reach_t[np.argmin(dist[reach_t])])
        path = [node]
        while node != s:
            node = int(pred[node])
            path.append(node)
        bad = [n for n in path if not ok[n]]
        if not bad:
            break                                           # blocked by the steps only (rare)
        cells = np.array([divmod(n, cols) for n in bad])
        found.append(_describe_block(cells, grid, rules))
        closed = np.zeros((rows, cols), dtype=bool)
        closed[cells[:, 0], cells[:, 1]] = True
        closed = binary_dilation(closed, iterations=max(1, round(1.0 / cell_km)))
        keep[closed.ravel()] = 0
        keep[s] = 1
    if not found:
        return None
    text = "the water here is cut off from the open sea with these settings. Blocked: " + \
           " | ".join(found) + "."
    if reachable is not None and len(found) < MAX_BLOCKAGES:
        edges = _edges_touched(reachable & ~plan.usable)
        if edges:
            text += (f" The water also continues across the routing area's {' and '.join(edges)}; "
                     "a way around outside the area can't be checked.")
    return text


# ---------------------------------------------------------------------------
# Search and smoothing
# ---------------------------------------------------------------------------

def shortest_cells(plan: Plan, start: tuple[int, int], goal: tuple[int, int]) -> tuple[list, int]:
    """Cheapest chain of squares (Dijkstra on the step graph) -> (cells, squares reached)."""
    cols = plan.usable.shape[1]
    s, g = start[0] * cols + start[1], goal[0] * cols + goal[1]
    dist, pred = dijkstra(plan.graph, directed=True, indices=s, return_predecessors=True)
    if not np.isfinite(dist[g]):
        raise RouteError("No water route between these points with these settings.")
    path, node = [g], g
    while node != s:
        node = int(pred[node])
        path.append(node)
    return [divmod(n, cols) for n in reversed(path)], int(np.isfinite(dist).sum())


def segment_ok(p, q, rules: Rules, grid: Grid, terms: list) -> bool:
    """May a straight line p -> q replace grid steps? The same rules as the steps, sampled
    every SAMPLE_KM, plus the exact land polygons."""
    line = LineString([p, q])
    if _land_exact().intersects(line):
        return False
    x, y = _sample([p, q])
    land_thr, zone_thr = _thresholds(rules, passage_raster().at(x, y) > 0)
    if (land_distance().at(x, y) < land_thr).any():
        return False
    if rules.required_depth_m is not None and (load_depth().at(x, y) < rules.required_depth_m).any():
        return False
    if rules.block:
        if _zone_union(rules.block, True, rules.turbine_radius_km).intersects(line):
            return False
        if rules.zone_buffer_km > 0:
            in_p = passage_raster().at(x, y) > 0 if rules.passages else np.zeros(len(x), dtype=bool)
            if (zone_distance(rules.block, False).at(x, y)[~in_p] < (zone_thr[~in_p] if np.ndim(zone_thr)
                                                                     else zone_thr)).any():
                return False
            if in_p.any():                                           # passage part: exact distance
                union = _zone_union(rules.block, False)
                inside_part = line.intersection(_passage_union())
                if union is not None and not inside_part.is_empty \
                        and union.distance(inside_part) < rules.passage_zone_buffer_km:
                    return False
    penalty_union = (_zone_union(rules.penalty, True, rules.turbine_radius_km)
                     if rules.penalty and rules.penalty_factor > 0 else None)
    if penalty_union is not None:
        # Straightening must not hide a penalty: no cutting through a penalty zone or its buffer.
        # (Factor 0: they cost nothing, so straight lines may cross them.)
        if penalty_union.intersects(line):
            return False
        if rules.zone_buffer_km > 0:
            in_p = passage_raster().at(x, y) > 0
            _, zone_thr = _thresholds(rules, in_p)
            if (zone_distance(rules.penalty, False).at(x, y) < zone_thr).any():
                return False
    return all(t.segment_ok(p, q, grid) for t in terms if hasattr(t, "segment_ok"))


def smooth(points, rules: Rules, grid: Grid, terms: list, extra_ok=None):
    """String pulling: from each kept point, extend the straight segment forward as long as
    it passes segment_ok() (and extra_ok, e.g. no rough squares for the weather-aware route)."""
    if len(points) <= 2:
        return points
    kept, i = [points[0]], 0
    while i < len(points) - 1:
        j = i + 1                                    # a grid step: allowed by construction
        while j + 1 < len(points) and segment_ok(points[i], points[j + 1], rules, grid, terms) \
                and (extra_ok is None or extra_ok(points[i], points[j + 1])):
            j += 1
        kept.append(points[j])
        i = j
    return kept


def _length(points) -> float:
    return sum(math.dist(p, q) for p, q in zip(points, points[1:]))


def eta_text(hours: float) -> str:
    total = round(hours * 60)
    h, m = divmod(total, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def _lat_lon(p_km) -> list[float]:
    lon, lat = to_lonlat(*p_km)
    return [round(float(lat), 5), round(float(lon), 5)]


def zones_on(points, turbine_km: float = TURBINE_RADIUS_M / 1000) -> dict[str, list[str]]:
    """Names of the zones the line touches, per category (a turbine: within its radius)."""
    if len(points) < 2:
        return {}
    line = LineString(points)
    found = {}
    for z in load_zones():
        if (z.geom_km.distance(line) < turbine_km) if z.kind == "turbine" else z.geom_km.intersects(line):
            found.setdefault(z.category, [])
            if z.name not in found[z.category]:
                found[z.category].append(z.name)
    return found


def _zone_distance_exact(points, categories, include_tiny: bool = False) -> tuple[float | None, str | None]:
    """Exact distance from the route to the nearest zone of these categories (and its name).
    Tiny unnamed zones and turbines have no zone buffer, so by default they don't count here."""
    if not categories:
        return None, None
    geom = LineString(points) if len(points) > 1 else Point(points[0])
    return min(((z.geom_km.distance(geom), z.name) for z in load_zones()
                if z.category in categories and (include_tiny or not no_buffer(z))), default=(None, None))


def turbine_distance_exact(points) -> tuple[float | None, str | None]:
    """Exact distance from the route to the nearest offshore wind turbine (and its farm's name)."""
    geom = LineString(points) if len(points) > 1 else Point(points[0])
    return min(((z.geom_km.distance(geom), z.name) for z in load_zones() if z.kind == "turbine"),
               default=(None, None))


def passages_used(points, rules: Rules) -> list[str]:
    """Narrow passages where the route actually uses the reduced clearance (closer to land or
    a closed zone than the open-water buffer allows)."""
    if not rules.passages or len(points) < 2:
        return []
    x, y = _sample(points)
    inside = passage_raster().at(x, y) > 0
    if not inside.any():
        return []
    tight = land_distance().at(x, y) < _threshold(rules.land_buffer_km)
    if rules.block and rules.zone_buffer_km > 0:
        tight |= zone_distance(rules.block, False).at(x, y) < _threshold(rules.zone_buffer_km)
    tight &= inside
    if not tight.any():
        return []
    from shapely.geometry import Polygon
    used = []
    for p in NARROW_PASSAGES:
        poly = shapely.transform(Polygon(p["polygon"]), lambda xy: xy * [tools._KM_PER_DEG_LON, tools._KM_PER_DEG_LAT])
        if shapely.contains_xy(poly, x[tight], y[tight]).any():
            used.append(p["name"])
    return used


VIA_MIN_KM = 1.0


def route_via(points) -> list[str]:
    """Named passages and landmarks the route goes through, in order ("around Møn", "Grønsund").
    A place counts if at least VIA_MIN_KM of the route is inside it (not a corner touch)."""
    if len(points) < 2:
        return []
    from shapely.geometry import Polygon
    x, y = _sample(points)
    hits = []
    for p in NARROW_PASSAGES + ROUTE_LANDMARKS:
        poly = shapely.transform(Polygon(p["polygon"]), lambda xy: xy * [tools._KM_PER_DEG_LON, tools._KM_PER_DEG_LAT])
        inside = np.flatnonzero(shapely.contains_xy(poly, x, y))
        if len(inside) * SAMPLE_KM >= VIA_MIN_KM:
            hits.append((int(inside[0]), p["name"]))
    return [name for _, name in sorted(hits)]


def length_inside(points, categories, turbine_km: float = TURBINE_RADIUS_M / 1000) -> float:
    """km of the route inside zones of these categories (turbines: inside their radius)."""
    union = _zone_union(categories, True, turbine_km) if categories else None
    if union is None or len(points) < 2:
        return 0.0
    return float(LineString(points).intersection(union).length)


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

ROUGH_PENALTY = 5.0          # weather-aware route: each km through a rough square costs 5 km extra


def _best_line(candidates, rules: Rules, grid: Grid, terms: list, extra_ok=None):
    """Straighten each candidate chain of squares both ways and keep the shortest line."""
    lines = []
    for cells in candidates:
        centres = [grid.center(c) for c in cells]
        lines.append(smooth(centres, rules, grid, terms, extra_ok))
        lines.append(smooth(centres[::-1], rules, grid, terms, extra_ok)[::-1])
    return min(lines, key=lambda pts: (round(_length(pts), 6), len(pts)))


def _departure(departure) -> "pd.Timestamp":
    import pandas as pd
    if departure is None:
        return pd.Timestamp.now(tz="UTC").floor("min")
    t = pd.Timestamp(departure)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _leg_midpoints(path):
    """Points every ~LEG_KM along the route (where the legs' weather is looked up)."""
    import route_weather as rw
    pts = [tools._to_km(lon, lat) for lat, lon in path]
    out = [path[0]]
    for p, q in zip(pts, pts[1:]):
        n = max(1, math.ceil(math.dist(p, q) / rw.LEG_KM))
        for k in range(1, n + 1):
            x, y = p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n
            out.append(_lat_lon((x, y)))
    return out


def _rough_squares(plan: "Plan", start, field, departure, speed_kn: float, limits: dict) -> np.ndarray:
    """Usable squares that are too rough at the time the boat would get there (arrival time =
    shortest distance from the start / speed: the time dependence, approximated)."""
    import pandas as pd
    import route_weather as rw
    rows, cols = plan.usable.shape
    dist = dijkstra(plan.graph, directed=True, indices=start[0] * cols + start[1]).reshape(rows, cols)
    r, c = np.nonzero(plan.usable & np.isfinite(dist))
    x = plan.grid.x0 + (c + 0.5) * plan.grid.cell_km
    y = plan.grid.y0 + (r + 0.5) * plan.grid.cell_km
    lon, lat = to_lonlat(x, y)
    idx = field.nearest(lat, lon)
    hours = (departure - field.times[0]) / pd.Timedelta(hours=1) + dist[r, c] / (speed_kn * KM_PER_NM)
    inside = (hours >= -0.5) & (hours <= len(field.times) - 0.5)        # beyond the forecast: unknown
    h = np.clip(np.round(hours).astype(int), 0, len(field.times) - 1)
    wind = field.wind_ms[idx, h]
    bft = np.searchsorted(tools.BEAUFORT_LIMITS, wind, side="right")
    rough = bft > limits["wind_limit_bft"]
    if field.waves_m is not None:
        waves = field.waves_m[idx, h]
        rough |= np.nan_to_num(waves, nan=0.0) > limits["waves_limit_m"]
    mask = np.zeros((rows, cols), dtype=bool)
    mask[r[inside & rough], c[inside & rough]] = True
    return mask


def _with_rough_penalty(plan: "Plan", rough: np.ndarray, penalty: float):
    """The step graph with an extra cost for every step INTO a rough square."""
    g = plan.graph.tocsr(copy=True)
    rows, cols = rough.shape
    src = np.repeat(np.arange(g.shape[0]), np.diff(g.indptr))
    dst = g.indices
    step = plan.grid.cell_km * np.hypot(dst // cols - src // cols, dst % cols - src % cols)
    g.data = g.data + penalty * step * rough.ravel()[dst]
    return g


def _rough_segment_check(grid: Grid, rough: np.ndarray):
    """Straightened lines of the weather-aware route may not cut through rough squares."""
    rows, cols = rough.shape

    def ok(p, q):
        x, y = _sample([p, q])
        c = ((x - grid.x0) // grid.cell_km).astype(int)
        r = ((y - grid.y0) // grid.cell_km).astype(int)
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        return not rough[r[inside], c[inside]].any()
    return ok


def route(from_lat: float, from_lon: float, to_lat: float, to_lon: float,
          speed_kn: float = DEFAULT_SPEED_KN, cell_km: float = CELL_KM,
          terms: list | None = None, departure: datetime | None = None,
          draft_m: float = DRAFT_M, margin_m: float = SAFETY_MARGIN_M, zones: str = "block",
          land_buffer_m: float = LAND_BUFFER_M, zone_buffer_m: float = ZONE_BUFFER_M,
          rules: Rules | None = None, passages: bool = True,
          passage_land_buffer_m: float = PASSAGE_LAND_BUFFER_M,
          passage_zone_buffer_m: float = PASSAGE_ZONE_BUFFER_M, penalty_factor: float = ZONE_PENALTY,
          alternative: bool = True, weather: str = "off", wind_limit_bft: int = None,
          waves_limit_m: float = None, rough_penalty: float = None, wind_farms: str = "block",
          turbine_radius_m: float = TURBINE_RADIUS_M) -> dict:
    """The cheapest water route between two points. Raises RouteError with a clear message.
    terms=None: the phase 2 rules from the settings (depth, zones, buffers, narrow passages),
    or `rules`. terms=[...]: land only plus exactly these custom terms (terms=[] is phase 1).
    If the route uses a narrow passage, `alternative` adds the full-buffer route for comparison.
    weather: "off" | "show" (legs with the weather at their passing time) | "aware" (also avoid
    rough legs as a cost; the shortest route is returned for comparison)."""
    if not speed_kn > 0:
        raise RouteError("Speed must be more than 0 knots.")
    if weather not in ("off", "show", "aware"):
        raise RouteError("weather must be 'off', 'show' or 'aware'.")
    with_rules = terms is None
    if rules is None:
        rules = (make_rules(draft_m, margin_m, zones, land_buffer_m, zone_buffer_m, passages,
                            passage_land_buffer_m, passage_zone_buffer_m, penalty_factor,
                            wind_farms, turbine_radius_m)
                 if with_rules else LAND_ONLY)
    terms = terms or []
    plan = build_plan(cell_km, rules, terms)
    grid = plan.grid
    try:
        start, start_snap = snap(grid, plan.usable, from_lat, from_lon, rules)
    except RouteError as exc:
        raise RouteError(f"Start: {exc}") from None
    try:
        goal, goal_snap = snap(grid, plan.usable, to_lat, to_lon, rules)
    except RouteError as exc:
        raise RouteError(f"Destination: {exc}") from None

    if start == goal:
        candidates, reached = [[start]], 1
    else:
        # Equal-cost paths are broken differently from each end, and smoothing is greedy:
        # search from both ends and straighten both ways, then keep the shortest line.
        # So A -> B and B -> A always pick from the same candidates (same distance).
        there, reached = shortest_cells(plan, start, goal)
        back, _ = shortest_cells(plan, goal, start)
        candidates = [there, back[::-1]]
    points = _best_line(candidates, rules, grid, terms)
    weather_info = None
    if weather != "off":
        import route_weather as rw
        limits = dict(wind_limit_bft=rw.ROUGH_WIND_BFT if wind_limit_bft is None else int(wind_limit_bft),
                      waves_limit_m=rw.ROUGH_WAVES_M if waves_limit_m is None else float(waves_limit_m))
        dep = _departure(departure)
        shortest_path = [_lat_lon(p) for p in points]
        ends = [_lat_lon(grid.center(start)), _lat_lon(grid.center(goal))]
        try:
            if weather == "aware":
                field = rw.get_field(rw.corridor_points(shortest_path, ends))
                shortest_legs = rw.legs_along(shortest_path, dep, speed_kn, field, **limits)
                rough = _rough_squares(plan, start, field, dep, speed_kn, limits)
                if rough.any() and start != goal:
                    graph = _with_rough_penalty(plan, rough, ROUGH_PENALTY if rough_penalty is None else rough_penalty)
                    weighted = replace(plan, graph=graph)
                    there, reached = shortest_cells(weighted, start, goal)
                    back, _ = shortest_cells(weighted, goal, start)
                    not_rough = _rough_segment_check(grid, rough)
                    points = _best_line([there, back[::-1]], rules, grid, terms, extra_ok=not_rough)
            else:
                field = None
            path_now = [_lat_lon(p) for p in points]
            if field is None:
                field = rw.get_field([(lat, lon) for lat, lon in _leg_midpoints(path_now)])
            legs = rw.legs_along(path_now, dep, speed_kn, field, **limits)
        except rw.RouteWeatherError as exc:
            raise RouteError(str(exc)) from exc
        weather_info = {"variant": weather, "departure_utc": dep.strftime("%Y-%m-%dT%H:%MZ"),
                        "legs": legs, **rw.summarize(legs),
                        "limits": {**limits, "caution_from_bft": rw.CAUTION_BFT, "caution_waves_m": rw.CAUTION_WAVES_M},
                        "source": field.source, "note": field.note}
        if weather == "aware":
            short_sum = rw.summarize(shortest_legs)
            changed = [_lat_lon(p) for p in points] != shortest_path
            if changed and weather_info["rough_count"] >= short_sum["rough_count"]:
                # Not calmer: keep the shortest route (an avoiding detour that doesn't help is no use).
                points = [tools._to_km(lon, lat) for lat, lon in shortest_path]
                weather_info.update({"legs": shortest_legs, **short_sum,
                                     "note": ((weather_info["note"] + "; ") if weather_info["note"] else "")
                                     + "no calmer route found, so this is the shortest route"})
                changed = False
            short_km = _length([tools._to_km(lon, lat) for lat, lon in shortest_path])
            weather_info["shortest"] = {"distance_km": round(short_km, 2),
                                        "eta_text": eta_text(short_km / KM_PER_NM / speed_kn),
                                        "path": shortest_path, "legs": shortest_legs, **rw.summarize(shortest_legs)}
            weather_info["avoided_rough_legs"] = ([g for g in shortest_legs if g["status"] == "red"]
                                                  if changed else [])
    distance_km = _length(points)
    distance_nm = distance_km / KM_PER_NM
    eta_hours = distance_nm / speed_kn

    result = {
        "path": [_lat_lon(p) for p in points],
        "distance_km": round(distance_km, 2),
        "distance_nm": round(distance_nm, 2),
        "speed_kn": speed_kn,
        "eta_hours": round(eta_hours, 3),
        "eta_text": eta_text(eta_hours),
        "start": {"clicked": [from_lat, from_lon], "snapped": _lat_lon(grid.center(start)),
                  "snap_km": round(start_snap, 2), "name": place_name(from_lat, from_lon)},
        "end": {"clicked": [to_lat, to_lon], "snapped": _lat_lon(grid.center(goal)),
                "snap_km": round(goal_snap, 2), "name": place_name(to_lat, to_lon)},
        "cells_expanded": reached,
        "cell_km": cell_km,
        "snap_warnings": [f"The {label} was moved {snapped:.1f} km to the nearest usable water: "
                          f"{point_reason(lat, lon, rules)}."
                          for label, snapped, lat, lon in (("start", start_snap, from_lat, from_lon),
                                                           ("destination", goal_snap, to_lat, to_lon))
                          if snapped > SNAP_WARN_KM],
        "cost_terms": ([n for n, on in [("depth", rules.required_depth_m is not None),
                                        ("zones (block)", bool(rules.block)),
                                        ("zones (penalty)", bool(rules.penalty)),
                                        ("buffers", rules.land_buffer_km > 0 or rules.zone_buffer_km > 0)] if on]
                       + [t.name for t in terms]),
    }
    if weather_info:
        result["weather"] = weather_info
    if with_rules:
        min_depth, where = depths_along(points)
        land_km, _ = _min_along(land_distance(), points)
        zone_km, zone_name = _zone_distance_exact(points, rules.block)
        turbine_km, turbine_name = turbine_distance_exact(points)
        on = zones_on(points, rules.turbine_radius_km)
        rule_of = zone_rules(zones, wind_farms)
        used = passages_used(points, rules)
        alt = None
        if used and alternative:
            try:
                full = replace(rules, passages=False)
                a = route(from_lat, from_lon, to_lat, to_lon, speed_kn, cell_km, rules=full,
                          draft_m=draft_m, margin_m=margin_m, zones=zones, wind_farms=wind_farms,
                          turbine_radius_m=turbine_radius_m, alternative=False)
                if a["snap_warnings"]:
                    # Moving the start / destination elsewhere is not a fair comparison.
                    raise RouteError("not possible from these points (" + " ".join(a["snap_warnings"]) + ")")
                alt = {"distance_km": a["distance_km"], "distance_nm": a["distance_nm"], "eta_text": a["eta_text"],
                       "eta_hours": a["eta_hours"], "path": a["path"],
                       "extra_km": round(a["distance_km"] - distance_km, 2),
                       "label": "with full buffers everywhere (no narrow-passage exception)"}
            except RouteError as exc:
                alt = {"error": str(exc), "label": "with full buffers everywhere (no narrow-passage exception)"}
        result.update({
            "min_depth_m": round(min_depth, 1),
            "min_depth_at": _lat_lon(where),
            "required_depth_m": rules.required_depth_m,
            "min_land_distance_km": round(land_km, 2),
            "min_zone_distance_km": None if zone_km is None else round(zone_km, 2),
            "nearest_zone": zone_name,
            "min_turbine_distance_km": None if turbine_km is None else round(turbine_km, 2),
            "nearest_turbine": turbine_name,
            "turbine_radius_m": rules.turbine_radius_km * 1000,
            "wind_farms_crossed": on.get("wind_farm", []),
            "zones_crossed": sorted({n for c, names in on.items() if rule_of.get(c) != "ignore" for n in names}),
            "exercise_areas_crossed": on.get("exercise", []),    # passage allowed, information only
            "passages_used": used,
            "via": route_via(points),
            "passage_warning": f"{', '.join(used)}: {PASSAGE_WARNING}" if used else None,
            "alternative": alt,
            "zone_km_inside_penalty": (round(length_inside(points, rules.penalty, rules.turbine_radius_km), 2)
                                       if rules.penalty else 0.0),
            "settings": {"draft_m": draft_m, "margin_m": margin_m, "zones": zones, "wind_farms": wind_farms,
                         "turbine_radius_m": rules.turbine_radius_km * 1000,
                         "land_buffer_m": rules.land_buffer_km * 1000, "zone_buffer_m": rules.zone_buffer_km * 1000,
                         "passages": rules.passages,
                         "passage_land_buffer_m": rules.passage_land_buffer_km * 1000,
                         "passage_zone_buffer_m": rules.passage_zone_buffer_km * 1000,
                         "penalty_factor": rules.penalty_factor, "rules": rule_of},
            "sources": SOURCES,
            "note": ("Depth from a ~100 m bathymetry model (not a nautical chart), firing areas treated as "
                     "always active, wind farms and turbines from OpenStreetMap (may be incomplete), "
                     "no traffic. " + NOT_FOR_NAVIGATION),
        })
    return result


# ---------------------------------------------------------------------------
# Explaining the route: which rule caused how much detour
# ---------------------------------------------------------------------------

def _side(points, a_km, b_km) -> dict | None:
    """On which side of the direct line the route bulges out, and how far at most."""
    ax, ay = a_km
    bx, by = b_km
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-6 or len(points) < 2:
        return None
    nx, ny = -(by - ay) / length, (bx - ax) / length           # left normal of the direct line
    offsets = [(x - ax) * nx + (y - ay) * ny for x, y in points]
    far = max(offsets, key=abs)
    if abs(far) < 0.3:
        return {"side": "along the direct line", "max_offset_km": round(abs(far), 1)}
    sx, sy = (nx, ny) if far > 0 else (-nx, -ny)                # direction of the bulge
    side = ("north" if sy > 0 else "south") if abs(sy) >= abs(sx) else ("east" if sx > 0 else "west")
    return {"side": side, "max_offset_km": round(abs(far), 1)}


def _extra_text(extra: float | None) -> str | None:
    """How a step's extra km is shown: small values are grid / straightening noise."""
    if extra is None:
        return None
    if abs(extra) < NOISE_KM:
        return "≈ 0 km (grid rounding)"
    return f"{'+' if extra > 0 else ''}{extra} km"


def _names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def side_sentence(side: dict | None) -> str:
    """'The route runs up to 11.6 km north of the direct line.' (or '' if it doesn't bulge)."""
    if not side or side["side"] == "along the direct line":
        return ""
    return f"The route runs up to {side['max_offset_km']} km {side['side']} of the direct line."


def main_reason_sentence(main: dict | None, side: dict | None, all_zones: list[str] | None = None) -> str:
    """The main reason as one plain sentence, built in code. Always names the avoided zones
    (also when the largest step is another rule)."""
    if main is None:
        text = "Main reason: there is no real detour; the route is close to the shortest water route."
    elif main.get("zones_avoided"):
        zones = main["zones_avoided"]
        farms = [z for z in zones if z in main.get("wind_farms_avoided", [])]
        n_other = len(zones) - len(farms)
        parts = ([f"restricted zone{'s' if n_other > 1 else ''}"] if n_other else []) + \
                ([f"wind farm{'s' if len(farms) > 1 else ''}"] if farms else [])
        verb = "adds" if len(zones) == 1 else "add"
        text = (f"Main reason: the {' and '.join(parts)} {verb} {main['extra_km']} km, because the route avoids "
                f"{_names(zones)}.")
    elif main["step"].startswith("+ weather"):
        text = f"Main reason: the weather adds {main['extra_km']} km" + (
            f" ({main['reason']})." if main.get("reason") else ".")
    elif main["step"].startswith("+ depth"):
        text = f"Main reason: the depth rule adds {main['extra_km']} km" + (
            f" ({main['reason']})." if main.get("reason") else ".")
    elif main["step"].startswith("+ safety buffers"):
        text = f"Main reason: the safety buffers add {main['extra_km']} km" + (
            f" ({main['reason']})." if main.get("reason") else ".")
    else:
        text = f"Main reason: {main['step'].removeprefix('+ ')} adds {main['extra_km']} km."
    named = set(main.get("zones_avoided", []) if main else [])
    others = [z for z in (all_zones or []) if z not in named]
    if others:
        text += f" The route also avoids {_names(others)}."
    extra = side_sentence(side)
    return f"{text} {extra}" if extra else text


def explain(from_lat: float, from_lon: float, to_lat: float, to_lon: float,
            speed_kn: float = DEFAULT_SPEED_KN, draft_m: float = DRAFT_M, margin_m: float = SAFETY_MARGIN_M,
            zones: str = "block", land_buffer_m: float = LAND_BUFFER_M, zone_buffer_m: float = ZONE_BUFFER_M,
            cell_km: float = CELL_KM, passages: bool = True,
            passage_land_buffer_m: float = PASSAGE_LAND_BUFFER_M,
            passage_zone_buffer_m: float = PASSAGE_ZONE_BUFFER_M, penalty_factor: float = ZONE_PENALTY,
            weather: str = "off", departure=None, wind_limit_bft: int = None, waves_limit_m: float = None,
            rough_penalty: float = None, wind_farms: str = "block", turbine_radius_m: float = TURBINE_RADIUS_M) -> dict:
    """The route plus a detour breakdown, all computed in code: the same trip with the rules
    added one by one (land only -> + depth -> + zones -> + buffers). Steps within +-NOISE_KM
    are shown as "≈ 0 km (grid rounding)"; adding a rule can't really make a route shorter."""
    full = make_rules(draft_m, margin_m, zones, land_buffer_m, zone_buffer_m, passages,
                      passage_land_buffer_m, passage_zone_buffer_m, penalty_factor, wind_farms, turbine_radius_m)
    depth_only = Rules(required_depth_m=full.required_depth_m)
    with_zones = replace(depth_only, block=full.block, penalty=full.penalty, penalty_factor=full.penalty_factor,
                         turbine_radius_km=full.turbine_radius_km)
    buffer_label = f"+ safety buffers ({land_buffer_m:g} m from land, {zone_buffer_m:g} m from closed zones"
    buffer_label += (f"; {full.passage_land_buffer_km * 1000:g} / {full.passage_zone_buffer_km * 1000:g} m "
                     "in narrow passages)" if passages else ")")
    stages = [
        ("direct water route (land only)", LAND_ONLY),
        (f"+ depth rule (at least {full.required_depth_m:g} m)", depth_only),
        (f"+ restricted zones and wind farms (turbines {full.turbine_radius_km * 1000:g} m)", with_zones),
        (buffer_label, full),
    ]
    common = dict(draft_m=draft_m, margin_m=margin_m, zones=zones, wind_farms=wind_farms,
                  turbine_radius_m=turbine_radius_m)
    wx = dict(departure=departure, wind_limit_bft=wind_limit_bft, waves_limit_m=waves_limit_m,
              rough_penalty=rough_penalty)
    final = route(from_lat, from_lon, to_lat, to_lon, speed_kn, cell_km, rules=full, weather=weather, **wx, **common)
    # Every step starts and ends at the FINAL route's snapped points (usable under all rules,
    # since each step only adds rules), so the steps compare the same trip.
    (s_lat, s_lon), (e_lat, e_lon) = final["start"]["snapped"], final["end"]["snapped"]
    rows, previous = [], None
    for name, rules in stages:
        try:
            r = route(s_lat, s_lon, e_lat, e_lon, speed_kn, cell_km, rules=rules, alternative=False, **common)
            if rules == full and weather != "aware":
                r = final
        except RouteError as exc:
            rows.append({"step": name, "distance_km": None, "extra_km": None, "extra_text": None,
                         "reason": str(exc)})
            previous = None
            continue
        pts = [tools._to_km(lon, lat) for lat, lon in r["path"]]
        extra = None if previous is None else round(r["distance_km"] - previous["distance_km"], 2)
        row = {"step": name, "distance_km": r["distance_km"], "extra_km": extra, "extra_text": _extra_text(extra),
               "noise": extra is not None and abs(extra) < NOISE_KM}
        if previous is not None:
            before, prev_pts = previous["rules"], previous["pts"]
            if rules.required_depth_m is not None and before.required_depth_m is None:
                d, at = depths_along(prev_pts)
                if d < rules.required_depth_m:
                    lat, lon = _lat_lon(at)
                    row["reason"] = (f"the direct route crosses water only {max(d, 0):.1f} m deep at "
                                     f"{lat:.3f}° N, {lon:.3f}° E; the boat needs {rules.required_depth_m:g} m")
            if (rules.block or rules.penalty) and not (before.block or before.penalty):
                on_prev = zones_on(prev_pts, rules.turbine_radius_km)
                blocked = {n for c, names in on_prev.items() if c in rules.block for n in names}
                # Penalty zones count as avoided when this route goes around them (cheaper than crossing).
                on_now = {n for names in zones_on(pts, rules.turbine_radius_km).values() for n in names}
                penalised = ({n for c, names in on_prev.items() if c in rules.penalty for n in names} - on_now
                             if rules.penalty_factor > 0 else set())
                avoided = sorted(blocked | penalised)
                if avoided:
                    farms = set(on_prev.get("wind_farm", [])) & set(avoided)

                    def label(name):            # "Rødsand II (wind farm; penalty: going around is cheaper)"
                        notes = (["wind farm"] if name in farms else []) + \
                                (["penalty: going around is cheaper"] if name in penalised - blocked else [])
                        return f"{name} ({'; '.join(notes)})" if notes else name
                    row["reason"] = "avoids " + ", ".join(label(n) for n in avoided)
                    row["zones_avoided"] = avoided
                    if farms:
                        row["wind_farms_avoided"] = sorted(farms)
            if (rules.land_buffer_km or rules.zone_buffer_km) and not (before.land_buffer_km or before.zone_buffer_km):
                land_km, _ = _min_along(land_distance(), prev_pts)
                zone_km, zone_name = _zone_distance_exact(prev_pts, rules.block)
                parts = []
                if land_km < rules.land_buffer_km:
                    parts.append(f"about {land_km * 1000:.0f} m of land")
                if zone_km is not None and zone_km < rules.zone_buffer_km:
                    parts.append(f"{zone_km * 1000:.0f} m of {zone_name}")
                if parts:
                    row["reason"] = "without buffers the route came within " + " and ".join(parts)
                if r.get("passage_warning"):
                    row["passage_warning"] = r["passage_warning"]
        rows.append(row)
        previous = {"distance_km": r["distance_km"], "pts": pts, "rules": rules}

    if weather == "aware" and previous is not None:
        # 5th step: the weather-aware route vs. the shortest route with all other rules.
        extra = round(final["distance_km"] - previous["distance_km"], 2)
        wi = final["weather"]
        avoided = wi.get("avoided_rough_legs", [])
        row = {"step": "+ weather (avoid rough legs)", "distance_km": final["distance_km"], "extra_km": extra,
               "extra_text": _extra_text(extra), "noise": abs(extra) < NOISE_KM}
        if avoided:
            row["reason"] = (f"the shortest route has {len(avoided)} rough leg(s), e.g. at "
                             f"{avoided[0]['time_utc'][11:16]} UTC near {avoided[0]['lat']:.2f}° N, "
                             f"{avoided[0]['lon']:.2f}° E (Beaufort {avoided[0]['wind_bft']}"
                             + (f", waves {avoided[0]['waves_m']} m" if avoided[0]["waves_m"] is not None else "")
                             + f"); this route has {wi['rough_count']}")
        rows.append(row)

    alternatives = []
    if final.get("alternative"):
        a = final["alternative"]
        alternatives.append({"kind": "full buffers", "label": a["label"], **{k: a.get(k) for k in
                             ("distance_km", "eta_text", "extra_km", "error")}})
    if full.penalty:
        # What crossing the penalty zones would mean: the route if they were ignored.
        free = replace(full, penalty=())
        try:
            c = route(s_lat, s_lon, e_lat, e_lon, speed_kn, cell_km, rules=free, alternative=False, **common)
            c_pts = [tools._to_km(lon, lat) for lat, lon in c["path"]]
            inside = length_inside(c_pts, full.penalty, full.turbine_radius_km)
            if inside > 0.01:
                saves = round(final["distance_km"] - c["distance_km"], 2)
                text = (f"crossing would save {max(saves, 0)} km but its {inside:.2f} km inside "
                        f"{', '.join(sorted({n for cat, names in zones_on(c_pts, full.turbine_radius_km).items() if cat in full.penalty for n in names}))} "
                        f"count as {inside * full.penalty_factor:.2f} km penalty (factor {full.penalty_factor:g})")
                if final["zone_km_inside_penalty"] > 0.01:
                    text = (f"this route crosses {final['zone_km_inside_penalty']} km of penalty zones, counted as "
                            f"{final['zone_km_inside_penalty'] * full.penalty_factor:.2f} km (factor "
                            f"{full.penalty_factor:g}); going around would add more than that")
                alternatives.append({"kind": "penalty", "label": text, "distance_km": c["distance_km"],
                                     "inside_km": round(inside, 2),
                                     "penalty_km": round(inside * full.penalty_factor, 2), "saves_km": saves})
        except RouteError:
            pass

    real = [r for r in rows[1:] if r.get("extra_km") is not None and not r.get("noise")]
    main = max(real, key=lambda r: r["extra_km"], default=None)
    a_km = tools._to_km(final["start"]["snapped"][1], final["start"]["snapped"][0])
    b_km = tools._to_km(final["end"]["snapped"][1], final["end"]["snapped"][0])
    final_pts = [tools._to_km(lon, lat) for lat, lon in final["path"]]
    if main and main["extra_km"] >= NOISE_KM:
        main_reason = (f"{main['step'].removeprefix('+ ')}: +{main['extra_km']} km"
                       + (f" ({main['reason']})" if main.get("reason") else ""))
    else:
        main_reason = "no real detour: the route is close to the shortest water route"
    side = _side(final_pts, a_km, b_km)
    main_zones = list(main.get("zones_avoided", [])) if main and main["extra_km"] >= NOISE_KM else []
    return {
        "route": final,
        "summary": {
            "distance_km": final["distance_km"], "distance_nm": final["distance_nm"], "eta": final["eta_text"],
            "speed_kn": speed_kn, "straight_line_km": round(math.dist(a_km, b_km), 2),
            "min_depth_m": final["min_depth_m"], "required_depth_m": final["required_depth_m"],
            "draft_m": draft_m, "margin_m": margin_m,
            "min_land_distance_km": final["min_land_distance_km"], "land_buffer_m": land_buffer_m,
            "min_zone_distance_km": final["min_zone_distance_km"], "nearest_zone": final["nearest_zone"],
            "zone_buffer_m": zone_buffer_m, "zone_mode": zones, "penalty_factor": full.penalty_factor,
            "zones_crossed": final["zones_crossed"], "exercise_areas_crossed": final["exercise_areas_crossed"],
            "zones_avoided": sorted({n for r in rows for n in r.get("zones_avoided", [])}),
            "wind_farms_avoided": sorted({n for r in rows for n in r.get("wind_farms_avoided", [])}),
            "wind_farm_mode": wind_farms, "wind_farms_crossed": final["wind_farms_crossed"],
            "min_turbine_distance_km": final["min_turbine_distance_km"], "nearest_turbine": final["nearest_turbine"],
            "turbine_radius_m": final["turbine_radius_m"],
            "passages_used": final["passages_used"], "passage_warning": final["passage_warning"],
            "via": final["via"],
            "detour_side": side,
        },
        "detour": rows,
        "alternatives": alternatives,
        "main_reason": main_reason,
        "main_reason_sentence": main_reason_sentence(main if main and main["extra_km"] >= NOISE_KM else None, side,
                                                     sorted({n for r in rows for n in r.get("zones_avoided", [])})),
        "weather": final.get("weather"),
        "main_zones": main_zones,
        "note": final["note"],
    }


def shallow_mask(draft_m: float = DRAFT_M, margin_m: float = SAFETY_MARGIN_M, cell_km: float = CELL_KM) -> dict:
    """Water squares shallower than draft + margin, for the map layer (rows south -> north,
    packed bits, base64)."""
    grid = build_grid(cell_km)
    required = round(draft_m + margin_m, 2)
    mask = grid.water & (centre_values(cell_km, "depth") < required)
    return {"required_depth_m": required, "rows": int(mask.shape[0]), "cols": int(mask.shape[1]),
            "bounds": grid.lonlat_bounds(),
            "bits": base64.b64encode(np.packbits(mask.ravel())).decode("ascii")}


def warm_up():
    """Build the default grid, rasters and plan once (a few seconds), e.g. at app start."""
    build_plan(CELL_KM, make_rules(), [])
