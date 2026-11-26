"""Sea routing, phase 1: the shortest WATER route between two points (code only, no LLM).

How it works:
  1. Grid: ~1 km squares over Kiel Bay (ROUTE_BBOX), in the same local km grid as the
     land/sea check (tools._to_km). A square is water only if it doesn't touch the land
     polygons at all (data/land_kiel_bay.geojson, unshrunk).
  2. Snap: each clicked point goes to the nearest water square (at most MAX_SNAP_KM away).
  3. A*: 8 neighbours; a diagonal step only if both side squares are water too, so every
     step stays inside water squares. Heuristic: straight-line distance.
  4. Smooth: the grid zigzag is straightened ("string pulling") wherever the straight line
     keeps SAFETY_MARGIN_KM away from land.
  5. Distance in km and nm, ETA for the chosen speed.

Cost function: edge cost = step length (km) + the extra cost of every term in COST_TERMS.
Phase 1 has no terms. Later phases add terms without touching the search:
  - depth (EMODnet Bathymetry): blocked() for squares that are too shallow
  - restricted zones: blocked() and segment_ok() for the zones
  - weather (Open-Meteo grid, wind + waves): extra() from the forecast at the ARRIVAL time
    (ctx.departure + g / speed), i.e. time-dependent
  - traffic (EMODnet vessel density): extra() for busy lanes
Every extra() must be >= 0, so the straight-line heuristic stays admissible (A* stays exact).
"""
import functools
import heapq
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.prepared import prep

import tools

ROUTE_BBOX = (9.8, 54.15, 11.6, 55.15)   # lon_min, lat_min, lon_max, lat_max (Kiel Bay + Fehmarn)
CELL_KM = 1.0                            # grid resolution
MAX_SNAP_KM = 5.0                        # a click farther than this from water is rejected
SAFETY_MARGIN_KM = 0.2                   # smoothed lines keep at least this far from land
KM_PER_NM = 1.852
DEFAULT_SPEED_KN = 6.0
STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]   # (row, col)


class RouteError(ValueError):
    """The route can't be computed (outside the area, no water nearby, no connection)."""


# ---------------------------------------------------------------------------
# Cost terms (phase 1: none). The interface for later phases.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteContext:
    speed_kn: float
    departure: datetime | None = None    # for time-dependent terms (weather), unused in phase 1


class CostTerm(Protocol):
    name: str

    def blocked(self, cell: tuple[int, int], grid: "Grid") -> bool:
        """Hard limit: this square may not be used at all (too shallow, restricted)."""

    def extra(self, a: tuple[int, int], b: tuple[int, int], step_km: float, ctx: RouteContext) -> float:
        """Extra cost (>= 0, in km-equivalents) for the step a -> b."""

    def segment_ok(self, p_km: tuple[float, float], q_km: tuple[float, float], grid: "Grid") -> bool:
        """May smoothing replace grid steps by the straight segment p -> q?"""


COST_TERMS: list[CostTerm] = []          # phase 1: distance only


# ---------------------------------------------------------------------------
# Grid
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


def to_lonlat(x_km: float, y_km: float) -> tuple[float, float]:
    """Inverse of tools._to_km."""
    return x_km / tools._KM_PER_DEG_LON, y_km / tools._KM_PER_DEG_LAT


@functools.lru_cache(maxsize=4)
def build_grid(cell_km: float = CELL_KM) -> Grid:
    """Water squares of ROUTE_BBOX. Built once per resolution (about 0.1 s at 1 km)."""
    land = tools._land_km()
    shapely.prepare(land)
    lon0, lat0, lon1, lat1 = ROUTE_BBOX
    x0, y0 = tools._to_km(lon0, lat0)
    x1, y1 = tools._to_km(lon1, lat1)
    cols, rows = int((x1 - x0) // cell_km), int((y1 - y0) // cell_km)
    xs = x0 + np.arange(cols) * cell_km
    ys = y0 + np.arange(rows) * cell_km
    gx, gy = np.meshgrid(xs, ys)                       # (rows, cols)
    squares = shapely.box(gx, gy, gx + cell_km, gy + cell_km)
    water = ~shapely.intersects(land, squares)         # touches land at all -> not water
    return Grid(x0, y0, cell_km, water)


@functools.lru_cache(maxsize=1)
def _land_with_margin():
    return prep(tools._land_km().buffer(SAFETY_MARGIN_KM))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def connected_sea(mask: np.ndarray) -> np.ndarray:
    """Only the largest connected water area (the open sea). Pockets cut off by the coarse
    grid (a narrow fjord end, a lake) are dropped, so a click there snaps to water that a
    route can actually leave. Diagonal steps need both side squares, so "connected"
    means connected through the 4 sides."""
    rows, cols = mask.shape
    label = np.zeros(mask.shape, dtype=np.int32)
    sizes = [0]
    for start in zip(*np.nonzero(mask)):
        if label[start]:
            continue
        current = len(sizes)
        label[start] = current
        stack, size = [start], 0
        while stack:
            r, c = stack.pop()
            size += 1
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not label[nr, nc]:
                    label[nr, nc] = current
                    stack.append((nr, nc))
        sizes.append(size)
    if len(sizes) == 1:
        return mask
    return label == int(np.argmax(sizes))


def usable_mask(grid: Grid, terms: list[CostTerm]) -> np.ndarray:
    """Water squares of the open sea that no term blocks."""
    mask = grid.water.copy()
    if terms:
        for row, col in zip(*np.nonzero(mask)):
            if any(t.blocked((row, col), grid) for t in terms):
                mask[row, col] = False
    return connected_sea(mask)


def snap(grid: Grid, mask: np.ndarray, lat: float, lon: float) -> tuple[tuple[int, int], float]:
    """Nearest usable water square to (lat, lon) -> (cell, distance in km)."""
    lon0, lat0, lon1, lat1 = ROUTE_BBOX
    if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
        raise RouteError(f"Outside the routing area ({lat0}–{lat1}° N, {lon0}–{lon1}° E).")
    x, y = tools._to_km(lon, lat)
    rows, cols = np.nonzero(mask)
    cx = grid.x0 + (cols + 0.5) * grid.cell_km
    cy = grid.y0 + (rows + 0.5) * grid.cell_km
    dist = np.hypot(cx - x, cy - y)
    best = int(np.argmin(dist))
    if dist[best] > MAX_SNAP_KM:
        raise RouteError(f"No water within {MAX_SNAP_KM:g} km of this point.")
    return (int(rows[best]), int(cols[best])), float(dist[best])


def astar(grid: Grid, mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
          terms: list[CostTerm], ctx: RouteContext) -> tuple[list[tuple[int, int]], int]:
    """Cheapest path of squares from start to goal -> (cells, squares expanded)."""
    rows, cols = mask.shape
    gx, gy = grid.center(goal)

    def h(cell):                                     # straight line: never more than the true cost
        x, y = grid.center(cell)
        return math.hypot(x - gx, y - gy)

    best = {start: 0.0}
    came_from = {}
    queue = [(h(start), 0.0, start)]
    expanded = 0
    while queue:
        _, g, cell = heapq.heappop(queue)
        if cell == goal:
            path = [cell]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return path[::-1], expanded
        if g > best.get(cell, math.inf):
            continue                                 # an outdated queue entry
        expanded += 1
        r, c = cell
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not mask[nr, nc]:
                continue
            if dr and dc and not (mask[r + dr, c] and mask[r, c + dc]):
                continue                             # no corner cutting past land
            step_km = grid.cell_km * (math.sqrt(2) if dr and dc else 1.0)
            cost = g + step_km + sum(t.extra(cell, (nr, nc), step_km, ctx) for t in terms)
            if cost < best.get((nr, nc), math.inf):
                best[(nr, nc)] = cost
                came_from[(nr, nc)] = cell
                heapq.heappush(queue, (cost + h((nr, nc)), cost, (nr, nc)))
    raise RouteError("No water route between these points.")


def smooth(points: list[tuple[float, float]], grid: Grid, terms: list[CostTerm]) -> list[tuple[float, float]]:
    """String pulling: from each kept point jump to the farthest later point whose straight
    segment keeps SAFETY_MARGIN_KM from land (and that every term allows)."""
    if len(points) <= 2:
        return points
    land = _land_with_margin()

    def ok(p, q):
        return (not land.intersects(LineString([p, q]))
                and all(t.segment_ok(p, q, grid) for t in terms))

    kept, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not ok(points[i], points[j]):
            j -= 1
        kept.append(points[j])                       # j = i + 1: a grid step, water by construction
        i = j
    return kept


def eta_text(hours: float) -> str:
    total = round(hours * 60)
    h, m = divmod(total, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def _lat_lon(p_km) -> list[float]:
    lon, lat = to_lonlat(*p_km)
    return [round(lat, 5), round(lon, 5)]


def route(from_lat: float, from_lon: float, to_lat: float, to_lon: float,
          speed_kn: float = DEFAULT_SPEED_KN, cell_km: float = CELL_KM,
          terms: list[CostTerm] | None = None, departure: datetime | None = None) -> dict:
    """The shortest water route between two points. Raises RouteError with a clear message."""
    if not speed_kn > 0:
        raise RouteError("Speed must be more than 0 knots.")
    terms = COST_TERMS if terms is None else terms
    grid = build_grid(cell_km)
    mask = usable_mask(grid, terms)
    start, start_snap = snap(grid, mask, from_lat, from_lon)
    goal, goal_snap = snap(grid, mask, to_lat, to_lon)
    ctx = RouteContext(speed_kn=speed_kn, departure=departure)

    if start == goal:
        cells, expanded = [start], 0
    else:
        cells, expanded = astar(grid, mask, start, goal, terms, ctx)
    # Smoothing is greedy, so it depends on the direction: straighten both ways and keep
    # the shorter line (also makes A->B and B->A give the same distance).
    centers = [grid.center(c) for c in cells]
    forward = smooth(centers, grid, terms)
    backward = smooth(centers[::-1], grid, terms)[::-1]
    points = min(forward, backward, key=lambda pts: sum(math.dist(p, q) for p, q in zip(pts, pts[1:])))
    distance_km = sum(math.dist(p, q) for p, q in zip(points, points[1:]))
    distance_nm = distance_km / KM_PER_NM
    eta_hours = distance_nm / speed_kn
    return {
        "path": [_lat_lon(p) for p in points],
        "distance_km": round(distance_km, 2),
        "distance_nm": round(distance_nm, 2),
        "speed_kn": speed_kn,
        "eta_hours": round(eta_hours, 3),
        "eta_text": eta_text(eta_hours),
        "start": {"clicked": [from_lat, from_lon], "snapped": _lat_lon(grid.center(start)),
                  "snap_km": round(start_snap, 2)},
        "end": {"clicked": [to_lat, to_lon], "snapped": _lat_lon(grid.center(goal)),
                "snap_km": round(goal_snap, 2)},
        "cells_expanded": expanded,
        "cell_km": cell_km,
        "cost_terms": [t.name for t in terms],
        "note": "Phase 1: shortest water route on a 1:10m coastline. No depth, traffic or weather yet.",
    }
