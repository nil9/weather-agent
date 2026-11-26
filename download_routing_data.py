"""Download the data for sea routing phase 2 ONCE and cache it in data/.

Writes:
  data/depth_kiel_bay.npz                 EMODnet Bathymetry DTM for the routing box
  data/restricted_zones_kiel_bay.geojson  restricted / military zones from OpenStreetMap

Sources (only the routing box and the tag names are sent, no personal data):
  - Depth: EMODnet Bathymetry, coverage emodnet__mean, via its WCS as plain text
    (no GDAL needed). Licence CC BY 4.0. Values are elevation in metres: negative =
    water depth, positive = land. Resolution 1/960 degree (~115 m N-S, ~67 m E-W).
    Fetched in tiles and stored as int16 decimetres.
  - Zones: OpenStreetMap / OpenSeaMap tags (seamark:type=restricted_area,
    military=range/danger_area) via the Overpass API. Licence ODbL,
    "(c) OpenStreetMap contributors". Each zone gets a rule category (from its tags):
      no_entry    - entry prohibited (military Sperrgebiete, nature reserves, aquaculture, ...)
      firing      - firing ranges / danger areas (Todendorf, Putlos, Torpedoschiessbahn, ...);
                    OSM: "passage allowed outside exercise times"
      restricted  - other military areas with restricted entry or "to be avoided"
      exercise    - military exercise areas WITHOUT an entry restriction (e.g. Walkyrien,
                    600 km2 of the outer bay): passage allowed, kept only for the map
    Zones that only restrict fishing, anchoring or similar (non-military) are not kept.

Usage:  python download_routing_data.py [--depth-only | --zones-only]
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import polygonize, unary_union

import routing

DATA = Path(__file__).resolve().parent / "data"
DEPTH_FILE = DATA / "depth_kiel_bay.npz"
ZONES_FILE = DATA / "restricted_zones_kiel_bay.geojson"
USER_AGENT = "marine-weather-agent (learning project)"

WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE = "emodnet__mean"
PIXELS_PER_DEGREE = 960                  # EMODnet DTM: 1/16 arc minute
TILE_LON, TILE_LAT = 0.3, 0.25           # degrees per request (~70k values)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _get(url: str, params: dict | None = None, data: bytes | None = None, timeout: int = 120) -> bytes:
    full = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    request = urllib.request.Request(full, data=data, headers={"User-Agent": USER_AGENT,
                                                               "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# ---------------------------------------------------------------------------
# Depth (EMODnet WCS, plain-text grid)
# ---------------------------------------------------------------------------

def parse_wcs_text(text: str) -> tuple[np.ndarray, float, float]:
    """EMODnet WCS text grid -> (values with row 0 = north, lon of the first pixel centre,
    lat of the first pixel centre)."""
    head, body = text.split("Band 0:", 1)
    x0 = float(re.search(r'"elt_0_2",\s*([-\d.eE]+)', head).group(1))
    y0 = float(re.search(r'"elt_1_2",\s*([-\d.eE]+)', head).group(1))
    rows = [line.split() for line in body.strip().splitlines() if line.strip()]
    return np.array(rows, dtype=np.float32), x0, y0


def download_depth():
    lon0, lat0, lon1, lat1 = routing.ROUTE_BBOX
    cols = round((lon1 - lon0) * PIXELS_PER_DEGREE)
    rows = round((lat1 - lat0) * PIXELS_PER_DEGREE)
    grid = np.full((rows, cols), np.nan, dtype=np.float32)        # row 0 = north
    tiles = [(x, y) for y in np.arange(lat0, lat1, TILE_LAT) for x in np.arange(lon0, lon1, TILE_LON)]
    for n, (x, y) in enumerate(tiles, 1):
        params = {"service": "WCS", "version": "2.0.1", "request": "GetCoverage", "coverageId": COVERAGE,
                  "format": "text/plain",
                  "subset": [f"Long({x:.4f},{min(x + TILE_LON, lon1):.4f})",
                             f"Lat({y:.4f},{min(y + TILE_LAT, lat1):.4f})"]}
        query = urllib.parse.urlencode(params, doseq=True)
        for attempt in range(3):
            try:
                text = _get(f"{WCS_URL}?{query}").decode("utf-8")
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(5)
        values, cx, cy = parse_wcs_text(text)
        # Place the tile by its pixel centres in the whole grid (overlaps simply overwrite).
        col = round((cx - lon0) * PIXELS_PER_DEGREE - 0.5)
        row = round((lat1 - cy) * PIXELS_PER_DEGREE - 0.5)
        r0, c0 = max(row, 0), max(col, 0)
        r1, c1 = min(row + values.shape[0], rows), min(col + values.shape[1], cols)
        grid[r0:r1, c0:c1] = values[r0 - row:r1 - row, c0 - col:c1 - col]
        print(f"  depth tile {n}/{len(tiles)}: {values.shape[1]} x {values.shape[0]} values")
    missing = int(np.isnan(grid).sum())
    if missing:
        raise SystemExit(f"{missing} depth values are missing after merging the tiles.")
    np.savez_compressed(
        DEPTH_FILE,
        elev_dm=np.round(grid * 10).astype(np.int16),            # elevation in decimetres
        lon0=lon0, lat1=lat1, pixels_per_degree=PIXELS_PER_DEGREE,
        source=f"EMODnet Bathymetry DTM ({COVERAGE}) via {WCS_URL}",
        licence="CC BY 4.0, EMODnet Bathymetry Consortium",
        fetched=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    water = grid[grid < 0]
    print(f"Wrote {DEPTH_FILE.name}: {cols} x {rows} pixels, deepest {-water.min():.1f} m, "
          f"{DEPTH_FILE.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Restricted zones (OpenStreetMap via Overpass)
# ---------------------------------------------------------------------------

FIRING_WORDS = ("schieß", "schiess", "gefahrenbereich", "firing")


def zone_category(tags: dict) -> str | None:
    """no_entry / firing / restricted / exercise, or None if it doesn't matter for routing."""
    category = tags.get("seamark:restricted_area:category", "")
    restriction = tags.get("seamark:restricted_area:restriction", "")
    name = f"{tags.get('name', '')} {tags.get('seamark:name', '')}".lower()
    military = tags.get("military") in ("range", "danger_area") or "military" in category
    if any(word in restriction or word in category for word in ("no_entry", "entry_prohibited")):
        return "no_entry"
    if tags.get("military") in ("range", "danger_area") or (military and any(w in name for w in FIRING_WORDS)):
        return "firing"
    if military and any(word in restriction for word in ("restricted_entry", "to_be_avoided")):
        return "restricted"
    if military:
        return "exercise"
    return None


def element_polygon(element: dict):
    """A closed way or a multipolygon relation -> shapely polygon (or None)."""
    if element["type"] == "way":
        points = [(p["lon"], p["lat"]) for p in element.get("geometry", [])]
        if len(points) >= 4 and points[0] == points[-1]:
            return Polygon(points).buffer(0)
        return None
    outers = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
              for m in element.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
    polygons = list(polygonize(unary_union(outers))) if outers else []
    return unary_union(polygons).buffer(0) if polygons else None


def download_zones():
    lon0, lat0, lon1, lat1 = routing.ROUTE_BBOX
    box = f"{lat0},{lon0},{lat1},{lon1}"
    query = f"""[out:json][timeout:90];
(way["seamark:type"="restricted_area"]({box});relation["seamark:type"="restricted_area"]({box});
 way["military"~"^(danger_area|range)$"]({box});relation["military"~"^(danger_area|range)$"]({box}););
out geom;"""
    for attempt in range(4):                       # Overpass is often busy (429 / 502 / 504)
        try:
            answer = json.loads(_get(OVERPASS_URL, data=urllib.parse.urlencode({"data": query}).encode()))
            break
        except OSError as exc:
            if attempt == 3:
                raise
            print(f"  Overpass busy ({exc}), retrying in {15 * (attempt + 1)} s")
            time.sleep(15 * (attempt + 1))
    features, counts, skipped = [], {}, 0
    for element in answer["elements"]:
        tags = element.get("tags", {})
        category = zone_category(tags)
        polygon = element_polygon(element) if category else None
        if polygon is None or polygon.is_empty:
            skipped += 1
            continue
        counts[category] = counts.get(category, 0) + 1
        features.append({"type": "Feature", "geometry": mapping(polygon), "properties": {
            "name": tags.get("name") or tags.get("seamark:name") or f"{category} area",
            "category": category,
            "restriction": tags.get("seamark:restricted_area:restriction"),
            "osm": f"{element['type']}/{element['id']}"}})
    ZONES_FILE.write_text(json.dumps({
        "type": "FeatureCollection",
        "properties": {"source": f"OpenStreetMap via {OVERPASS_URL}",
                       "licence": "ODbL, (c) OpenStreetMap contributors",
                       "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "bbox": routing.ROUTE_BBOX},
        "features": features}, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {ZONES_FILE.name}: {counts} (ignored {skipped} fishing/anchoring-only or open areas)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--zones-only", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    if not args.zones_only:
        download_depth()
    if not args.depth_only:
        download_zones()


if __name__ == "__main__":
    main()
