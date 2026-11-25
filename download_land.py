"""Download Natural Earth land polygons ONCE and keep only the Kiel Bay area.

Writes:  data/land_kiel_bay.geojson   (used by tools.is_on_land for the land/sea check)

Source: Natural Earth 1:10m "land" + "minor islands" (public domain), pinned to a
release tag so the data never changes silently. The full files (~10 MB) are not kept;
only the clipped Kiel Bay part is saved.
"""
import json
import urllib.request
from pathlib import Path

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

TAG = "v5.1.2"
BASE_URL = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{TAG}/geojson"
LAYERS = ["ne_10m_land", "ne_10m_minor_islands"]
BBOX = (9.0, 53.8, 12.0, 55.6)        # lon_min, lat_min, lon_max, lat_max (Kiel Bay + margin)
OUT = Path(__file__).resolve().parent / "data" / "land_kiel_bay.geojson"


def download(layer: str) -> dict:
    url = f"{BASE_URL}/{layer}.geojson"
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def main():
    area = box(*BBOX)
    pieces = []
    for layer in LAYERS:
        features = download(layer)["features"]
        clipped = [shape(f["geometry"]).intersection(area) for f in features
                   if shape(f["geometry"]).intersects(area)]
        clipped = [g for g in clipped if not g.is_empty]
        print(f"  {layer}: {len(clipped)} feature(s) inside the Kiel Bay box")
        pieces += clipped

    land = unary_union(pieces)
    polygons = list(land.geoms) if land.geom_type == "MultiPolygon" else [land]
    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "source": f"Natural Earth 1:10m {', '.join(LAYERS)} ({BASE_URL})",
            "license": "Public domain (naturalearthdata.com)",
            "bbox": BBOX,
        },
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(p)} for p in polygons],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(geojson), encoding="utf-8")
    print(f"Wrote {OUT} ({len(polygons)} polygons, {OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
