"""Passage-planning rules: load the rulebook (rules/passage_rules.yaml) and check a route.

Code decides and checks; the LLM only explains the report. Not for navigation.
"""
import functools
from pathlib import Path

import yaml

import tools

RULES_DIR = Path(__file__).resolve().parent / "rules"
RULES_FILE = "passage_rules.yaml"
RULE_TYPES = ("hard", "penalty", "check", "advisory", "not_modelled")
STATUSES = ("pass", "warn", "fail", "not_modelled")
REQUIRED_FIELDS = ("id", "factor", "source", "type")


class RulebookError(ValueError):
    """The rulebook file is missing or malformed."""


@functools.lru_cache(maxsize=1)
def load_rules() -> tuple[dict, ...]:
    """The rules, validated: required fields, known types, unique ids, a reason for
    not_modelled. Read from a fixed path inside rules/ (read-only configuration)."""
    path = tools._safe_path(RULES_DIR, RULES_FILE)
    if not path.exists():
        raise RulebookError(f"Rulebook missing: {path}")
    rules = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("rules") or []
    seen = set()
    for r in rules:
        missing = [f for f in REQUIRED_FIELDS if not r.get(f)]
        if missing:
            raise RulebookError(f"Rule {r.get('id', '?')} lacks {missing}")
        if r["type"] not in RULE_TYPES:
            raise RulebookError(f"Rule {r['id']}: unknown type {r['type']!r}")
        if r["type"] == "not_modelled" and not r.get("reason"):
            raise RulebookError(f"Rule {r['id']}: not_modelled needs a reason")
        if r["id"] in seen:
            raise RulebookError(f"Duplicate rule id {r['id']}")
        seen.add(r["id"])
    return tuple(rules)


def rule(rule_id: str) -> dict:
    return next(r for r in load_rules() if r["id"] == rule_id)


# ---------------------------------------------------------------------------
# Checking a route against the rulebook
# ---------------------------------------------------------------------------
import json
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, shape

import route_weather as rw
import routing

KM_PER_NM = routing.KM_PER_NM
# Harbours in routing.PLACES (the other places there are sea areas): ports of refuge for R-CONT-01.
HARBOURS = ["Kiel", "Laboe", "Eckernförde", "Damp", "Schleimünde", "Marstal", "Ærøskøbing", "Bagenkop",
            "Rudkøbing", "Spodsbjerg", "Heiligenhafen", "Großenbrode", "Orth (Fehmarn)", "Burgtiefe (Fehmarn)",
            "Puttgarden", "Rødby", "Nysted", "Gedser", "Kühlungsborn", "Warnemünde", "Klintholm (Møn)", "Stege",
            "Rødvig", "Vordingborg", "Femø", "Omø", "Karrebæksminde", "Bandholm", "Stubbekøbing"]
DEFAULT_GUST_LIMIT_BFT = 8
DEFAULT_AIR_DRAUGHT_M = 14.0          # small sailboat mast (the default vessel)
STATUS_ORDER = {"fail": 0, "warn": 1, "pass": 2, "not_modelled": 3}
BRIDGES_FILE = "bridges_kiel_bay.geojson"     # bridge lines from OSM (download_routing_data.py --bridges-only)
APPROXIMATE = "approximate - verify on chart"


@functools.lru_cache(maxsize=1)
def bridge_lines() -> dict:
    """{bridge name: (line in km, clearance from an OSM seamark tag or None)} from the OSM
    bridge file; {} if the file is missing (R-AIR-01 then uses the rulebook points)."""
    path = tools._safe_path(tools.DATA_DIR, BRIDGES_FILE)
    if not path.exists():
        return {}
    out = {}
    for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
        geom = shapely.transform(shape(f["geometry"]), lambda xy: xy * [tools._KM_PER_DEG_LON, tools._KM_PER_DEG_LAT])
        out[f["properties"]["name"]] = (geom, f["properties"].get("clearance_m_osm"))
    return out


def bridges_passed(line, p: dict) -> list[dict]:
    """The rulebook bridges the route passes under, each with its clearance and where the
    clearance comes from ("osm", "approximate" or "unknown")."""
    lines, passed = bridge_lines(), []
    for b in p["bridges"]:
        found = lines.get(b["name"])
        if found:
            geom, osm_clearance = found
            if line.distance(geom) > p["pass_within_km"]:
                continue
        elif line.distance(Point(tools._to_km(b["lon"], b["lat"]))) <= p["pass_within_km_point"]:
            osm_clearance = None
        else:
            continue
        clearance = osm_clearance if osm_clearance is not None else b["clearance_m"]
        passed.append({"name": b["name"], "clearance_m": clearance,
                       "source": "osm" if osm_clearance is not None else "approximate" if clearance is not None
                       else "unknown"})
    return passed


def _row(r: dict, status: str, value=None, limit=None, detail: str | None = None) -> dict:
    return {"id": r["id"], "factor": r["factor"], "type": r["type"], "status": status,
            "value": value, "limit": limit, "source": r["source"],
            "detail": detail if detail is not None else r.get("reason")}


def _points_km(result: dict) -> list[tuple[float, float]]:
    return [tools._to_km(lon, lat) for lat, lon in result["path"]]


def _samples(points):
    x, y = routing._sample(points)
    in_passage = routing.passage_raster().at(x, y) > 0
    return x, y, in_passage


def harbour_distances(lat: float, lon: float) -> tuple[str, float]:
    """Nearest harbour (name, km, straight line) to a point."""
    x, y = tools._to_km(lon, lat)
    best = min(((math.dist((x, y), tools._to_km(routing.PLACES[h][1], routing.PLACES[h][0])), h) for h in HARBOURS))
    return best[1], best[0]


def check_route(result: dict, air_draught_m: float = DEFAULT_AIR_DRAUGHT_M,
                gust_limit_bft: int = DEFAULT_GUST_LIMIT_BFT, block_coefficient: float | None = None,
                dwt: float | None = None, weather_error: str | None = None) -> dict:
    """Evaluate a route (routing.route() with the phase 2 rules) against every rule.
    Returns {"rows": [...], "counts": {...}, "blocking": [...ids that fail]}. Values come from
    the route result and the data; limits from the route settings (no duplicate settings)."""
    s = result["settings"]
    draft, margin = s["draft_m"], s["margin_m"]
    speed = result["speed_kn"]
    rules_obj = routing.make_rules(draft, margin, s["zones"], s["land_buffer_m"], s["zone_buffer_m"],
                                   s["passages"], s["passage_land_buffer_m"], s["passage_zone_buffer_m"],
                                   s["penalty_factor"], s.get("wind_farms", "block"),
                                   s.get("turbine_radius_m", routing.TURBINE_RADIUS_M), s.get("squat_m") or 0)
    pts = _points_km(result)
    x, y, in_p = _samples(pts)
    in_p &= bool(s["passages"])          # passages off: the full buffers apply everywhere
    depth = routing.load_depth().at(x, y)
    on = routing.zones_on(pts, rules_obj.turbine_radius_km)
    rows = []
    for r in load_rules():
        rid, p = r["id"], r.get("parameters") or {}
        if r["type"] == "not_modelled":
            rows.append(_row(r, "not_modelled"))
        elif rid == "R-UKC-01":
            ukc = round(result["min_depth_m"] - draft, 1)
            squat_m = s.get("squat_m") or 0
            need = f"draft {draft:g} m" + (f" + squat {squat_m:g} m" if squat_m else "") + f" + {margin:g} m"
            rows.append(_row(r, "fail" if ukc < margin + squat_m else "pass",
                             f"min depth {result['min_depth_m']} m = UKC {ukc} m",
                             f"UKC ≥ {margin + squat_m:g} m ({need} = {result['required_depth_m']:g} m)"))
        elif rid == "R-UKC-02":
            cb = block_coefficient if block_coefficient is not None else p["block_coefficient_default"]
            squat = cb * speed ** 2 / 100
            left = depth - draft - squat * np.where(in_p, 2.0, 1.0)
            worst = float(left.min())
            status = "fail" if worst < 0 else "warn" if worst < margin else "pass"
            rows.append(_row(r, status, f"UKC after squat {worst:.2f} m (squat {squat:.2f} m at {speed:g} kn, Cb {cb:g}"
                                        + (", x2 in narrow passages" if in_p.any() else "") + ")",
                             f"≥ {margin:g} m (UKC policy)"))
        elif rid == "R-UKC-03":
            spare = result["min_depth_m"] - result["required_depth_m"]
            rows.append(_row(r, "warn" if spare < p["warn_if_spare_below_m"] else "pass",
                             f"{spare:.1f} m more than required", f"warn below {p['warn_if_spare_below_m']:g} m "
                             "(depth model ~115 m grid)"))
        elif rid in ("R-ZONE-01", "R-ZONE-02", "R-WIND-01"):
            cats = p["categories"]
            names = sorted({n for c in cats for n in on.get(c, [])})
            mode = "block" if rid == "R-ZONE-01" else s["zones"] if rid == "R-ZONE-02" else s.get("wind_farms", "block")
            if rid == "R-WIND-01":
                turb = result.get("min_turbine_distance_km")
                value = ((f"crosses {', '.join(names)}; " if names else "")
                         + (f"nearest turbine {turb * 1000:.0f} m" if turb is not None else "no turbine nearby"))
                limit = f"{mode}; turbine radius {s.get('turbine_radius_m', routing.TURBINE_RADIUS_M):g} m"
            else:
                km_in = routing.length_inside(pts, tuple(cats), rules_obj.turbine_radius_km)
                value = f"crosses {', '.join(names)} ({km_in:.2f} km inside)" if names else "none crossed"
                limit = "closed (always)" if rid == "R-ZONE-01" else f"{mode} (penalty factor {s['penalty_factor']:g})"
            status = "pass" if not names else "fail" if mode == "block" else "warn"
            rows.append(_row(r, status, value, limit))
        elif rid == "R-ZONE-03":
            names = on.get("exercise", [])
            rows.append(_row(r, "warn" if names else "pass", f"passes {', '.join(names)}" if names else "none passed",
                             "passage allowed; check firing notices"))
        elif rid == "R-BUF-01":
            d = routing.land_distance().at(x, y)
            open_min = float(d[~in_p].min()) if (~in_p).any() else None
            buf = s["land_buffer_m"] / 1000
            rows.append(_row(r, "pass" if open_min is None or open_min >= buf else "fail",
                             None if open_min is None else f"{open_min * 1000:.0f} m from land (outside narrow passages)",
                             f"≥ {s['land_buffer_m']:g} m"))
        elif rid == "R-BUF-02":
            if not rules_obj.block:
                rows.append(_row(r, "pass", "no closed zones", f"≥ {s['zone_buffer_m']:g} m"))
                continue
            d = routing.zone_distance(rules_obj.block, False).at(x, y)
            open_min = float(d[~in_p].min()) if (~in_p).any() else None
            buf = s["zone_buffer_m"] / 1000
            rows.append(_row(r, "pass" if open_min is None or open_min >= buf else "fail",
                             None if open_min is None else f"{open_min * 1000:.0f} m from closed zones "
                                                           f"(outside narrow passages; nearest {result['nearest_zone']})",
                             f"≥ {s['zone_buffer_m']:g} m"))
        elif rid == "R-COLREG-09":
            used = result.get("passages_used") or []
            via = [v for v in result.get("via") or [] if v in p["passages"]]
            if used:
                d = routing.land_distance().at(x, y)[in_p]
                rows.append(_row(r, "warn", f"reduced clearance in {', '.join(used)} (min {float(d.min()) * 1000:.0f} m "
                                            "from land)", f"{s['passage_land_buffer_m']:g} m land / "
                                                          f"{s['passage_zone_buffer_m']:g} m zones; keep to the "
                                                          "starboard side of the fairway"))
            else:
                rows.append(_row(r, "pass", f"passes {', '.join(via)} with full clearance" if via
                                 else "no narrow passage", "open-water buffers kept"))
        elif rid == "R-AIR-01":
            line = LineString(pts) if len(pts) > 1 else Point(pts[0])
            passed = bridges_passed(line, p)
            limit = f"clearance ≥ air draught + {p['warn_margin_m']:g} m"
            if not passed:
                rows.append(_row(r, "pass", "no fixed bridge on the route", f"air draught {air_draught_m:g} m"))
                continue
            known = [b for b in passed if b["clearance_m"] is not None]
            unknown = [b["name"] for b in passed if b["clearance_m"] is None]
            status, parts = "pass", []
            if known:
                low = min(known, key=lambda b: b["clearance_m"])
                spare = low["clearance_m"] - air_draught_m
                status = "fail" if spare < 0 else "warn" if spare < p["warn_margin_m"] else "pass"
                tag = "OSM seamark tag" if low["source"] == "osm" else APPROXIMATE
                parts.append(f"air draught {air_draught_m:g} m under {low['name']} "
                             f"(clearance {low['clearance_m']:g} m, {tag})")
            if unknown:
                parts.append(f"passes under {', '.join(unknown)}: clearance unknown - verify on chart")
                status = "warn" if status == "pass" else status
            rows.append(_row(r, status, "; ".join(parts), limit))
        elif rid.startswith("R-WX-"):
            w = result.get("weather")
            if weather_error:                       # no weather at all (departure outside the forecast)
                rows.append(_row(r, "warn", weather_error, "whole passage inside the forecast") if rid == "R-WX-04"
                            else _row(r, "not_modelled", detail="no weather: the passage is outside the forecast"))
                continue
            if not w:
                rows.append(_row(r, "not_modelled", detail="weather along the route is off"))
                continue
            beyond = w.get("no_forecast_legs") or []
            if rid == "R-WX-04":
                if beyond:
                    total = len(w["legs"]) + len(beyond)
                    rows.append(_row(r, "warn", f"no forecast for {len(beyond)} of {total} legs (legs "
                                                f"{beyond[0]['leg']}–{beyond[-1]['leg']}, from "
                                                f"{beyond[0]['time_utc'][11:16]} UTC); the forecast ends "
                                                f"{w['forecast_end_utc'][:10]} {w['forecast_end_utc'][11:16]} UTC",
                                     "whole passage inside the forecast"))
                else:
                    rows.append(_row(r, "pass", f"forecast covers {w['departure_utc'][11:16]}–"
                                                f"{w['legs'][-1]['time_utc'][11:16]} UTC",
                                     "whole passage inside the forecast"))
                continue
            if not w.get("legs"):
                rows.append(_row(r, "not_modelled", detail="no leg inside the forecast (see R-WX-04)"))
                continue
            legs = w["legs"]
            if rid == "R-WX-01":
                lim, caution = w["limits"]["wind_limit_bft"], p["caution_from_bft"]
                top = max(legs, key=lambda g: (g["wind_bft"], g["wind_ms"]))
                status = "fail" if top["wind_bft"] > lim else "warn" if top["wind_bft"] >= caution else "pass"
                rows.append(_row(r, status, f"max Beaufort {top['wind_bft']} at {top['time_utc'][11:16]} UTC (leg {top['leg']})",
                                 f"≤ Beaufort {lim} (caution from {caution})"))
            elif rid == "R-WX-02":
                gusts = [g for g in legs if g.get("gust_bft") is not None]
                if not gusts:
                    rows.append(_row(r, "not_modelled", detail="no gust data (dummy data: wind only)"))
                    continue
                top = max(gusts, key=lambda g: (g["gust_bft"], g["gust_ms"]))
                status = "fail" if top["gust_bft"] > gust_limit_bft else "warn" if top["gust_bft"] == gust_limit_bft else "pass"
                rows.append(_row(r, status, f"max gusts Beaufort {top['gust_bft']} at {top['time_utc'][11:16]} UTC "
                                            f"(leg {top['leg']})", f"≤ Beaufort {gust_limit_bft}"))
            elif rid == "R-WX-03":
                waves = [g for g in legs if g.get("waves_m") is not None]
                if not waves:
                    rows.append(_row(r, "not_modelled", detail="no wave data (dummy data: wind only)"))
                    continue
                lim, caution = w["limits"]["waves_limit_m"], p["caution_from_m"]
                top = max(waves, key=lambda g: g["waves_m"])
                status = "fail" if top["waves_m"] > lim else "warn" if top["waves_m"] >= caution else "pass"
                rows.append(_row(r, status, f"max waves {top['waves_m']} m at {top['time_utc'][11:16]} UTC (leg {top['leg']})",
                                 f"≤ {lim:g} m (caution from {caution:g} m)"))
        elif rid == "R-PILOT-01":
            rows.append(_row(r, "warn" if draft >= p["draft_from_m"] else "pass", f"draft {draft:g} m",
                             f"pilot recommended from {p['draft_from_m']:g} m"))
        elif rid == "R-SIZE-01":
            over = draft > p["max_draft_m"] or air_draught_m > p["max_air_draught_m"]
            rows.append(_row(r, "warn" if over else "pass", f"draft {draft:g} m, air draught {air_draught_m:g} m",
                             f"≤ {p['max_draft_m']:g} m draft, ≤ {p['max_air_draught_m']:g} m air draught"))
        elif rid == "R-ROUTE-T":
            big = dwt is not None and dwt >= p["dwt_from"]
            rows.append(_row(r, "warn" if big else "pass",
                             f"{dwt:,.0f} dwt" if dwt is not None else "dwt not given (vessel profile)",
                             f"Route T from {p['dwt_from']:,} dwt (track not modelled)"))
        elif rid == "R-CONT-01":
            w = result.get("weather") or {}
            legs = (w.get("legs") or []) + (w.get("no_forecast_legs") or []) or [
                {"leg": 1, "lat": result["path"][len(result["path"]) // 2][0], "lon": result["path"][len(result["path"]) // 2][1]}]
            far = max(((harbour_distances(g["lat"], g["lon"]), g) for g in legs), key=lambda t: t[0][1])
            (name, km_), g = far
            nm = km_ / KM_PER_NM
            rows.append(_row(r, "warn" if nm > p["warn_if_farther_than_nm"] else "pass",
                             f"farthest from a harbour: leg {g['leg']}, {nm:.1f} nm to {name}",
                             f"≤ {p['warn_if_farther_than_nm']:g} nm"))
        elif rid == "R-AREA-01":
            lon0, lat0, lon1, lat1 = routing.ROUTE_BBOX
            lats = np.array([q[0] for q in result["path"]])
            lons = np.array([q[1] for q in result["path"]])
            edge_km = float(min((lats - lat0).min() * tools._KM_PER_DEG_LAT, (lat1 - lats).min() * tools._KM_PER_DEG_LAT,
                                (lons - lon0).min() * tools._KM_PER_DEG_LON, (lon1 - lons).min() * tools._KM_PER_DEG_LON))
            rows.append(_row(r, "warn" if edge_km < p["warn_within_km"] else "pass",
                             f"{edge_km:.1f} km from the routing-area edge",
                             f"≥ {p['warn_within_km']:g} km (a better route may leave the area)"))
        else:
            rows.append(_row(r, "not_modelled", detail="no check implemented"))
    rows.sort(key=lambda row: STATUS_ORDER[row["status"]])
    counts = {st: sum(row["status"] == st for row in rows) for st in STATUSES}
    return {"rows": rows, "counts": counts,
            "blocking": [row["id"] for row in rows if row["status"] == "fail"],
            "warnings": [row["id"] for row in rows if row["status"] == "warn"],
            "note": "Rules from docs/Ship_Voyage_Planning.pdf, checked in code. " + routing.NOT_FOR_NAVIGATION}


# ---------------------------------------------------------------------------
# Vessel profiles (rules/vessel_profiles.yaml)
# ---------------------------------------------------------------------------
PROFILES_FILE = "vessel_profiles.yaml"
PROFILE_FIELDS = ("label", "draft_m", "air_draught_m", "speed_kn", "wind_limit_bft", "gust_limit_bft",
                  "waves_limit_m", "ukc_policy", "block_coefficient")


@functools.lru_cache(maxsize=1)
def load_profiles() -> dict:
    """{"profiles": {name: {...}}, "default": name}, validated."""
    path = tools._safe_path(RULES_DIR, PROFILES_FILE)
    if not path.exists():
        raise RulebookError(f"Vessel profiles missing: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles") or {}
    for name, p in profiles.items():
        missing = [f for f in PROFILE_FIELDS if p.get(f) is None]
        if missing:
            raise RulebookError(f"Vessel profile {name} lacks {missing}")
        if p["ukc_policy"].get("type") not in ("fixed", "percent_of_draft"):
            raise RulebookError(f"Vessel profile {name}: unknown UKC policy {p['ukc_policy']}")
    if data.get("default") not in profiles:
        raise RulebookError("vessel_profiles.yaml: 'default' must name a profile")
    return {"profiles": profiles, "default": data["default"]}


def ukc_margin(policy: dict, draft_m: float) -> float:
    """The UKC margin in metres from a profile's policy (fixed, or % of the draft with a minimum)."""
    if policy["type"] == "fixed":
        return float(policy["value_m"])
    return round(max(float(policy["min_m"]), draft_m * float(policy["percent"]) / 100), 1)


def profile_settings(name: str) -> dict:
    """The route settings a profile fills in the Route tab (the user can still change them),
    plus the values only the rule checker uses (block coefficient, dwt)."""
    profiles = load_profiles()["profiles"]
    if name not in profiles:
        raise KeyError(f"Unknown vessel profile {name!r}; known: {sorted(profiles)}")
    p = profiles[name]
    return {"profile": name, "label": p["label"], "draft_m": p["draft_m"],
            "margin_m": ukc_margin(p["ukc_policy"], p["draft_m"]), "air_draught_m": p["air_draught_m"],
            "speed_kn": p["speed_kn"], "wind_limit_bft": p["wind_limit_bft"], "gust_limit_bft": p["gust_limit_bft"],
            "waves_limit_m": p["waves_limit_m"], "block_coefficient": p["block_coefficient"], "dwt": p.get("dwt"),
            "ukc_policy": p["ukc_policy"], "squat_in_router": bool(p.get("squat_in_router"))}
