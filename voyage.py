"""Voyage planning: route candidates, a recommendation, and a passage plan (code only).

For two points, up to 3 candidate routes are computed and checked against the rulebook
(rule_checker.py). Code scores them and recommends one; the LLM only explains the result.
For analysis and learning only. Not for navigation.

Candidates (all with the user's settings, see CANDIDATES):
  shortest   the shortest route that keeps the rules (weather shown, not avoided)
  balanced   weather-aware: rough legs cost extra
  safest     weather-aware, full buffers everywhere (no narrow-passage exception), and firing
             areas / wind farms always avoided (even if the Route tab allows them with a penalty)
Score (lower is better): any failed rule -> not recommended; then fewer warnings, then lower
worst wind, then lower waves, then shorter.
"""
import math
import time
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

import routing
import rule_checker
import tools

CANDIDATES = [
    ("shortest", "Shortest", {"weather": "show"}),
    ("balanced", "Balanced (weather-aware)", {"weather": "aware"}),
    ("safest", "Safest (weather-aware, full buffers, all zones avoided)",
     {"weather": "aware", "passages": False, "zones": "block", "wind_farms": "block"}),
]
SAME_KM = 0.1                       # candidates this close in length, via the same places, are duplicates
RULE_KEYS = ("air_draught_m", "gust_limit_bft", "profile")     # settings only the rule checker uses


def route_with_weather(from_lat, from_lon, to_lat, to_lon, **settings) -> tuple[dict, str | None]:
    """routing.route(); if the departure is before the forecast starts, the route without
    weather and the reason (rule R-WX-04 then warns). Other errors are raised. (Legs AFTER the
    end of the forecast don't raise: they come as weather["no_forecast_legs"].)"""
    try:
        return routing.route(from_lat, from_lon, to_lat, to_lon, **settings), None
    except routing.RouteError as exc:
        if "outside the forecast" not in str(exc):
            raise
        return routing.route(from_lat, from_lon, to_lat, to_lon, **{**settings, "weather": "off"}), str(exc)


def vessel_args(profile: str | None) -> dict:
    """What the rule checker takes from a vessel profile (squat block coefficient, dwt)."""
    if not profile:
        return {}
    p = rule_checker.profile_settings(profile)
    return {"block_coefficient": p["block_coefficient"], "dwt": p["dwt"]}


def squat_m(profile: str | None, speed_kn: float) -> float:
    """The squat allowance the router adds to the required depth: Barrass Cb x V^2 / 100 with
    the profile's block coefficient and the route's speed (the profile fills the speed; the
    user may change it). 0 without a profile or for profiles without squat_in_router."""
    if not profile:
        return 0.0
    p = rule_checker.profile_settings(profile)
    return round(p["block_coefficient"] * speed_kn ** 2 / 100, 2) if p["squat_in_router"] else 0.0


def router_settings(settings: dict) -> dict:
    """The Route tab's settings as routing.route() takes them: without the rule-only settings
    (air draught, gust limit, profile), plus the profile's squat allowance (squat_m)."""
    out = {k: v for k, v in settings.items() if k not in RULE_KEYS}
    squat = squat_m(settings.get("profile"), float(settings.get("speed_kn") or routing.DEFAULT_SPEED_KN))
    if squat:
        out["squat_m"] = squat
    return out


def checked_route(from_lat, from_lon, to_lat, to_lon, settings: dict) -> dict:
    """One route with its rule check. settings: route settings + air_draught_m, gust_limit_bft,
    profile (the rule-only settings). A RouteError names the vessel in its short reason
    ("Too shallow for Large ship: 0.0 m here, needs 16.1 m")."""
    rule_args = {k: settings[k] for k in ("air_draught_m", "gust_limit_bft") if k in settings}
    try:
        result, weather_error = route_with_weather(from_lat, from_lon, to_lat, to_lon, **router_settings(settings))
    except routing.RouteError as exc:
        if exc.kind == "shallow" and exc.short and settings.get("profile"):
            label = rule_checker.profile_settings(settings["profile"])["label"].split(" (")[0]
            exc.short = exc.short.replace("Too shallow:", f"Too shallow for {label}:", 1)
        raise
    if weather_error:
        result["weather_error"] = weather_error
    result["rule_check"] = rule_checker.check_route(result, weather_error=weather_error, **rule_args,
                                                    **vessel_args(settings.get("profile")))
    return result


def summarize(kind: str, label: str, r: dict) -> dict:
    """What the side-by-side table shows for one candidate."""
    w = r.get("weather") or {}
    legs = w.get("legs") or []
    top = max(legs, key=lambda g: (g["wind_bft"], g["wind_ms"])) if legs else None
    waves = [g["waves_m"] for g in legs if g.get("waves_m") is not None]
    gusts = [g["gust_bft"] for g in legs if g.get("gust_bft") is not None]
    rc = r["rule_check"]
    return {
        "kind": kind, "label": label, "distance_km": r["distance_km"], "distance_nm": r["distance_nm"],
        "eta_text": r["eta_text"], "eta_hours": r["eta_hours"], "via": r.get("via", []),
        "worst_wind_bft": top["wind_bft"] if top else None,
        "worst_wind_at": f"{top['time_utc'][11:16]} UTC (leg {top['leg']})" if top else None,
        "worst_gust_bft": max(gusts) if gusts else None, "worst_waves_m": max(waves) if waves else None,
        "rough_legs": w.get("rough_count"), "no_forecast_legs": len(w.get("no_forecast_legs") or []),
        "min_ukc_m": round(r["min_depth_m"] - r["settings"]["draft_m"], 1),
        "fails": rc["blocking"], "warnings": rc["warnings"], "counts": rc["counts"],
        "weather_error": r.get("weather_error"), "path": r["path"],
    }


def score(c: dict) -> tuple:
    """Lower is better: no failed rule first, then fewer warnings, worst wind, waves, distance."""
    return (len(c["fails"]) > 0, len(c["warnings"]), c["worst_wind_bft"] if c["worst_wind_bft"] is not None else 99,
            c["worst_waves_m"] if c["worst_waves_m"] is not None else 99, c["distance_km"])


def _duplicate(c: dict, others: list[dict]) -> str | None:
    for o in others:
        if c["path"] == o["path"] or (abs(c["distance_km"] - o["distance_km"]) < SAME_KM and c["via"] == o["via"]):
            return o["kind"]
    return None


def recommend(cands: list[dict]) -> dict:
    """Rank the candidates in code and pick the recommended one (None if every one fails a rule)."""
    ranked = sorted(cands, key=score)
    best = ranked[0] if ranked else None
    ok = best is not None and not best["fails"]
    reasons = []
    if best is not None:
        for other in ranked[1:]:
            reasons.append(compare(best, other))
    return {"recommended": best["kind"] if ok else None, "least_bad": None if ok else (best["kind"] if best else None),
            "ranking": [c["kind"] for c in ranked], "reasons": reasons,
            "rule": "any failed rule = not recommended; then fewer warnings, lower worst wind, lower waves, shorter"}


def compare(best: dict, other: dict) -> str:
    """Why `best` ranks above `other`, from the score (first criterion that differs)."""
    a, b = score(best), score(other)
    name = f"{best['label']} over {other['label']}"
    if a[0] != b[0]:
        return f"{name}: {other['label']} fails {', '.join(other['fails'])}"
    if a[1] != b[1]:
        return f"{name}: {len(best['warnings'])} warnings vs {len(other['warnings'])}"
    if a[2] != b[2]:
        return f"{name}: worst wind Beaufort {best['worst_wind_bft']} vs Beaufort {other['worst_wind_bft']}"
    if a[3] != b[3]:
        return f"{name}: worst waves {best['worst_waves_m']} m vs {other['worst_waves_m']} m"
    return f"{name}: shorter ({best['distance_km']} km vs {other['distance_km']} km)"


def candidates(from_lat, from_lon, to_lat, to_lon, settings: dict):
    """Yield progress events: {"type": "candidate", ...} per candidate (a duplicate says which
    one it equals, a failed one says why), then {"type": "result", "candidates": [...the
    different ones...], "rows": [...all three, in CANDIDATES order...], ...}."""
    found, rows, started = [], [], time.time()
    for i, (kind, label, overrides) in enumerate(CANDIDATES, 1):
        head = {"type": "candidate", "kind": kind, "label": label, "index": i, "of": len(CANDIDATES)}
        try:
            r = checked_route(from_lat, from_lon, to_lat, to_lon, {**settings, **overrides})
        except routing.RouteError as exc:
            row = {"kind": kind, "label": label, "error": str(exc), "short": exc.short}
        else:
            c = summarize(kind, label, r)
            dup = _duplicate(c, found)
            if dup:
                dup_label = next(o["label"] for o in found if o["kind"] == dup)
                row = {"kind": kind, "label": label, "duplicate_of": dup, "duplicate_label": dup_label.split(" (")[0]}
            else:
                c["rule_check"] = r["rule_check"]
                found.append(c)
                row = {k: v for k, v in c.items() if k != "rule_check"}
        rows.append(row)
        yield {**head, **row, "seconds": round(time.time() - started, 1)}
    if not found:
        yield {"type": "result", "candidates": [], "rows": rows, "error": "No candidate route between these points."}
        return
    yield {"type": "result", "candidates": found, "rows": rows, **recommend(found),
           "seconds": round(time.time() - started, 1),
           "note": "Candidates checked against rules/passage_rules.yaml in code. " + routing.NOT_FOR_NAVIGATION}


def candidate_settings(kind: str, settings: dict) -> dict:
    overrides = next((o for k, _, o in CANDIDATES if k == kind), None)
    if overrides is None:
        raise routing.RouteError(f"Unknown candidate {kind!r}; known: {[k for k, _, _ in CANDIDATES]}")
    return {**settings, **overrides}


# ---------------------------------------------------------------------------
# Passage plan
# ---------------------------------------------------------------------------

def _bearing(a, b) -> float:
    """True course from a to b (degrees), in the local km grid (good enough over a few km)."""
    (x0, y0), (x1, y1) = tools._to_km(a[1], a[0]), tools._to_km(b[1], b[0])
    return round((math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360) % 360)


def passage_plan(from_lat, from_lon, to_lat, to_lon, kind: str, settings: dict) -> dict:
    """The passage plan for one candidate: waypoints with ETA, weather per leg, minimum UKC,
    nearest port of refuge per leg, and the rule-check summary."""
    r = checked_route(from_lat, from_lon, to_lat, to_lon, candidate_settings(kind, settings))
    speed_kmh = r["speed_kn"] * routing.KM_PER_NM
    departure = routing._departure((r.get("weather") or {}).get("departure_utc") or settings.get("departure"))
    path = r["path"]
    waypoints, dist = [], 0.0
    for i, p in enumerate(path):
        if i:
            dist += math.dist(tools._to_km(path[i - 1][1], path[i - 1][0]), tools._to_km(p[1], p[0]))
        eta = departure + pd.Timedelta(hours=dist / speed_kmh)
        port, port_km = rule_checker.harbour_distances(p[0], p[1])
        waypoints.append({"wp": f"WP{i + 1:02d}", "lat": p[0], "lon": p[1],
                          "distance_nm": round(dist / routing.KM_PER_NM, 1), "eta_utc": eta.isoformat(),
                          "course_to_next_deg": _bearing(p, path[i + 1]) if i + 1 < len(path) else None,
                          "nearest_port": port, "nearest_port_nm": round(port_km / routing.KM_PER_NM, 1)})
    draft = r["settings"]["draft_m"]
    legs = []
    w = r.get("weather") or {}
    for g in (w.get("legs") or []) + (w.get("no_forecast_legs") or []):
        pts = [tools._to_km(lon, lat) for lat, lon in g["coords"]]
        min_depth, _ = routing.depths_along(pts)
        port, port_km = rule_checker.harbour_distances(g["lat"], g["lon"])
        legs.append({**{k: g[k] for k in ("leg", "from_km", "to_km", "time_utc", "lat", "lon", "wind_bft",
                                         "gust_bft", "waves_m", "status")},
                     "min_ukc_m": round(min_depth - draft, 1), "port_of_refuge": port,
                     "port_of_refuge_nm": round(port_km / routing.KM_PER_NM, 1)})
    rc = r["rule_check"]
    return {
        "kind": kind, "label": next(lbl for k, lbl, _ in CANDIDATES if k == kind),
        "start": r["start"], "end": r["end"], "distance_km": r["distance_km"], "distance_nm": r["distance_nm"],
        "eta_text": r["eta_text"], "departure_utc": departure.isoformat(), "speed_kn": r["speed_kn"],
        "min_ukc_m": round(r["min_depth_m"] - draft, 1), "via": r.get("via", []), "waypoints": waypoints,
        "legs": legs, "weather_error": r.get("weather_error"),
        "no_forecast_legs": [g["leg"] for g in w.get("no_forecast_legs") or []],
        "forecast_end_utc": w.get("forecast_end_utc"),
        "rule_summary": {"counts": rc["counts"], "fails": rc["blocking"], "warnings": rc["warnings"],
                         "rows": [row for row in rc["rows"] if row["status"] in ("fail", "warn")]},
        "path": path, "note": routing.NOT_FOR_NAVIGATION,
    }


def gpx(plan: dict) -> str:
    """The passage plan as GPX 1.1 (a route with one point per waypoint, ETA as time)."""
    name = escape(f"{plan['start']['name']} to {plan['end']['name']} ({plan['label']})")
    pts = "\n".join(
        f'    <rtept lat="{w["lat"]:.5f}" lon="{w["lon"]:.5f}"><name>{w["wp"]}</name>'
        f'<time>{pd.Timestamp(w["eta_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ")}</time>'
        f'<desc>{escape(f"{w["distance_nm"]} nm; nearest port {w["nearest_port"]} {w["nearest_port_nm"]} nm")}</desc></rtept>'
        for w in plan["waypoints"])
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<gpx version="1.1" creator="weather-agent (learning project)" xmlns="http://www.topografix.com/GPX/1/1">\n'
            f'  <metadata><name>{name}</name><desc>Passage plan, {plan["distance_nm"]} nm, ETA {plan["eta_text"]}. '
            f'Checked against rules/passage_rules.yaml. NOT FOR NAVIGATION.</desc><time>{created}</time></metadata>\n'
            f'  <rte>\n    <name>{name}</name>\n{pts}\n  </rte>\n</gpx>\n')
