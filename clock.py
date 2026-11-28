"""The one place that reads the current time ("now").

Code that needs "now" (the default departure, "today" in the chat, cache ages) calls
clock.now():
- real data: the real time
- dummy data: the dummy calendar, DUMMY_START_HOURS after the start of the dummy forecast
  (the synthetic scenario has its own fixed dates, so its "now" is fixed too)

Tests freeze it, so they give the same result on any day:

    clock.freeze_for_tests()

The app never freezes it.
"""
import pandas as pd

DUMMY_START_HOURS = 8          # dummy "now": the first forecast day, 08:00 UTC

_frozen: pd.Timestamp | None = None


def dummy_now() -> pd.Timestamp | None:
    """The dummy calendar's "now" (None if there is no dummy forecast yet)."""
    import tools
    try:
        dates = tools.forecast_dates()
    except Exception:                       # no forecast for this data source yet
        return None
    if not dates:
        return None
    return pd.Timestamp(dates[0], tz="UTC") + pd.Timedelta(hours=DUMMY_START_HOURS)


def is_dummy_time() -> bool:
    """Does now() follow the dummy calendar (dummy data, not frozen)?"""
    import tools
    return _frozen is None and tools.DATA_SOURCE == "dummy" and dummy_now() is not None


def now() -> pd.Timestamp:
    """The current time in UTC: the frozen time in tests, the dummy calendar for dummy data,
    otherwise the real time."""
    if _frozen is not None:
        return _frozen
    import tools
    if tools.DATA_SOURCE == "dummy":
        dummy = dummy_now()
        if dummy is not None:
            return dummy
    return pd.Timestamp.now(tz="UTC")


def info() -> dict:
    """What the Route tab shows: the default departure and, for dummy data, a clear label."""
    t = now().floor("min")
    dummy = is_dummy_time()
    return {"now_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "dummy": dummy,
            "label": f"Dummy time: {t:%a %d %b %H:%M} UTC (synthetic scenario)" if dummy else None}


def freeze(when) -> pd.Timestamp:
    """Use `when` as "now" from here on (None: the normal clock again)."""
    global _frozen
    if when is None:
        _frozen = None
        return now()
    t = pd.Timestamp(when)
    _frozen = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    return _frozen


def freeze_for_tests() -> pd.Timestamp:
    """Dummy data: the dummy calendar's "now" (inside the dummy forecast on any day). Real data:
    the real forecast is fetched live and starts today, so "now" is frozen at the real time
    once, at the start of the run (a run over midnight keeps one "today")."""
    return freeze(now())
