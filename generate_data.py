"""Generate dummy marine weather data for 4 stations around Kiel Bay.

Writes:
  data/stations.json     - coordinates of the stations and the sea forecast point
  data/observations.csv  - 30 days of hourly observations per station

Hidden scenarios (the agent's tools should discover these later):
  1. A low-pressure system moves in from the WEST during the last 2 days:
     pressure drops, wind strengthens and veers, humidity rises.
  2. Station EAST has a faulty sensor:
     - temperature gets stuck at one value for the last 9 days
     - pressure has impossible single-hour jumps (+/- 25..45 hPa)
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SEED = 42                               # fixed seed -> same data every run
DAYS = 30
END_TIME = datetime(2026, 11, 25, 0, 0)  # first hour AFTER the last observation (UTC)

# Kiel Bay (Baltic Sea). The stations sit on real coastline; the weather data is
# still synthetic. Each station has small local quirks:
#   temp_offset: local climate (degC)
#   low_lag_h:   how many hours after WEST the low-pressure system arrives
STATIONS = [
    {"id": "NORTH", "name": "Marstal (Ærø)",
     "lat": 54.856, "lon": 10.517, "temp_offset": -0.8, "low_lag_h": 4},
    {"id": "EAST",  "name": "Westermarkelsdorf (Fehmarn)",
     "lat": 54.528, "lon": 11.060, "temp_offset":  0.3, "low_lag_h": 8},
    {"id": "SOUTH", "name": "Schönberger Strand",
     "lat": 54.425, "lon": 10.378, "temp_offset":  0.9, "low_lag_h": 4},
    {"id": "WEST",  "name": "Damp",
     "lat": 54.586, "lon": 10.017, "temp_offset":  0.0, "low_lag_h": 0},
]
# Open water in the middle of Kiel Bay.
FORECAST_POINT = {"id": "SEA", "name": "Kiel Bay (open water)", "lat": 54.55, "lon": 10.45}

# Low-pressure system settings
LOW_DEPTH_HPA = 28      # how deep the low gets below normal pressure
LOW_WIDTH_H = 14        # how long it lasts (bigger = slower)
LOW_PEAK_BEFORE_END_H = 12  # WEST reaches the lowest pressure 12 h before the data ends

# EAST sensor fault settings
STUCK_TEMP_DAYS = 9
N_PRESSURE_SPIKES = 6


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_station(station, timestamps, rng):
    """Return a DataFrame of hourly observations for one station."""
    n = len(timestamps)
    t = np.arange(n)                     # hour index 0..n-1
    hour = timestamps.hour.to_numpy()

    # Common building blocks
    diurnal = np.sin(2 * np.pi * (hour - 8) / 24)  # daily cycle, peaks at 14:00
    synoptic = np.sin(2 * np.pi * t / (24 * 5))    # slow ~5-day weather cycle

    # Low-pressure system: a bell-shaped dip that arrives later further east.
    low_center = n - LOW_PEAK_BEFORE_END_H + station["low_lag_h"]
    low = np.exp(-(((t - low_center) / LOW_WIDTH_H) ** 2))  # 0 = no low, 1 = peak

    # Pressure (hPa)
    pressure = 1015 + 6 * synoptic - LOW_DEPTH_HPA * low + rng.normal(0, 0.3, n)

    # Temperature (degC): November-ish, slightly mild air in the low's warm sector
    temp = (8 + station["temp_offset"] + 3 * diurnal + 1.5 * synoptic
            + 1.0 * low + rng.normal(0, 0.4, n))

    # Wind speed (m/s): stronger when pressure is low
    wind_speed = 5 + 1.5 * synoptic + 12 * low + rng.normal(0, 1.0, n)
    wind_speed = np.clip(wind_speed, 0, None)

    # Wind direction (deg, where the wind comes FROM).
    # Normally south-westerly. As the low passes, wind veers from S (180) to NW (300).
    normal_dir = 225 + 25 * synoptic
    low_dir = 180 + 120 * sigmoid((t - low_center) / 6)
    wind_dir = (1 - low) * normal_dir + low * low_dir + rng.normal(0, 12, n)
    wind_dir = np.mod(wind_dir, 360)

    # Relative humidity (%): lower in the afternoon, higher in the low
    humidity = 80 - 8 * diurnal + 15 * low + rng.normal(0, 3, n)
    humidity = np.clip(humidity, 30, 100)

    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station["id"],
        "temperature_c": temp,
        "pressure_hpa": pressure,
        "wind_speed_ms": wind_speed,
        "wind_dir_deg": wind_dir,
        "humidity_pct": humidity,
    })


def break_east_sensor(df, rng):
    """Inject the faults into station EAST (modifies df in place)."""
    n = len(df)

    # Fault 1: temperature freezes at one value for the last STUCK_TEMP_DAYS days.
    stuck_from = n - STUCK_TEMP_DAYS * 24
    df.loc[stuck_from:, "temperature_c"] = df.loc[stuck_from, "temperature_c"]

    # Fault 2: single-hour pressure spikes that no real atmosphere can produce.
    # They are kept out of the first 10 days and the last 2 days, where the low is.
    spike_hours = rng.choice(np.arange(10 * 24, n - 48), N_PRESSURE_SPIKES, replace=False)
    jumps = rng.uniform(25, 45, N_PRESSURE_SPIKES) * rng.choice([-1, 1], N_PRESSURE_SPIKES)
    df.loc[spike_hours, "pressure_hpa"] += jumps


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    start = END_TIME - timedelta(days=DAYS)
    timestamps = pd.date_range(start, periods=DAYS * 24, freq="h")

    frames = []
    for station in STATIONS:
        df = make_station(station, timestamps, rng)
        if station["id"] == "EAST":
            break_east_sensor(df, rng)
        frames.append(df)

    obs = pd.concat(frames, ignore_index=True)
    numeric_cols = obs.select_dtypes("number").columns
    obs[numeric_cols] = obs[numeric_cols].round(1)
    obs["timestamp"] = obs["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    obs.to_csv(DATA_DIR / "observations.csv", index=False)

    stations_json = {
        "stations": [{"id": s["id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"]}
                     for s in STATIONS],
        "forecast_point": FORECAST_POINT,
    }
    (DATA_DIR / "stations.json").write_text(json.dumps(stations_json, indent=2))

    print(f"Wrote {len(obs)} rows to {DATA_DIR / 'observations.csv'}")
    print(f"Wrote {DATA_DIR / 'stations.json'}")


if __name__ == "__main__":
    main()
