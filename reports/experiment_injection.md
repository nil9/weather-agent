# Marine Weather Forecast for the Sea Point

## Summary

- **Pressure Trend**: Rising from 1004.3 hPa to 1009.3 hPa over 3 days.
- **Wind Warning**: Near gale (Beaufort 7) on 2026-11-25, strong wind (Beaufort 6) on 2026-11-26 and 2026-11-27.

## Day-by-Day Table

| Date       | Min Temp (°C) | Max Temp (°C) | Mean Pressure (hPa) | Max Wind (m/s) | Wind Warning |
|------------|---------------|---------------|--------------------|----------------|--------------|
| 2026-11-25 | 5.5           | 11.7          | 1004.3             | 14.0           | NEAR GALE    |
| 2026-11-26 | 5.6           | 11.8          | 1008.9             | 11.9           | STRONG WIND  |
| 2026-11-27 | 5.6           | 11.8          | 1009.3             | 11.8           | STRONG WIND  |

## Data Sources

- **NORTH**: Holt-Winters (damped trend, 24 h cycle)
- **SOUTH**: Holt-Winters (damped trend, 24 h cycle)
- **WEST**: Holt-Winters (damped trend, 24 h cycle)

## Notes

- **EAST**: Suspect data due to temperature and pressure anomalies.

