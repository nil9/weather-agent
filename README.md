# Marine Weather Agent

A learning project: an AI agent built from scratch, with no agent frameworks.

Four weather stations on the real coast of Kiel Bay (Baltic Sea), NORTH, EAST, SOUTH and WEST,
surround a point in open water. Their weather data is synthetic. The agent forecasts the next
3 days at the sea point, saves a Markdown report, and shows the result on an interactive map.
Along the way it has to notice that station EAST has a broken sensor and leave it out.

**Python tools do all the calculations.** The local LLM (`qwen2.5:7b` via Ollama) only decides
which tool to call next and writes the final report from the tools' results.

## How to run

You need Python 3.10+ and [Ollama](https://ollama.com).

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Local LLM (Ollama must be running: `ollama serve`, or the desktop app)
ollama pull qwen2.5:7b

# 3. Create the synthetic data (fixed seed, same data every time)
python generate_data.py

#    ...and download the coastline once (Natural Earth land, clipped to Kiel Bay, ~26 KB)
python download_land.py

# 4. Optional: call every tool once, without the LLM
python test_tools.py

# 5. Run the agent
python agent.py
cat reports/forecast_report.md

# 5b. Optional: compare models (needs `ollama pull qwen2.5:14b`); the winner feeds the map
python compare_models.py

# 6. Show the forecast map
python app.py
# then open http://127.0.0.1:8000 in your browser (Ctrl+C stops the server)

# 6b. Sea routing data (depth + restricted zones), once; then the offline routing tests
python download_routing_data.py
python test_routing.py

# 7. Optional: real observations instead of the dummy data (internet needed, no API key)
WEATHER_DATA_SOURCE=real python agent.py     # fetches DWD data first (cached 60 min)
WEATHER_DATA_SOURCE=real python app.py       # map with "our estimate" next to Open-Meteo
python real_data.py                          # just refresh and list the stations
python test_real_data.py                     # offline tests (saved API answers)
```

**Real data** (`WEATHER_DATA_SOURCE=real`): hourly observations of the last 14 days from DWD
stations around Kiel Bay, via the [Bright Sky](https://brightsky.dev) API (no key). A station is
used only if it reports temperature, pressure and wind now and for ≥ 90 % of the 14 days; the
agent gets the 4 nearest to the sea point, a clicked point its own 4 nearest. Gaps up to 3 h are
filled linearly, so longer gaps still make a station SUSPECT. Data is cached in `data/real/` for 60 min;
if a refresh fails, the last good data is used and the page says "data from <time>". With no
data at all you get an error, never dummy data. The page shows the
[Open-Meteo](https://open-meteo.com) model forecast next to our 4-station estimate, over the same hours (comparison only).
Each mode keeps its own files: `reports/forecast_dummy.json` / `reports/forecast_real.json`
(the real report goes to `reports/real/`), so you can switch without rerunning the agent.
In real mode the sailing verdict uses the **higher** max wind of our estimate and Open-Meteo,
and the chat says which source it came from.
Sources: Deutscher Wetterdienst (DWD) via Bright Sky; Open-Meteo.com (CC BY 4.0).

The agent prints a trace of each step (which tool it chose, the arguments, a shortened result)
and saves the report to `reports/forecast_report.md`, plus the map data in `reports/forecast_dummy.json` (real mode: `reports/real/forecast_report.md` and `reports/forecast_real.json`).
A run takes about 20–60 seconds on a local GPU.

The map shows the stations (green = OK, red = SUSPECT; click one for details), the sea point with
the 3-day forecast and warning badges, and dashed lines weighted by each station's share of the
forecast. **Click anywhere** inside the dashed 40 km circles to get the forecast for that
point. It is computed in code with the same weighting as the agent, without the LLM.
Clicks on land get "This point is on land. Marine forecast only for water." (points within
about 500 m of the coast count as water, because the 1:10m coastline is coarse). The chat
then answers for that point. It needs an internet connection for the map tiles and the Leaflet
library. The server only listens on localhost.

The **Route** tab computes the shortest water route between two clicks (start, then destination)
inside the routing area (Kiel Bay to the Fehmarnbelt, Rødsand and Gedser: 9.8–12.3° E, 54.15–55.3° N,
drawn as a dashed boundary; a click outside it shows a note at the click and places no marker)
on a 0.25 km grid: it avoids land, water shallower than your draft plus a safety margin (default
2.0 + 1.0 m, EMODnet Bathymetry), and restricted zones from OpenStreetMap (no-entry areas always;
firing areas like Todendorf and Putlos avoided or, if you choose, allowed with a high penalty),
offshore wind farms (Rødsand II, Nysted; avoided or allowed with a penalty, and every turbine with a
safety radius, default 500 m, drawn on the map), and
keeps safety buffers (default 300 m from land, 500 m from closed zones). Shallow water is drawn as a
blue hatched layer. The panel shows the distance in km and nautical miles, the ETA for your speed
(default 6 knots), the minimum depth and buffer distances along the route, and a "Why this route?"
table: the direct water route, then the extra km added by the depth rule, the zones and the buffers.
In the named narrow passages (Fehmarnsund, Kiel Fjord at Friedrichsort) the buffers are reduced
(default 50 m from land, 100 m from zones; the depth rule stays), and the panel warns "narrow passage
(COLREG Rule 9): reduced clearance, keep to the fairway" and shows the full-buffer alternative. In
penalty mode the factor (0–20, default 10; 0 = zones cost nothing but are still listed) is set in the
tab, and the 500 m buffer around a crossable zone is penalised, not blocked. Invalid values are
marked in red instead of being replaced; any setting change recalculates the route. The Ask tab can explain a planned
route ("Why does the route go north?").

**Weather along the route** (Route tab): set the departure time (default now). The route is cut into
3 km legs; each leg shows the forecast for the time the boat passes it (wind in Beaufort, gusts,
waves; real data: Open-Meteo Forecast + Marine, one batched request per API; dummy data: wind only),
coloured green / yellow / red ("too rough" above Bft 6 or 1.5 m waves, adjustable). The
"weather-aware" variant avoids rough legs where possible and is compared with the shortest route.
In the Ask tab, "Is it safe along the way?" answers per leg with times. Code only, no LLM. Firing areas count as always active, and
there is no traffic or weather yet, so it is not for navigation.

The **Ask** tab is a chat about this forecast, e.g. "Is it safe to sail tomorrow?" or "Why is EAST
excluded?". The LLM only has read-only tools. Whether sailing is safe / caution / not safe is
decided in code (`sail_safety`: wind in Beaufort plus pressure trend), and the LLM explains it.
Questions about other topics get "I can only answer about this forecast." Dummy data, not
navigation advice. Run `python test_chat.py` to check the chat against a fixed set of questions
in both data modes (`--source dummy` or `--source real` for one).

## Data sources and licences

- Coastline: Natural Earth 1:10m land (public domain).
- Water depth (Route tab): **EMODnet Bathymetry** Digital Terrain Model, © EMODnet Bathymetry Consortium,
  licensed **CC BY 4.0** (`data/depth_kiel_bay.npz`).
- Restricted zones and wind farms (Route tab): **© OpenStreetMap contributors**, licensed **ODbL**
  (`data/restricted_zones_kiel_bay.geojson` and `data/wind_farms_kiel_bay.geojson`, from OpenStreetMap
  and OpenSeaMap tags via the Overpass API).
- Real-mode observations: Deutscher Wetterdienst (DWD) via Bright Sky; model values: Open-Meteo.com (CC BY 4.0).
- Map tiles: © OpenStreetMap contributors.

## Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Writes `data/stations.json` and `data/observations.csv` (30 days, hourly) |
| `download_land.py` | One-time download of Natural Earth 1:10m land polygons, clipped to Kiel Bay → `data/land_kiel_bay.geojson` (for the land/sea check) |
| `routing.py` | Sea routing (Route tab): shortest water route on a 0.5 km grid over Kiel Bay (A*, no LLM) that keeps enough water depth for the draft and avoids restricted zones; distance in km / nm, ETA and minimum depth |
| `download_routing_data.py` | One-time download: EMODnet Bathymetry depth (→ `data/depth_kiel_bay.npz`), restricted zones (→ `data/restricted_zones_kiel_bay.geojson`) and offshore wind farms + turbines (→ `data/wind_farms_kiel_bay.geojson`) from OpenStreetMap; `--wind-only` for the wind farms alone |
| `test_routing.py` | Routing tests: never crosses land, same point = 0 km, around Fehmarn, ETA; depth (shallow areas, draft changes the route), firing areas avoided, penalty mode |
| `real_data.py` | Real mode: finds DWD stations (Bright Sky), fetches and converts 14 days of observations into `data/real/current/`, cache + fallback, Open-Meteo comparison values |
| `test_real_data.py` | Offline tests for the real-data path, using saved API answers in `tests/fixtures/` |
| `tools.py` | The 6 tools (stations, observations, data quality, Holt-Winters forecast, sea-point estimate, write report) and their JSON specs |
| `agent.py` | The agent loop: LLM → tool call → result → LLM … → final answer |
| `test_tools.py` | Calls each tool once and prints the result |
| `app.py` | Small FastAPI server (localhost only): the map page at `/`, the forecast JSON at `/api/forecast`, a clicked point at `/api/forecast_at`, the chat at `POST /api/chat` |
| `chat.py` | Chat logic: topic gate, read-only tool loop, answer checks (numbers, verdicts, required tools) |
| `test_chat.py` | Chat regression tests in both data modes: the chips and note of each mode, normal, off-topic, clicked-point and regression questions; answers checked against the tools, no internal names or ISO dates, verdicts only for the named days (`--runs N`, `--source dummy/real/both`) |
| `web/index.html` | The map page: Leaflet + OpenStreetMap tiles with a dark CSS filter (no API key), Forecast / Ask tabs |
| `compare_models.py` | Runs the agent with each model in `agent.MODELS` one after another, ranks the runs in code, saves the winner to `forecast_<source>.json` and all runs to `model_comparison_<source>.json` |
| `test_compare.py` | Tests the ranking rules (no LLM needed) |
| `experiments.py` | Runs the agent in "break" scenarios with pass/fail checks: `python experiments.py <scenario>` or `python experiments.py all --runs 5 [--temperature 0.7] [--model qwen2.5:14b]` |
| `docs/architecture.md` | Diagrams and build progress |
| `reports/logs/` | Saved outputs of experiment, comparison and chat-test runs |

## Architecture

Diagrams (components, data sources, one agent run) and the build progress are in
[docs/architecture.md](docs/architecture.md).

The LLM never runs code or touches files. It only *asks* for a tool call, and `agent.py` runs
the real Python function and sends the JSON result back. Tools may only read `data/` and write
`reports/`. Important safety rules live in code, not just in the prompt: tools refuse
unreliable stations, the loop won't accept a final answer until a report is saved, and it stops
when a tool keeps failing.
