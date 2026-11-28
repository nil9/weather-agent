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
python test_route_weather.py                 # weather along the route (offline fixture)
python test_voyage.py                        # rulebook, rule check, profiles, candidates, passage plan

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
inside the routing area (Kiel Bay to the Fehmarnbelt, Gedser, Møn and Faxe Bugt: 9.8–12.8° E, 54.15–55.3° N,
drawn as a dashed boundary; a click outside it shows a note at the click and places no marker)
on a 0.25 km grid: it avoids land, water shallower than your draft plus a safety margin (default
2.0 + 1.0 m, EMODnet Bathymetry), and restricted zones from OpenStreetMap (no-entry areas always;
firing areas like Todendorf and Putlos avoided or, if you choose, allowed with a high penalty),
offshore wind farms (Rødsand II, Nysted, EnBW Baltic 1, Kriegers Flak; avoided or allowed with a penalty, and every turbine with a
safety radius, default 500 m, drawn on the map), and
keeps safety buffers (default 300 m from land, 500 m from closed zones). Shallow water is drawn as a
blue hatched layer. The panel shows the distance in km and nautical miles, the ETA for your speed
(default 6 knots), the minimum depth and buffer distances along the route, and a "Why this route?"
table: the direct water route, then the extra km added by the depth rule, the zones and the buffers.
In the named narrow passages (Fehmarnsund, Kiel Fjord at Friedrichsort, and the Danish sounds
Guldborgsund, Storstrøm / Masnedsund with the Farø narrows, Ulvsund / Bøgestrøm, Grønsund) the buffers
are reduced (default 50 m from land, 100 m from zones; the depth rule stays), and the panel warns "narrow
passage (COLREG Rule 9): reduced clearance, keep to the fairway" and shows the full-buffer alternative.
The panel also says which way the route goes ("Via: around Møn (east of Møns Klint) → Fehmarnsund").
With 2.0 + 1.0 m, Bøgestrøm stays closed (the depth model has only 0.3 m there), so Faxe Bugt →
Heiligenhafen goes around Møn (158 km). When there is no route, the message names what blocks the
connection to the open sea: a too shallow or too narrow passage (with the values), a closed zone, the
routing-area edge, or a channel that the coastline data doesn't show. A route that isn't possible
is shown in a red box at the top of the Route tab (a short headline such as "Start: Too shallow for
Large ship: 7.0 m here, needs 16.11 m", the details below, a hint like "Try a smaller vessel profile
or click in deeper water"), and the point that failed gets a red × with the reason in a popup.
A click on land (or too close to it) is moved to the nearest usable water: the marker is drawn
there, with a small dashed line from the click. In
penalty mode the factor (0–20, default 10; 0 = zones cost nothing but are still listed) is set in the
tab, and the 500 m buffer around a crossable zone is penalised, not blocked. Invalid values are
marked in red instead of being replaced; any setting change recalculates the route. The Ask tab can explain a planned
route ("Why does the route go north?").

**Weather along the route** (Route tab): set the departure time (default now). The route is cut into
3 km legs; each leg shows the forecast for the time the boat passes it (wind in Beaufort, gusts,
waves; real data: Open-Meteo Forecast + Marine, one batched request per API; dummy data: wind only),
coloured green / yellow / red ("too rough" above Bft 6 or 1.5 m waves, adjustable). The
"weather-aware" variant avoids rough legs where possible and is compared with the shortest route.
If the trip runs past the end of the forecast, the route is still given: the legs after the end are
listed as "no forecast" (grey dashed on the map), in the panel and in the passage plan, and rule
R-WX-04 warns.
In the Ask tab, questions about the weather along the route are answered in code (no LLM). The first
sentence answers the question itself: "Will there be a storm along the way?" → "No storm expected:
strongest is Beaufort 5 (gusts Beaufort 6) at 15:39 UTC on leg 3." (storm = Beaufort 10+, gale = 8+,
strong wind = 6–7); "Is this route safe?" → "Within your limits …" / "Not within your limits …"
(against the limits set in the Route tab); "When is it worst?" → time and place of the worst leg. Then
one summary line (departure, arrival, legs above the limits, worst leg). The leg list is a table
under "Details". Each weather answer has a label on top: "Along your route" or "Sea point: Kiel Bay
(open water)", and the "Answering for:" line names the planned route ("Your route: Heiligenhafen →
Faxe Bugt"). With a route, "Is it safe to sail tomorrow?" is answered for the route departing
tomorrow at the same time of day, plus one line "Sea point for comparison: …". Firing areas count
as always active, and there is no traffic data, so it is not for navigation.

The **Ask** tab is a chat about this forecast, e.g. "Is it safe to sail tomorrow?" or "Why is EAST
excluded?". The LLM only has read-only tools. Whether sailing is safe / caution / not safe is
decided in code (`sail_safety`: wind in Beaufort plus pressure trend), and the LLM explains it.
Questions about other topics get "I can only answer about this forecast." In a mixed question
("can I dance and laugh while sailing tomorrow?") the LLM only sees the weather part ("Can I sail
tomorrow?"); code adds "The other part of your question is outside what this forecast can answer."
Safety words are typo-tolerant ("tsafe", "safty"), and a question with a safety word is a safety
question even if it names the route; if the intent is unclear ("Is my route ok?"), the chat asks back
with two clickable chips. Dummy data, not navigation advice. Run `python test_chat.py` to check the chat against a fixed set
of questions in both data modes (`--source dummy` or `--source real` for one). The tests freeze
"now" (`clock.py`; dummy data: a time inside the dummy forecast), so they pass on any day.

## Voyage planning (rulebook, rule check, recommendation)

The Route tab checks every route against a **rulebook** built from
`docs/Ship_Voyage_Planning.pdf`: `rules/passage_rules.yaml` has one entry per factor with an id
(e.g. `R-UKC-01`), the regulation or guidance the document names (SOLAS V/34, IMO A.893(21), COLREG
Rule 9, ISM / Nautical Institute UKC, …), a type (`hard`, `penalty`, `check`, `advisory`,
`not_modelled`), its parameters and the data it uses. Checked now: under-keel clearance, squat
(Barrass), depth-data resolution, no-entry / firing / exercise areas, wind farms, clearing
distances, narrow passages (COLREG Rule 9), air draught under six fixed bridges (bridge lines from
OpenStreetMap; OSM has no clearance tags for them, so the clearances are hand values marked
"approximate - verify on chart", and the old Storstrøm Bridge and the north Farø Bridge have none:
"clearance unknown - verify on chart" warns), wind / gusts / waves along the route, forecast coverage
(warns and lists the legs after the end of the forecast), pilot
recommendation (≥ 11 m draught), Baltimax, Route T (≥ 40,000 dwt), ports of refuge, and the
routing-area edge. Not modelled (with the reason): ENCs and Notices to Mariners, navigational
warnings, AIS traffic, ice, currents and water level, TSS, pilotage areas, reporting / VTS,
visibility, emissions and security.

- **Rule check** (`rule_checker.py`): every route gets a compact table (failed and warned rules
  first, value and limit per rule, not-modelled rules collapsed). The route settings feed the
  rules; gust limit and air draught are new settings.
- **Vessel profiles** (`rules/vessel_profiles.yaml`): small sailboat (the old defaults), motor
  yacht, coaster, large ship. A profile fills the settings (draft, UKC margin from its UKC policy,
  air draught, speed, wind / gust / wave limits); you can still change every value. The profile
  also gives the squat block coefficient and the dwt for the size rules. For the motor yacht,
  coaster and large ship the router adds the squat (Barrass: block coefficient × speed² / 100, with
  the speed set in the tab) to the required depth: draft + squat + margin. The small sailboat has
  no squat allowance, so its routes are unchanged.
- **Candidates** ("Compare route candidates"): shortest, balanced (weather-aware) and safest
  (weather-aware, full buffers, all zones avoided), with live progress. The table always has all
  three columns: a candidate that equals another says "same route as Shortest", one that can't be
  calculated says why.
  Scored in code: any failed rule = not recommended; then fewer warnings, lower worst wind, lower
  waves, shorter. You can pick any candidate. Its **passage plan** lists the waypoints with ETA and
  course, weather and minimum UKC per leg, the nearest port of refuge per leg and the rule-check
  summary, and downloads as **GPX** (`/api/passage_plan.gpx`).
- **Chat**: "Which rules does it break?", "Why not the shortest?" and "Which route do you
  recommend?" are answered in code with rule ids, values and limits (label "Your route: rule
  check"); "Why this route?" ends with a rule-check line. If you picked a candidate, the chat
  explains that one. Other questions can use the read-only tool `get_route_recommendation`.

For analysis and learning only. Not for navigation.

## Data sources and licences

- Coastline: Natural Earth 1:10m land (public domain; `data/land_kiel_bay.geojson`, 9.0–13.3° E, 53.8–55.8° N).
- Water depth (Route tab): **EMODnet Bathymetry** Digital Terrain Model, © EMODnet Bathymetry Consortium,
  licensed **CC BY 4.0** (`data/depth_kiel_bay.npz`).
- Restricted zones and wind farms (Route tab): **© OpenStreetMap contributors**, licensed **ODbL**
  (`data/restricted_zones_kiel_bay.geojson` and `data/wind_farms_kiel_bay.geojson`, from OpenStreetMap
  and OpenSeaMap tags via the Overpass API).
- Bridge lines for the air-draught rule (R-AIR-01): **© OpenStreetMap contributors**, licensed **ODbL**
  (`data/bridges_kiel_bay.geojson`, `python download_routing_data.py --bridges-only`). The clearances
  are not from OSM: they are approximate hand values in the rulebook.
- The routing data covers the routing area 9.8–12.8° E, 54.15–55.3° N and was downloaded on 27 Nov 2026
  (`python download_land.py`, `python download_routing_data.py`).
- Real-mode observations: Deutscher Wetterdienst (DWD) via Bright Sky; model values: Open-Meteo.com (CC BY 4.0).
- Map tiles: © OpenStreetMap contributors.

## Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Writes `data/stations.json` and `data/observations.csv` (30 days, hourly) |
| `download_land.py` | One-time download of Natural Earth 1:10m land polygons, clipped to Kiel Bay → `data/land_kiel_bay.geojson` (for the land/sea check) |
| `routing.py` | Sea routing (Route tab): shortest water route on a 0.5 km grid over Kiel Bay (A*, no LLM) that keeps enough water depth for the draft and avoids restricted zones; distance in km / nm, ETA and minimum depth |
| `download_routing_data.py` | One-time download: EMODnet Bathymetry depth (→ `data/depth_kiel_bay.npz`), restricted zones (→ `data/restricted_zones_kiel_bay.geojson`) and offshore wind farms + turbines (→ `data/wind_farms_kiel_bay.geojson`) and the bridge lines of rule R-AIR-01 (→ `data/bridges_kiel_bay.geojson`) from OpenStreetMap; `--wind-only` / `--bridges-only` for one of them alone |
| `test_routing.py` | Routing tests: never crosses land, same point = 0 km, around Fehmarn, ETA; depth (shallow areas, draft changes the route), firing areas avoided, penalty mode |
| `real_data.py` | Real mode: finds DWD stations (Bright Sky), fetches and converts 14 days of observations into `data/real/current/`, cache + fallback, Open-Meteo comparison values |
| `test_real_data.py` | Offline tests for the real-data path, using saved API answers in `tests/fixtures/` |
| `tools.py` | The 6 tools (stations, observations, data quality, Holt-Winters forecast, sea-point estimate, write report) and their JSON specs |
| `agent.py` | The agent loop: LLM → tool call → result → LLM … → final answer |
| `test_tools.py` | Calls each tool once and prints the result |
| `app.py` | Small FastAPI server (localhost only): the map page at `/`, the forecast JSON at `/api/forecast`, a clicked point at `/api/forecast_at`, the chat at `POST /api/chat`, the default departure at `/api/clock` |
| `chat.py` | Chat logic: topic gate, read-only tool loop, answer checks (numbers, verdicts, required tools) |
| `clock.py` | The one place that reads the current time (`clock.now()`): real data the real time, dummy data the dummy calendar (first forecast day, 08:00 UTC, shown in the Route tab as "Dummy time: …"); tests freeze it |
| `test_chat.py` | Chat regression tests in both data modes: the chips and note of each mode, normal, off-topic, clicked-point and regression questions; answers checked against the tools, no internal names or ISO dates, verdicts only for the named days (`--runs N`, `--source dummy/real/both`) |
| `web/index.html` | The map page: Leaflet + OpenStreetMap tiles with a dark CSS filter (no API key), Forecast / Ask tabs |
| `compare_models.py` | Runs the agent with each model in `agent.MODELS` one after another, ranks the runs in code, saves the winner to `forecast_<source>.json` and all runs to `model_comparison_<source>.json` |
| `test_compare.py` | Tests the ranking rules (no LLM needed) |
| `experiments.py` | Runs the agent in "break" scenarios with pass/fail checks: `python experiments.py <scenario>` or `python experiments.py all --runs 5 [--temperature 0.7] [--model qwen2.5:14b]` |
| `rules/passage_rules.yaml` | The passage-planning rulebook (rule ids, sources, types, parameters, data), from `docs/Ship_Voyage_Planning.pdf` |
| `rules/vessel_profiles.yaml` | Vessel profiles (small sailboat, motor yacht, coaster, large ship) that fill the Route tab settings |
| `rule_checker.py` | Loads the rulebook and the profiles; checks a route against every rule (pass / warn / fail / not modelled) |
| `voyage.py` | Route candidates, the recommendation (scored in code), the passage plan and its GPX export |
| `test_voyage.py` | Offline tests: rulebook, rule check, profiles, candidates, recommendation, passage plan, GPX |
| `docs/Ship_Voyage_Planning.pdf` | The source document of the rulebook |
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
