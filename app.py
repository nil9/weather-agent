"""Small web server for the forecast map.

  GET /              -> web/index.html (the map page)
  GET /api/forecast  -> reports/forecast.json (written by `python agent.py`)

Read-only: it serves two fixed files and takes no user input.
Run:  python app.py   then open http://127.0.0.1:8000
"""
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "web" / "index.html"
FORECAST_JSON = BASE_DIR / "reports" / "forecast.json"

HOST = "127.0.0.1"     # localhost only: other machines on the network can't connect
PORT = 8000

app = FastAPI(title="Marine weather map", docs_url=None, redoc_url=None)


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/forecast")
def forecast():
    if not FORECAST_JSON.exists():
        raise HTTPException(404, "No forecast yet. Run `python agent.py` first.")
    return json.loads(FORECAST_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT} in your browser (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=PORT)
