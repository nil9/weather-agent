"""Small web server for the forecast map.

  GET  /              -> web/index.html (the map page)
  GET  /api/forecast  -> reports/forecast.json (written by `python agent.py`)
  POST /api/chat      -> answer a question about the forecast (chat.py, read-only tools)

Nothing here writes files. The only user input is the chat question, which is
length-checked and passed to the LLM, which can only use read-only tools.
Run:  python app.py   then open http://127.0.0.1:8000
"""
import json
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import chat

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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[ChatMessage] = Field(default_factory=list, max_length=chat.MAX_HISTORY)


@app.post("/api/chat")
def ask(request: ChatRequest):
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so a slow LLM call
    # doesn't block the map from loading.
    try:
        return chat.answer_question(request.question.strip(),
                                    [m.model_dump() for m in request.history])
    except chat.NoForecast as exc:
        raise HTTPException(404, str(exc))
    except chat.ChatUnavailable as exc:
        raise HTTPException(503, f"The local LLM is not reachable (is Ollama running?): {exc}")


if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT} in your browser (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=PORT)
