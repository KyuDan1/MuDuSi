"""Read-only, isolated-room presentation of one concurrent matrix experiment."""
import json
import os
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .presentation import load_presentation


def load_matrix(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "matrix.json").read_text())
    rooms, audio, seen = [], {}, set()
    for spec in manifest["rooms"]:
        key = spec["id"]
        if not re.fullmatch(r"[a-z0-9-]+", key) or key in seen:
            raise ValueError("Matrix room identifiers must be unique safe slugs")
        seen.add(key)
        folder = (root / spec.get("directory", key)).resolve()
        if not folder.is_relative_to(root):
            raise ValueError("Matrix recordings must remain inside the experiment directory")
        recording, pcm = load_presentation(folder)
        if len(recording["participants"]) != 2:
            raise ValueError(f"Room {key} must contain exactly two model participants")
        rooms.append({"id": key, "left": spec["left"], "right": spec["right"], "recording": recording})
        audio[key] = pcm
    if len(rooms) != 9:
        raise ValueError("A three-by-three matrix requires exactly nine rooms")
    return {"duration": min(r["recording"]["duration"] for r in rooms),
            "model_order": manifest.get("model_order", ["Moshi", "PersonaPlex", "Raon"]),
            "rooms": rooms, "concurrency": {key: manifest.get(key) for key in (
                "job_id", "wall_start_utc", "wall_end_utc", "start_spread_ms",
                "completed", "completed_rooms", "simultaneous_room_count")},
            "audio_kind": "One selected room at a time; rooms never hear other rooms"}, audio


def create_matrix_app(directory):
    data, audio = load_matrix(directory)
    app = FastAPI(title="MuDuSi — Nine isolated conversations")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "mode": "matrix", "rooms": len(data["rooms"])}

    @app.get("/api/matrix")
    async def matrix_data():
        return data

    @app.get("/api/matrix/audio/{room_id}.wav")
    async def matrix_audio(room_id: str):
        if room_id not in audio:
            raise HTTPException(404, "Unknown room")
        return Response(audio[room_id], media_type="audio/wav")

    dist = Path(os.environ.get("MUDUSI_WEB_DIR", os.environ.get("DUPLEX_ARENA_WEB_DIR", Path.cwd() / "web" / "dist")))
    if not dist.exists():
        dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        async def index():
            return FileResponse(dist / "index.html")
    return app
