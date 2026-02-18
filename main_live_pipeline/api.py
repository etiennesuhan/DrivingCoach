from __future__ import annotations

import json
import mimetypes
import queue
import sqlite3
import threading
import time
import csv
import io
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from listener import Listener, resolve_repo_path

DEFAULT_SAMPLE_FIELDS = [
    "timestamp_utc",
    "lap_number",
    "position_x",
    "position_z",
    "speed",
    "yaw",
]
MAX_SAMPLE_LIMIT = 10000


def create_app(listener: Listener) -> FastAPI:
    app = FastAPI(title="FH5 Live Pipeline API", version="0.1.0")
    app.state.listener = listener
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    metadata_lock = threading.Lock()

    def _db_path() -> Path:
        db_path = getattr(listener, "DB_FILE", None)
        if db_path:
            return Path(db_path)
        return resolve_repo_path("data/test.db")

    def _open_db() -> sqlite3.Connection:
        db_path = _db_path()
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"DB not found: {db_path}")
        conn = sqlite3.connect(db_path.as_posix())
        conn.row_factory = sqlite3.Row
        return conn

    def _get_columns(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute("PRAGMA table_info(telemetry_samples)").fetchall()
        if not rows:
            return []
        return [row["name"] for row in rows]

    def _select_fields(columns: list[str], fields: str | None) -> list[str]:
        allowed = [c for c in columns if c != "raw"]
        if fields is None:
            return [c for c in DEFAULT_SAMPLE_FIELDS if c in allowed]
        if fields.strip().lower() == "all":
            return allowed
        requested = [f.strip() for f in fields.split(",") if f.strip()]
        if not requested:
            raise HTTPException(status_code=400, detail="fields must not be empty")
        unknown = [f for f in requested if f not in allowed]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown fields: {unknown}")
        return requested

    def _read_metadata() -> dict:
        metadata_path = resolve_repo_path("main_live_pipeline/metadata.json")
        if not metadata_path.exists():
            return {"tracks": {}}
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read metadata: {exc}")
        if isinstance(data, dict) and "tracks" in data:
            return data
        if isinstance(data, dict):
            return {"tracks": data}
        return {"tracks": {}}

    def _write_metadata(data: dict) -> None:
        metadata_path = resolve_repo_path("main_live_pipeline/metadata.json")
        try:
            metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to write metadata: {exc}")

    def _normalize_track_id(payload: dict) -> str:
        if "track_id" in payload and payload["track_id"]:
            return str(payload["track_id"])
        point = payload.get("point")
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                x = int(point[0])
                z = int(point[1])
            except Exception:
                raise HTTPException(status_code=400, detail="point must contain 2 integers")
            return f"[{x}, {z}]"
        raise HTTPException(status_code=400, detail="track_id or point required")

    def _validate_segments(segments) -> list[list[int]]:
        if segments is None:
            return []
        if not isinstance(segments, list):
            raise HTTPException(status_code=400, detail="segments must be a list")
        out = []
        for seg in segments:
            if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                raise HTTPException(status_code=400, detail="each segment must be [x, z]")
            try:
                out.append([int(seg[0]), int(seg[1])])
            except Exception:
                raise HTTPException(status_code=400, detail="segment coordinates must be integers")
        return out

    def _fetch_lap_positions(
        lap_id: int,
        step: int = 1,
        limit: int | None = None,
        fields: list[str] | None = None,
        track_id: str | None = None,
    ) -> list[dict]:
        conn = _open_db()
        try:
            columns = _get_columns(conn)
            if not columns:
                return []
            selected = fields if fields else [c for c in ["timestamp_utc", "position_x", "position_z", "speed", "yaw", "lap_number"] if c in columns]
            where_clauses = ["lap_id = ?"]
            params: list[object] = [lap_id]
            if track_id is not None and "track_id" in columns:
                where_clauses.append("track_id = ?")
                params.append(track_id)
            query = (
                f"SELECT {', '.join(selected)} "
                "FROM telemetry_samples "
                f"WHERE {' AND '.join(where_clauses)} "
                "ORDER BY id ASC"
            )
            rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail=f"DB error: {exc}")
        finally:
            conn.close()
        items = [dict(row) for row in rows]
        if step and step > 1:
            items = items[::step]
        if limit and limit > 0:
            items = items[:limit]
        return items

    def _resolve_track_filter(track_id: str | None) -> str | None:
        if track_id is not None:
            normalized = str(track_id).strip()
            return normalized if normalized else None
        current = getattr(listener, "track_id", None)
        if current is None:
            return None
        normalized = str(current).strip()
        return normalized if normalized else None

    def _fetch_lap_track_id(conn: sqlite3.Connection, lap_id: int) -> str | None:
        columns = _get_columns(conn)
        if "track_id" not in columns:
            return None
        row = conn.execute(
            """
            SELECT track_id
            FROM telemetry_samples
            WHERE lap_id = ? AND track_id IS NOT NULL AND track_id != ''
            ORDER BY id ASC
            LIMIT 1
            """,
            (lap_id,),
        ).fetchone()
        if row is None:
            return None
        return row["track_id"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict:
        return listener.get_status()

    @app.get("/tracks")
    def tracks() -> dict:
        data = _read_metadata()
        tracks_data = data.get("tracks", {})
        return {"tracks": tracks_data}

    @app.get("/tracks/{track_id}")
    def track(track_id: str) -> dict:
        data = _read_metadata()
        tracks_data = data.get("tracks", {})
        if track_id not in tracks_data:
            raise HTTPException(status_code=404, detail="track not found")
        return {"track_id": track_id, "track": tracks_data[track_id]}

    @app.post("/tracks")
    def create_track(payload: dict = Body(...)) -> dict:
        track_id = _normalize_track_id(payload)
        name = payload.get("name")
        lap = payload.get("lap")
        time_val = payload.get("time")
        segments = _validate_segments(payload.get("segments", []))
        with metadata_lock:
            data = _read_metadata()
            tracks_data = data.get("tracks", {})
            if track_id in tracks_data:
                raise HTTPException(status_code=409, detail="track already exists")
            tracks_data[track_id] = {
                "name": name,
                "lap": lap,
                "time": time_val,
                "segments": segments,
            }
            data["tracks"] = tracks_data
            _write_metadata(data)
        return {"track_id": track_id, "track": tracks_data[track_id]}

    @app.put("/tracks/{track_id}")
    def update_track(track_id: str, payload: dict = Body(...)) -> dict:
        with metadata_lock:
            data = _read_metadata()
            tracks_data = data.get("tracks", {})
            if track_id not in tracks_data:
                raise HTTPException(status_code=404, detail="track not found")
            track = tracks_data[track_id]
            if "name" in payload:
                track["name"] = payload.get("name")
            if "lap" in payload:
                track["lap"] = payload.get("lap")
            if "time" in payload:
                track["time"] = payload.get("time")
            if "segments" in payload:
                track["segments"] = _validate_segments(payload.get("segments"))
            tracks_data[track_id] = track
            data["tracks"] = tracks_data
            _write_metadata(data)
        return {"track_id": track_id, "track": track}

    @app.delete("/tracks/{track_id}")
    def delete_track(track_id: str) -> dict:
        with metadata_lock:
            data = _read_metadata()
            tracks_data = data.get("tracks", {})
            if track_id not in tracks_data:
                raise HTTPException(status_code=404, detail="track not found")
            removed = tracks_data.pop(track_id)
            data["tracks"] = tracks_data
            _write_metadata(data)
        return {"track_id": track_id, "removed": removed}

    @app.post("/tracks/{track_id}/segments")
    def update_track_segments(track_id: str, payload: dict = Body(...)) -> dict:
        segments = _validate_segments(payload.get("segments"))
        with metadata_lock:
            data = _read_metadata()
            tracks_data = data.get("tracks", {})
            if track_id not in tracks_data:
                raise HTTPException(status_code=404, detail="track not found")
            tracks_data[track_id]["segments"] = segments
            data["tracks"] = tracks_data
            _write_metadata(data)
        return {"track_id": track_id, "segments": segments}

    @app.get("/tracks/optimal-line")
    def optimal_line(
        track_id: str = Query(...),
        step: int = Query(1, ge=1, le=100),
        limit: int | None = Query(None, ge=1, le=MAX_SAMPLE_LIMIT),
    ) -> dict:
        data = _read_metadata()
        tracks_data = data.get("tracks", {})
        if track_id not in tracks_data:
            raise HTTPException(status_code=404, detail="track not found")
        lap_id = tracks_data[track_id].get("lap")
        if lap_id is None:
            raise HTTPException(status_code=404, detail="optimal lap not set for track")
        items = _fetch_lap_positions(
            int(lap_id),
            step=step,
            limit=limit,
            track_id=track_id,
        )
        return {"track_id": track_id, "lap_id": lap_id, "items": items}

    @app.get("/laps")
    def laps(track_id: str | None = Query(None)) -> dict:
        effective_track_id = _resolve_track_filter(track_id)
        conn = _open_db()
        try:
            columns = _get_columns(conn)
            has_track_id = "track_id" in columns
            selected_track_col = "track_id" if has_track_id else "NULL AS track_id"
            where_clause = ""
            params: list[object] = []
            if effective_track_id is not None and has_track_id:
                where_clause = "WHERE track_id = ?"
                params.append(effective_track_id)
            group_by = "lap_id, track_id" if has_track_id else "lap_id"
            query = (
                "SELECT "
                "lap_id, "
                f"{selected_track_col}, "
                "MIN(lap_number) AS lap_number_min, "
                "MAX(lap_number) AS lap_number_max, "
                "MIN(timestamp_utc) AS start_utc, "
                "MAX(timestamp_utc) AS end_utc, "
                "COUNT(*) AS samples "
                "FROM telemetry_samples "
                f"{where_clause} "
                f"GROUP BY {group_by} "
                "ORDER BY lap_id"
            )
            rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail=f"DB error: {exc}")
        finally:
            conn.close()
        laps_out = []
        for row in rows:
            laps_out.append(
                {
                    "lap_id": row["lap_id"],
                    "track_id": row["track_id"],
                    "lap_number_min": row["lap_number_min"],
                    "lap_number_max": row["lap_number_max"],
                    "start_utc": row["start_utc"],
                    "end_utc": row["end_utc"],
                    "samples": row["samples"],
                }
            )
        return {"laps": laps_out}

    @app.get("/laps/{lap_id}/samples")
    def lap_samples(
        lap_id: int,
        limit: int = Query(1000, ge=1, le=MAX_SAMPLE_LIMIT),
        offset: int = Query(0, ge=0),
        order: str = Query("asc"),
        fields: str | None = None,
    ) -> dict:
        order_norm = order.strip().lower()
        if order_norm not in ("asc", "desc"):
            raise HTTPException(status_code=400, detail="order must be asc or desc")
        conn = _open_db()
        try:
            columns = _get_columns(conn)
            if not columns:
                return {"lap_id": lap_id, "fields": [], "samples": []}
            selected = _select_fields(columns, fields)
            query = (
                f"SELECT {', '.join(selected)} "
                "FROM telemetry_samples "
                "WHERE lap_id = ? "
                f"ORDER BY id {order_norm} "
                "LIMIT ? OFFSET ?"
            )
            rows = conn.execute(query, (lap_id, limit, offset)).fetchall()
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail=f"DB error: {exc}")
        finally:
            conn.close()
        return {
            "lap_id": lap_id,
            "fields": selected,
            "samples": [dict(row) for row in rows],
        }

    @app.get("/laps/{lap_id}/path")
    def lap_path(
        lap_id: int,
        step: int = Query(1, ge=1, le=100),
        limit: int | None = Query(None, ge=1, le=MAX_SAMPLE_LIMIT),
    ) -> dict:
        items = _fetch_lap_positions(lap_id, step=step, limit=limit)
        return {"lap_id": lap_id, "items": items}

    @app.delete("/laps/{lap_id}")
    def delete_lap(lap_id: int) -> dict:
        if getattr(listener, "_race_active", False) and getattr(listener, "lap_id", None) == lap_id:
            raise HTTPException(
                status_code=409,
                detail="cannot delete active lap while race is running",
            )

        conn = _open_db()
        deleted_samples = 0
        track_id = None
        try:
            columns = _get_columns(conn)
            if "track_id" in columns:
                row = conn.execute(
                    """
                    SELECT track_id
                    FROM telemetry_samples
                    WHERE lap_id = ? AND track_id IS NOT NULL AND track_id != ''
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (lap_id,),
                ).fetchone()
                if row is not None:
                    track_id = row["track_id"]

            cur = conn.execute("DELETE FROM telemetry_samples WHERE lap_id = ?", (lap_id,))
            conn.commit()
            deleted_samples = int(cur.rowcount or 0)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail=f"DB error: {exc}")
        finally:
            conn.close()

        if deleted_samples <= 0:
            raise HTTPException(status_code=404, detail="lap not found")

        if hasattr(listener, "_log_event"):
            try:
                listener._log_event(
                    "lap_deleted",
                    {
                        "lap_id": lap_id,
                        "track_id": track_id,
                        "samples": deleted_samples,
                    },
                )
            except Exception:
                pass

        return {
            "lap_id": lap_id,
            "track_id": track_id,
            "deleted_samples": deleted_samples,
        }

    @app.get("/feedback")
    def feedback(
        lap: int | None = None,
        segment: int | None = None,
        since_id: int | None = None,
        limit: int = Query(100, ge=1, le=1000),
    ) -> dict:
        items = listener.get_feedback(
            lap=lap,
            segment=segment,
            since_id=since_id,
            limit=limit,
        )
        return {"items": items}

    @app.get("/feedback/latest")
    def feedback_latest() -> dict:
        items = listener.get_feedback(limit=1)
        return items[0] if items else {}

    @app.post("/feedback/{feedback_id}/ack")
    def feedback_ack(feedback_id: int) -> dict:
        ok = listener.ack_feedback(feedback_id)
        if not ok:
            raise HTTPException(status_code=404, detail="feedback not found")
        return {"id": feedback_id, "acked": True}

    @app.get("/events")
    def events(
        since_id: int | None = None,
        limit: int = Query(100, ge=1, le=1000),
    ) -> dict:
        items = listener.get_events(since_id=since_id, limit=limit)
        return {"items": items}

    @app.get("/live/positions")
    def live_positions(
        since_id: int | None = None,
        limit: int = Query(1000, ge=1, le=5000),
    ) -> dict:
        items = listener.get_live_positions(limit=limit, since_id=since_id)
        return {"items": items}

    @app.get("/live/path")
    def live_path(limit: int = Query(5000, ge=1, le=20000)) -> dict:
        items = listener.get_current_lap_positions(limit=limit)
        return {"items": items}

    @app.get("/compare")
    def compare_laps(
        lap_a: int = Query(...),
        lap_b: int = Query(...),
        track_id: str | None = Query(None),
        step: int = Query(1, ge=1, le=100),
        limit: int | None = Query(None, ge=1, le=MAX_SAMPLE_LIMIT),
    ) -> dict:
        requested_track_id = _resolve_track_filter(track_id)
        conn = _open_db()
        try:
            lap_a_track_id = _fetch_lap_track_id(conn, lap_a)
            lap_b_track_id = _fetch_lap_track_id(conn, lap_b)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail=f"DB error: {exc}")
        finally:
            conn.close()

        if lap_a_track_id is None or lap_b_track_id is None:
            raise HTTPException(
                status_code=400,
                detail="lap track assignment missing; record laps again with track mapping",
            )
        if lap_a_track_id != lap_b_track_id:
            raise HTTPException(
                status_code=400,
                detail="laps from different tracks cannot be compared",
            )
        if requested_track_id is not None and requested_track_id != lap_a_track_id:
            raise HTTPException(
                status_code=400,
                detail="laps do not belong to requested track",
            )
        compare_track_id = lap_a_track_id

        items_a = _fetch_lap_positions(lap_a, step=step, limit=limit, track_id=compare_track_id)
        items_b = _fetch_lap_positions(lap_b, step=step, limit=limit, track_id=compare_track_id)
        return {
            "track_id": compare_track_id,
            "lap_a": {"lap_id": lap_a, "track_id": lap_a_track_id, "items": items_a},
            "lap_b": {"lap_id": lap_b, "track_id": lap_b_track_id, "items": items_b},
        }

    @app.get("/export/lap/{lap_id}")
    def export_lap(
        lap_id: int,
        format: str = Query("json"),
        fields: str | None = None,
        step: int = Query(1, ge=1, le=100),
        limit: int | None = Query(None, ge=1, le=MAX_SAMPLE_LIMIT),
    ):
        format_norm = format.strip().lower()
        conn = _open_db()
        try:
            columns = _get_columns(conn)
            if not columns:
                return {"lap_id": lap_id, "items": []}
            selected = _select_fields(columns, fields)
        finally:
            conn.close()
        items = _fetch_lap_positions(lap_id, step=step, limit=limit, fields=selected)
        if format_norm == "json":
            return {"lap_id": lap_id, "fields": selected, "items": items}
        if format_norm == "csv":
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=selected)
            writer.writeheader()
            for row in items:
                writer.writerow(row)
            return PlainTextResponse(buf.getvalue(), media_type="text/csv")
        if format_norm in ("md", "markdown"):
            header = "| " + " | ".join(selected) + " |"
            sep = "| " + " | ".join(["---"] * len(selected)) + " |"
            lines = [header, sep]
            for row in items:
                lines.append("| " + " | ".join(str(row.get(col, "")) for col in selected) + " |")
            return PlainTextResponse("\n".join(lines), media_type="text/markdown")
        raise HTTPException(status_code=400, detail="format must be json, csv, or md")

    @app.get("/diagnostics")
    def diagnostics() -> dict:
        return {
            "packet_rate": listener.packet_rate,
            "queues": {
                "model": listener.model_queue.qsize(),
                "voice": listener.voice_queue.qsize(),
                "tts": listener.tts_queue.qsize(),
                "telemetry": listener.telemetry_queue.qsize(),
            },
            "workers": listener._worker_last_active,
        }

    @app.post("/session/pause")
    def session_pause() -> dict:
        listener.set_paused(True)
        return {"paused": True}

    @app.post("/session/resume")
    def session_resume() -> dict:
        listener.set_paused(False)
        return {"paused": False}

    @app.post("/session/reset")
    def session_reset() -> dict:
        listener.reset_session()
        return {"reset": True}

    @app.put("/model")
    def model_control(payload: dict = Body(...)) -> dict:
        if "enabled" in payload:
            listener.set_model_enabled(bool(payload.get("enabled")))
        if "model_name" in payload and payload.get("model_name"):
            listener.set_model_name(str(payload.get("model_name")), warmup=bool(payload.get("warmup", False)))
        if "system_prompt" in payload or "user_prompt" in payload:
            listener.set_prompts(payload.get("system_prompt"), payload.get("user_prompt"))
        return {
            "enabled": listener.model_enabled,
            "model_name": listener.model.MODEL_NAME,
        }

    @app.put("/voice")
    def voice_control(payload: dict = Body(...)) -> dict:
        if "enabled" in payload:
            listener.set_voice_enabled(bool(payload.get("enabled")))
        return {"enabled": listener.voice_enabled}

    @app.put("/tts")
    def tts_control(payload: dict = Body(...)) -> dict:
        if "enabled" in payload:
            listener.set_tts_enabled(bool(payload.get("enabled")))
        removed = 0
        if payload.get("clear_cache"):
            removed = listener.clear_tts_cache()
        return {"enabled": listener.tts_enabled, "cleared": removed}

    @app.get("/tts/audio")
    def tts_audio(
        lap: int = Query(..., ge=0),
        segment: int = Query(..., ge=0),
    ):
        wav_path = listener.get_tts_audio_path(
            lap=lap,
            segment=segment,
            synthesize_if_missing=True,
        )
        if wav_path is None or not Path(wav_path).exists():
            raise HTTPException(status_code=404, detail="tts audio not available")
        media_type = mimetypes.guess_type(Path(wav_path).name)[0] or "application/octet-stream"
        return FileResponse(
            path=Path(wav_path).as_posix(),
            media_type=media_type,
            filename=Path(wav_path).name,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/tts/preview")
    def tts_preview(
        text: str = Query(..., min_length=1, max_length=300),
    ):
        wav_path = listener.synthesize_preview_tts(text)
        if wav_path is None or not Path(wav_path).exists():
            raise HTTPException(status_code=500, detail="tts preview generation failed")
        media_type = mimetypes.guess_type(Path(wav_path).name)[0] or "application/octet-stream"
        return FileResponse(
            path=Path(wav_path).as_posix(),
            media_type=media_type,
            filename=Path(wav_path).name,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/stream/telemetry")
    def stream_telemetry():
        def gen():
            last_ping = time.time()
            while True:
                try:
                    item = listener.telemetry_queue.get(timeout=1.0)
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    if time.time() - last_ping > 15:
                        yield "event: ping\ndata: {}\n\n"
                        last_ping = time.time()
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/stream/events")
    def stream_events():
        def gen():
            last_id = 0
            last_ping = time.time()
            while True:
                items = listener.get_events(since_id=last_id, limit=100)
                if items:
                    last_id = items[-1]["id"]
                    for item in items:
                        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                else:
                    time.sleep(0.5)
                if time.time() - last_ping > 15:
                    yield "event: ping\ndata: {}\n\n"
                    last_ping = time.time()
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/stream/feedback")
    def stream_feedback():
        def gen():
            last_id = 0
            last_ping = time.time()
            while True:
                items = listener.get_feedback(since_id=last_id, limit=100)
                if items:
                    last_id = items[-1]["id"]
                    for item in items:
                        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                else:
                    time.sleep(0.5)
                if time.time() - last_ping > 15:
                    yield "event: ping\ndata: {}\n\n"
                    last_ping = time.time()
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def start_api(listener: Listener, host: str = "127.0.0.1", port: int = 8000) -> threading.Thread:
    import uvicorn

    app = create_app(listener)

    def _run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=_run, name="APIServer", daemon=True)
    thread.start()
    return thread
