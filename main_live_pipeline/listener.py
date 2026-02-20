import datetime
import os
import socket
import sqlite3
import struct
import time
import ast
import hashlib
import numpy as np
import threading
import queue
import json
import re
from collections import deque

from typing import Iterable
from pathlib import Path
from preprocessing import Preprocessing
from model import Model
from voice import Voice


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


class Listener():
    def __init__(self, DB_FILE, debug=False):
        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT = 5300
        self.DB_FILE = resolve_repo_path(DB_FILE)
        self.MAX_PACKETS_PER_SECOND = 10
        self.MIN_SAVE_INTERVAL = 1.0 / self.MAX_PACKETS_PER_SECOND
        self.FORZA_FIELDS = self.load_tuples(BASE_DIR / "forza_fields.txt")
        self.SCHEMAS = [("fh5_dash_324", self.FORZA_FIELDS, struct.calcsize("<" + "".join(fmt for _, fmt in self.FORZA_FIELDS)))]
        self.BASE_STRUCT_FORMAT = "<" + "".join(fmt for _, fmt in self.FORZA_FIELDS)
        self.distance = np.inf
        self.threshold = 15
        self.preplay_distance = 20.0
        self.segment = 0
        self.in_zone = False
        self.segments = [[610, 2485], [635, 2780], [525, 2720], [880, 2790]]
        self.track_name = None
        self.lap = 0
        self.lap_id = None
        self.model = Model()
        self.voice = Voice()
        self.LLM_CONTEXT_PATH = resolve_repo_path("prompts/llm_context.txt")
        self._llm_context_lock = threading.Lock()
        self._llm_context_generation = 0
        self.driver_name = self._normalize_driver_name(os.getenv("FH5_DRIVER_NAME", "Etienne"))
        self.model_queue = queue.Queue()
        self.model_worker = threading.Thread(target=self._model_worker, name="ModelWorker", daemon=True)
        self.model_worker.start()
        self.voice_queue = queue.Queue()
        self.voice_worker = threading.Thread(target=self._voice_worker, name="VoiceWorker", daemon=True)
        self.voice_worker.start()
        self.tts_queue = queue.Queue()
        self.tts_worker = threading.Thread(target=self._tts_worker, name="TTSWorker", daemon=True)
        self.tts_worker.start()
        self._responses_lock = threading.Lock()
        self._model_responses = {}
        self._spoken_for_lap = set()
        self._tts_lock = threading.Lock()
        self._tts_cache = {}
        self._tts_dir = (REPO_ROOT / "data" / "tts_cache")
        self.DEBUG = debug
        self._seen_packet_sizes = set()
        self._last_no_data_log = 0.0
        self._packet_count = 0
        self.track_id = None
        self._race_active = False
        self.optimal_lap_id = None
        self.last_packet_utc = None
        self._response_log_lock = threading.Lock()
        self._response_log = []
        self._response_seq = 0
        self._response_log_max = 1000
        self.paused = False
        self.model_enabled = True
        # Browser clients handle playback; backend voice playback can still be
        # enabled via /voice when explicitly requested.
        self.voice_enabled = False
        self.tts_enabled = True
        self.last_position = None
        self.last_speed = None
        self.last_yaw = None
        self.last_lap_time = None
        self.best_lap_time = None
        self.current_distance_to_segment = None
        self.current_segment_target = None
        self.packet_rate = 0.0
        self._packet_times = deque(maxlen=200)
        self._telemetry_lock = threading.Lock()
        self._telemetry_buffer = deque(maxlen=5000)
        self._telemetry_seq = 0
        self.telemetry_queue = queue.Queue(maxsize=2000)
        self._events_lock = threading.Lock()
        self._events = []
        self._events_seq = 0
        self._events_max = 1000
        self._worker_last_active = {"model": None, "voice": None, "tts": None}
        self._lap_positions_lock = threading.Lock()
        self._current_lap_positions = []
        self.INCOMPLETE_LAP_SEGMENT_TOLERANCE = 1
        self.INCOMPLETE_LAP_MIN_SAMPLES = 80
        self._lap_sample_count = 0
        self._lap_max_segment = 0
        self._last_race_time = None
        self._last_race_lap_number = None
        
        
    def run(self):
        conn = self.init_db(self.DB_FILE)
        sock = self.create_socket(self.UDP_IP, self.UDP_PORT)
        print(f"Lausche auf UDP {self.UDP_IP}:{self.UDP_PORT} ...")
        last_saved = 0.0
        sock.settimeout(2.0)

        try:
            while True:
                try:
                    data, addr = sock.recvfrom(512)
                except socket.timeout:
                    if self.DEBUG and (time.monotonic() - self._last_no_data_log) > 2.0:
                        print("Kein UDP-Datenempfang in den letzten 2 Sekunden.")
                        self._last_no_data_log = time.monotonic()
                    continue
                if self.DEBUG:
                    self._packet_count += 1
                    if self._packet_count <= 5:
                        print(f"UDP-Paket #{self._packet_count}: {len(data)} Bytes von {addr}")
                parsed_result = self.parse_telemetry(data)
                if parsed_result is None:
                    continue
                schema_name, parsed = parsed_result
                in_race = parsed.get("engine_max_rpm", 0) != 0 and (
                    parsed.get("distance_traveled", 0) != 0
                    or parsed.get("current_race_time", 0) > 0
                    or parsed.get("lap_number", 0) > 0
                )
                self._update_race_activity_state(
                    in_race=in_race,
                    parsed=parsed,
                    conn=conn,
                )

                # User drives in open world
                if parsed.get("distance_traveled", 0) == 0 and parsed.get("engine_max_rpm", 0) != 0:
                    if self.DEBUG:
                        print(f"{parsed.get('position_x')}, {parsed.get('position_z')}")
                    self.identify_track_id([parsed.get('position_x'), parsed.get('position_z')])
                    now = time.monotonic()
                    if now - last_saved < self.MIN_SAVE_INTERVAL:
                        continue
                    last_saved = now
                    timestamp_utc = (datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z")
                    if self.paused:
                        continue
                    self._ingest_free_roam_sample(timestamp_utc, parsed)
                    continue
                
                # User is in menu
                if parsed.get("distance_traveled", 0) == 0 and parsed.get("engine_max_rpm", 0) == 0:
                    continue
                
                # User is in race
                if parsed.get("distance_traveled", 0) != 0 and parsed.get("engine_max_rpm", 0) != 0:
                    self.identify_track_id([parsed.get("position_x"), parsed.get("position_z")])
                    if not self._race_active or self.lap_id is None:
                        self._start_new_lap(parsed.get("lap_number", 0), reset_session=True)
                        self._log_event("race_start", {"lap": self.lap, "lap_id": self.lap_id})

                    self._race_active = True
                    now = time.monotonic()
                    if now - last_saved < self.MIN_SAVE_INTERVAL:
                        continue
                    last_saved = now
                    timestamp_utc = (datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z")
                    if self.paused:
                        continue
                    self.insert_sample(conn, timestamp_utc, data, schema_name, parsed)

        except KeyboardInterrupt:
            print("\nBeende Listener...")
        finally:
            conn.close()
            sock.close()

    def _current_lap_is_complete_enough(self) -> bool:
        total_segments = len(self.segments)
        if total_segments <= 0:
            required_segments = 0
        else:
            required_segments = max(1, total_segments - self.INCOMPLETE_LAP_SEGMENT_TOLERANCE)
        reached_segments = int(self._lap_max_segment)
        return (
            self._lap_sample_count >= self.INCOMPLETE_LAP_MIN_SAMPLES
            and reached_segments >= required_segments
        )

    def _is_lap_switch_completion_valid(self, next_lap: int) -> bool:
        """Validate lap completion on lap-number switch to avoid restart artifacts."""
        if self.lap_id is None:
            return False
        if next_lap <= int(self.lap):
            return False
        return self._current_lap_is_complete_enough()

    def _delete_lap_samples(self, lap_id: int, reason: str, conn: sqlite3.Connection | None = None) -> None:
        if lap_id is None:
            return
        own_conn = False
        db_conn = conn
        if db_conn is None:
            db_conn = sqlite3.connect(self.DB_FILE)
            own_conn = True
        try:
            cur = db_conn.execute("DELETE FROM telemetry_samples WHERE lap_id = ?", (lap_id,))
            db_conn.commit()
            if cur.rowcount and cur.rowcount > 0:
                self._log_event(
                    "lap_discarded",
                    {
                        "lap_id": lap_id,
                        "lap": self.lap,
                        "reason": reason,
                        "samples": self._lap_sample_count,
                        "segments_reached": int(self._lap_max_segment),
                        "segments_total": len(self.segments),
                    },
                )
        except sqlite3.Error as exc:
            if self.DEBUG:
                print(f"Lap cleanup failed for lap_id={lap_id}: {exc}")
        finally:
            if own_conn:
                db_conn.close()

    def _drop_current_lap_if_incomplete(
        self,
        reason: str,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if self.lap_id is None:
            return
        if self._current_lap_is_complete_enough():
            return
        self._delete_lap_samples(self.lap_id, reason=reason, conn=conn)

    def _end_active_race(
        self,
        reason: str = "race_end",
        conn: sqlite3.Connection | None = None,
    ) -> None:
        self._drop_current_lap_if_incomplete(reason=reason, conn=conn)
        if self._race_active:
            self._log_event("race_end", {"lap": self.lap, "lap_id": self.lap_id})
        self._race_active = False
        self._last_race_time = None
        self._last_race_lap_number = None

    def _update_race_activity_state(
        self,
        in_race: bool,
        parsed: dict,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if in_race:
            try:
                current_race_time = float(parsed.get("current_race_time", 0.0) or 0.0)
            except Exception:
                current_race_time = 0.0
            try:
                current_lap_number = int(parsed.get("lap_number", 0) or 0)
            except Exception:
                current_lap_number = 0

            if self._race_active and self._last_race_time is not None:
                race_time_rewound = current_race_time + 1.0 < self._last_race_time
                if race_time_rewound and current_lap_number <= 1:
                    # New race was started from menu; close old race context.
                    self._end_active_race(reason="race_restart", conn=conn)

            self._last_race_time = current_race_time
            self._last_race_lap_number = current_lap_number
            return
        if not self._race_active:
            return

        in_open_world = (
            parsed.get("distance_traveled", 0) == 0
            and parsed.get("engine_max_rpm", 0) != 0
        )
        if not in_open_world:
            # Menu/paused overlays should keep the active race context, regardless of duration.
            return

        self._end_active_race(reason="open_world", conn=conn)


    def identify_track_id(self, position: list[int, int]):
        metadata_path = Path("main_live_pipeline/metadata.json")
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        tracks = data.get("tracks") if isinstance(data.get("tracks"), dict) else data
        if not isinstance(tracks, dict):
            return

        best_track_id = None
        best_track_info = None
        best_distance = np.inf
        for point, track_info in tracks.items():
            if not isinstance(track_info, dict):
                continue
            try:
                point_conv = [int(i.strip()) for i in point.strip("[]").split(",")]
            except (ValueError, AttributeError):
                continue
            distance_to_point = np.sqrt((position[0] - point_conv[0])**2 + (position[1] - point_conv[1])**2)
            if distance_to_point < best_distance:
                best_distance = distance_to_point
                best_track_id = point
                best_track_info = track_info

        if best_track_id is None or best_track_info is None:
            return

        strict_lock_distance = 30.0
        relaxed_race_distance = 300.0
        should_assign = best_distance <= strict_lock_distance
        if (
            not should_assign
            and self._race_active
            and self.track_id is None
            and best_distance <= relaxed_race_distance
        ):
            should_assign = True

        if not should_assign:
            return

        previous = self.track_id
        self.track_id = best_track_id
        self.segments = best_track_info.get("segments", self.segments)
        self.track_name = best_track_info.get("name")
        self.optimal_lap_id = best_track_info.get("lap")
        if previous != self.track_id:
            self._log_event(
                "track_changed",
                {
                    "track_id": self.track_id,
                    "track_name": self.track_name,
                    "distance": float(best_distance),
                },
            )

    def update_optimal_lap_time(self, new_id: int, new_time: float) -> None:
        if self.track_id is None:
            return
        path = Path("main_live_pipeline/metadata.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        tracks = data.get("tracks") if isinstance(data.get("tracks"), dict) else data
        if not isinstance(tracks, dict) or self.track_id not in tracks:
            return
        tracks[self.track_id]["lap"] = new_id
        tracks[self.track_id]["time"] = new_time
        if isinstance(data.get("tracks"), dict):
            data["tracks"] = tracks
        else:
            data = tracks
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.optimal_lap_id = new_id


    def _start_new_lap(self, lap_number: int | None = None, reset_session: bool = False) -> None:
        self.lap_id = self._get_next_lap_id()
        self.segment = 0
        self._lap_max_segment = 0
        self._lap_sample_count = 0
        self.distance = np.inf
        self.in_zone = False
        with self._lap_positions_lock:
            self._current_lap_positions = []
        if reset_session:
            self._spoken_for_lap.clear()
            with self._responses_lock:
                self._model_responses.clear()
            self._reset_llm_context()
        if lap_number is not None and lap_number > 0:
            self.lap = lap_number
        elif reset_session:
            self.lap = 0
        self._log_event("lap_start", {"lap": self.lap, "lap_id": self.lap_id})
        if self.DEBUG:
            print(f"Neue Runde gestartet: lap_id={self.lap_id}, lap_number={self.lap}")

    def _normalize_driver_name(self, name: str | None) -> str:
        normalized = (name or "").strip()
        return normalized if normalized else "Etienne"

    def set_driver_name(self, name: str | None) -> None:
        self.driver_name = self._normalize_driver_name(name)


    def _enqueue_model_run(
        self,
        lap_id: int | None,
        lap: int,
        segment: int,
        persist_context: bool = False,
    ) -> None:
        if lap_id is None or not self.model_enabled:
            return
        with self._llm_context_lock:
            generation = self._llm_context_generation
        self.model_queue.put((lap_id, lap, segment, persist_context, generation))


    def _model_worker(self) -> None:
        while True:
            item = self.model_queue.get()
            try:
                if len(item) == 5:
                    lap_id, lap, segment, persist_context, generation = item
                else:
                    lap_id, lap, segment = item
                    persist_context = False
                    with self._llm_context_lock:
                        generation = self._llm_context_generation
                self._worker_last_active["model"] = time.time()
                preprocessing = Preprocessing(
                    self.DB_FILE,
                    lap_id,
                    self.optimal_lap_id,
                    self.segments,
                    track_id=self.track_id,
                )
                md_text = preprocessing.run()
                if not md_text or not md_text.strip():
                    print("No markdown data available; skipping model run.")
                    continue
                if persist_context:
                    self._append_round_markdown_to_llm_context(
                        lap_id=lap_id,
                        lap=lap,
                        segment=segment,
                        markdown=md_text,
                        generation=generation,
                    )
                response_text = self.model.run(lap, segment, md_text=md_text)
                self._store_model_response(lap_id, lap, segment, response_text)
            except Exception as exc:
                print(f"Model worker error: {exc}")
            finally:
                self.model_queue.task_done()

    def _reset_llm_context(self) -> None:
        with self._llm_context_lock:
            self._llm_context_generation += 1
            self.LLM_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.LLM_CONTEXT_PATH.write_text("", encoding="utf-8")

    def _append_round_markdown_to_llm_context(
        self,
        lap_id: int,
        lap: int,
        segment: int,
        markdown: str,
        generation: int,
    ) -> None:
        with self._llm_context_lock:
            if generation != self._llm_context_generation:
                return
            self.LLM_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.LLM_CONTEXT_PATH.open("a", encoding="utf-8") as f:
                f.write("\n\n[Round Markdown]\n")
                f.write(
                    f"Lap ID: {lap_id} | Lap: {lap} | Segment: {segment} | "
                    f"Track ID: {self.track_id} | Driver: {self.driver_name}\n"
                )
                f.write("```markdown\n")
                f.write(markdown.strip())
                f.write("\n```\n")

    def _enqueue_voice(self, text: str) -> None:
        if not text or not text.strip() or not self.voice_enabled:
            return
        self.voice_queue.put(("text", text.strip()))

    def _enqueue_voice_wav(self, wav_path: str | Path, key: tuple[int, int]) -> None:
        if not wav_path or not self.voice_enabled:
            return
        self.voice_queue.put(("wav", Path(wav_path), key))

    def _voice_worker(self) -> None:
        while True:
            item = self.voice_queue.get()
            try:
                self._worker_last_active["voice"] = time.time()
                kind, payload, *rest = item
                if kind == "wav":
                    wav_key = rest[0] if rest else None
                    self.voice.play_wav(payload)
                    if wav_key is not None:
                        with self._tts_lock:
                            cached = self._tts_cache.get(wav_key)
                            if cached and Path(cached) == Path(payload):
                                self._tts_cache.pop(wav_key, None)
                    try:
                        Path(payload).unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    self.voice.run(payload)
            except Exception as exc:
                print(f"Voice error: {exc}")
            finally:
                self.voice_queue.task_done()

    def _enqueue_tts(self, lap: int, segment: int, text: str) -> None:
        if not text or not text.strip() or not self.tts_enabled:
            return
        self.tts_queue.put((lap, segment, text.strip()))

    def _tts_worker(self) -> None:
        while True:
            lap, segment, text = self.tts_queue.get()
            try:
                self._worker_last_active["tts"] = time.time()
                suffix = getattr(self.voice, "audio_suffix", ".wav")
                wav_path = self._tts_dir / f"lap{lap}_seg{segment}{suffix}"
                wav_path = self.voice.synthesize_to_wav(text, wav_path)
                with self._tts_lock:
                    self._tts_cache[(lap, segment)] = wav_path
            except Exception as exc:
                print(f"TTS error: {exc}")
            finally:
                self.tts_queue.task_done()

    def _store_model_response(self, lap_id: int | None, lap: int, segment: int, response_text: str | None) -> None:
        payload = self._parse_model_payload(response_text)
        if payload is None:
            return
        self._log_model_response(lap_id, lap, segment, payload)
        message = payload.get("message") if payload.get("status") is True else None
        if not message:
            return
        message = self._normalize_feedback_language(message)
        key = (lap, segment)
        with self._responses_lock:
            self._model_responses[key] = message
        self._enqueue_tts(lap, segment, message)

    def _parse_model_payload(self, response_text: str | None) -> dict | None:
        if not response_text or not response_text.strip():
            return None
        text = response_text.strip()
        data = None
        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        candidates.extend(chunk.strip() for chunk in fenced if chunk and chunk.strip())
        brace_candidate = self._extract_first_brace_block(text)
        if brace_candidate:
            candidates.append(brace_candidate)

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)

        for candidate in unique_candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(candidate)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    data = parsed
                    break
            if isinstance(data, dict):
                break
        if isinstance(data, dict):
            status = data.get("status")
            if isinstance(status, str):
                status = status.strip().lower() in ("true", "1", "yes", "y")
            elif status is None:
                status = None
            else:
                status = bool(status)
            message = data.get("message")
            message = str(message).strip() if message is not None else None
            return {"status": status, "message": message, "raw": text, "parsed": True}
        return {"status": None, "message": None, "raw": text, "parsed": False}

    def _extract_first_brace_block(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1].strip()
        return None

    def _normalize_feedback_language(self, message: str) -> str:
        normalized = (message or "").strip()
        if not normalized:
            return normalized
        english_to_german = {
            "Good throttle use, speed carried well.": "Gute Gasannahme, die Geschwindigkeit wurde sauber mitgenommen.",
            "Clean line, no time lost here.": "Saubere Linie, hier wurde keine Zeit verloren.",
            "Brake earlier to stabilize the car.": "Bremse frueher, um das Auto zu stabilisieren.",
            "Turn in earlier for better exit.": "Lenke frueher ein, um einen besseren Kurvenausgang zu haben.",
            "Strong exit, close to optimal.": "Starker Kurvenausgang, fast optimal.",
            "Over-slowed entry cost exit speed.": "Zu langsamer Kurveneingang hat Ausgangsgeschwindigkeit gekostet.",
            "Strong exit on turn-in to maintain optimal speed.": "Starker Kurvenausgang beim Einlenken, um die optimale Geschwindigkeit zu halten.",
            "The braking in this segment could be improved to stabilize the car better.": "Das Bremsen in diesem Segment kann verbessert werden, um das Auto besser zu stabilisieren.",
            "Good throttle use, but brake earlier to stabilize the car.": "Gute Gasannahme, aber bremse frueher, um das Auto zu stabilisieren.",
            "Slow entry at the start of the segment cost exit speed.": "Der langsame Kurveneingang zu Segmentbeginn hat Ausgangsgeschwindigkeit gekostet.",
        }
        return english_to_german.get(normalized, normalized)

    def _log_model_response(self, lap_id: int | None, lap: int, segment: int, payload: dict) -> None:
        entry = {
            "lap_id": lap_id,
            "lap": lap,
            "segment": segment,
            "status": payload.get("status"),
            "message": payload.get("message"),
            "raw": payload.get("raw"),
            "parsed": payload.get("parsed"),
            "acked": False,
            "timestamp_utc": (datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"),
        }
        with self._response_log_lock:
            self._response_seq += 1
            entry["id"] = self._response_seq
            self._response_log.append(entry)
            if len(self._response_log) > self._response_log_max:
                self._response_log = self._response_log[-self._response_log_max:]

    def get_feedback(
        self,
        lap: int | None = None,
        segment: int | None = None,
        since_id: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        with self._response_log_lock:
            items = list(self._response_log)
        if since_id is not None:
            items = [item for item in items if item.get("id", 0) > since_id]
        if lap is not None:
            items = [item for item in items if item.get("lap") == lap]
        if segment is not None:
            items = [item for item in items if item.get("segment") == segment]
        if limit and limit > 0:
            items = items[-limit:]
        return items

    def ack_feedback(self, feedback_id: int) -> bool:
        with self._response_log_lock:
            for item in self._response_log:
                if item.get("id") == feedback_id:
                    item["acked"] = True
                    return True
        return False

    def get_events(self, since_id: int | None = None, limit: int = 100) -> list[dict]:
        with self._events_lock:
            items = list(self._events)
        if since_id is not None:
            items = [item for item in items if item.get("id", 0) > since_id]
        if limit and limit > 0:
            items = items[-limit:]
        return items

    def get_live_positions(self, limit: int = 1000, since_id: int | None = None) -> list[dict]:
        with self._telemetry_lock:
            items = list(self._telemetry_buffer)
        if since_id is not None:
            items = [item for item in items if item.get("id", 0) > since_id]
        if limit and limit > 0:
            items = items[-limit:]
        return items

    def get_current_lap_positions(self, limit: int = 5000) -> list[dict]:
        with self._lap_positions_lock:
            items = list(self._current_lap_positions)
        if limit and limit > 0:
            items = items[-limit:]
        return items

    def clear_tts_cache(self) -> int:
        removed = 0
        with self._tts_lock:
            cached_paths = list(self._tts_cache.values())
            self._tts_cache.clear()
        for path in cached_paths:
            try:
                Path(path).unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
        return removed

    def get_tts_audio_path(
        self,
        lap: int,
        segment: int,
        synthesize_if_missing: bool = True,
    ) -> Path | None:
        key = (int(lap), int(segment))
        with self._tts_lock:
            cached = self._tts_cache.get(key)
        if cached is not None:
            path = Path(cached)
            if path.exists():
                return path

        if not synthesize_if_missing or not self.tts_enabled:
            return None

        with self._responses_lock:
            message = self._model_responses.get(key)
        if not message:
            return None

        try:
            suffix = getattr(self.voice, "audio_suffix", ".wav")
            out_path = self._tts_dir / f"lap{key[0]}_seg{key[1]}{suffix}"
            out_path = self.voice.synthesize_to_wav(message, out_path)
            with self._tts_lock:
                self._tts_cache[key] = out_path
            return Path(out_path)
        except Exception:
            return None

    def synthesize_preview_tts(self, text: str) -> Path | None:
        text = (text or "").strip()
        if not text:
            return None
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        suffix = getattr(self.voice, "audio_suffix", ".wav")
        out_path = self._tts_dir / f"preview_{digest}{suffix}"
        if out_path.exists():
            return out_path
        try:
            return Path(self.voice.synthesize_to_wav(text, out_path))
        except Exception:
            return None

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def set_model_enabled(self, enabled: bool) -> None:
        self.model_enabled = bool(enabled)

    def set_voice_enabled(self, enabled: bool) -> None:
        self.voice_enabled = bool(enabled)

    def set_tts_enabled(self, enabled: bool) -> None:
        self.tts_enabled = bool(enabled)

    def set_model_name(self, model_name: str, warmup: bool = False) -> None:
        if not model_name:
            return
        self.model = Model()
        self.model.MODEL_NAME = model_name
        if warmup:
            self.model.warmup_model()

    def set_prompts(self, system_prompt: str | None = None, user_prompt: str | None = None) -> None:
        if system_prompt is not None:
            self.model.SYSTEM_PROMPT = system_prompt
        if user_prompt is not None:
            self.model.USER_PROMPT = user_prompt

    def reset_session(self) -> None:
        self._drop_current_lap_if_incomplete(reason="session_reset", conn=None)
        self._reset_llm_context()
        self._race_active = False
        self._last_race_time = None
        self._last_race_lap_number = None
        self.lap = 0
        self.lap_id = None
        self.segment = 0
        self._lap_max_segment = 0
        self._lap_sample_count = 0
        self.distance = np.inf
        self.in_zone = False
        with self._responses_lock:
            self._model_responses.clear()
        with self._lap_positions_lock:
            self._current_lap_positions = []
        self._log_event("session_reset", {})

    def get_status(self) -> dict:
        return {
            "race_active": self._race_active,
            "track_id": self.track_id,
            "track_name": self.track_name,
            "lap": self.lap,
            "lap_id": self.lap_id,
            "segment": self.segment,
            "segments": self.segments,
            "optimal_lap_id": self.optimal_lap_id,
            "udp_ip": self.UDP_IP,
            "udp_port": self.UDP_PORT,
            "db_file": str(self.DB_FILE),
            "last_packet_utc": self.last_packet_utc,
            "paused": self.paused,
            "model_enabled": self.model_enabled,
            "voice_enabled": self.voice_enabled,
            "tts_enabled": self.tts_enabled,
            "last_position": self.last_position,
            "last_speed": self.last_speed,
            "last_yaw": self.last_yaw,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "driver_name": self.driver_name,
            "current_distance_to_segment": self.current_distance_to_segment,
            "current_segment_target": self.current_segment_target,
            "packet_rate": self.packet_rate,
            "worker_last_active": self._worker_last_active,
        }

    def _speak_previous_lap_response(self, current_lap: int, segment: int) -> None:
        if current_lap <= 0:
            return
        previous_key = (current_lap - 1, segment)
        with self._tts_lock:
            wav_path = self._tts_cache.get(previous_key)
        if wav_path is None:
            return
        spoken_key = (current_lap, segment)
        if spoken_key in self._spoken_for_lap:
            return
        self._spoken_for_lap.add(spoken_key)
        self._enqueue_voice_wav(wav_path, previous_key)

    def load_tuples(self, path: str | Path):
        return ast.literal_eval("[" + Path(path).read_text(encoding="utf-8").strip().strip().rstrip(",") + "]")

    def create_socket(self, ip: str, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        return sock

    def _get_next_lap_id(self) -> int:
        if not self.DB_FILE.exists():
            return 0
        try:
            conn = sqlite3.connect(self.DB_FILE)
            try:
                row = conn.execute("SELECT MAX(lap_id) FROM telemetry_samples").fetchone()
                max_id = row[0] if row and row[0] is not None else -1
                return int(max_id) + 1
            except sqlite3.Error:
                return 0
            finally:
                conn.close()
        except Exception:
            return 0


    def ensure_columns(self, conn: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, sql_type in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
        conn.commit()


    def init_db(self, db_path: str) -> sqlite3.Connection:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)

        db_types = {"f": "REAL", "i": "INTEGER", "I": "INTEGER", "H": "INTEGER", "B": "INTEGER"}
        column_defs = [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "lap_id INTEGER NOT NULL",
            "track_id TEXT",
            "driver_name TEXT",
            "timestamp_utc TEXT NOT NULL",
            "packet_length INTEGER NOT NULL",
            "packet_schema TEXT NOT NULL",
            "raw BLOB NOT NULL",
        ]
        for name, fmt in self.FORZA_FIELDS:
            column_defs.append(f"{name} {db_types[fmt]} NOT NULL")

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS telemetry_samples (
                {", ".join(column_defs)}
            );
            """
        )

        self.ensure_columns(
            conn,
            "telemetry_samples",
            [
                ("lap_id", "INTEGER"),
                ("track_id", "TEXT"),
                ("driver_name", "TEXT"),
                ("timestamp_utc", "TEXT"),
                ("packet_length", "INTEGER"),
                ("packet_schema", "TEXT"),
                ("raw", "BLOB")
            ]
            + [(name, db_types[fmt]) for name, fmt in self.FORZA_FIELDS]
        )
        # Existing laps in historical sessions belong to Etienne by default.
        conn.execute(
            """
            UPDATE telemetry_samples
            SET driver_name = 'Etienne'
            WHERE driver_name IS NULL OR TRIM(driver_name) = ''
            """
        )
        conn.commit()
        return conn


    def parse_telemetry(self, data: bytes) -> tuple[str, dict] | None:
        packet_size = len(data)
        for schema_name, fields, size in self.SCHEMAS:
            if packet_size != size:
                continue
            fmt = "<" + "".join(fmt for _, fmt in fields)
            values = struct.unpack_from(fmt, data, 0)
            return schema_name, {name: value for (name, _), value in zip(fields, values)}
        if self.DEBUG and packet_size not in self._seen_packet_sizes:
            expected_sizes = [size for _, _, size in self.SCHEMAS]
            print(f"Unerwartete Paketgröße {packet_size} Bytes. Erwartet: {expected_sizes}.")
            self._seen_packet_sizes.add(packet_size)
        return None


    def _update_runtime_telemetry(self, timestamp_utc: str, parsed: dict) -> dict:
        self.last_packet_utc = timestamp_utc
        self._packet_times.append(time.monotonic())
        while self._packet_times and (time.monotonic() - self._packet_times[0]) > 1.0:
            self._packet_times.popleft()
        self.packet_rate = float(len(self._packet_times))

        x = parsed.get("position_x", 0.0)
        z = parsed.get("position_z", 0.0)
        self.last_position = [x, z]
        self.last_speed = parsed.get("speed", None)
        self.last_yaw = parsed.get("yaw", None)

        current_target = self.segments[self.segment] if self.segment < len(self.segments) else None
        self.current_segment_target = current_target
        if current_target is not None:
            try:
                self.current_distance_to_segment = float(np.sqrt((x-current_target[0])**2 + (z-current_target[1])**2))
            except Exception:
                self.current_distance_to_segment = None
        else:
            self.current_distance_to_segment = None

        return {
            "lap_id": self.lap_id,
            "lap": self.lap,
            "segment": self.segment,
            "track_id": self.track_id,
            "driver_name": self.driver_name,
            "position_x": x,
            "position_z": z,
            "speed": self.last_speed,
            "yaw": self.last_yaw,
        }

    def _ingest_free_roam_sample(self, timestamp_utc: str, parsed: dict) -> None:
        self.lap = int(parsed.get("lap_number", 0) or 0)
        payload = self._update_runtime_telemetry(timestamp_utc, parsed)
        self._log_telemetry(timestamp_utc, payload)

    def insert_sample(
        self, conn: sqlite3.Connection, timestamp_utc: str, raw: bytes, schema_name: str, parsed: dict
    ) -> None:
        columns = ["lap_id", "track_id", "driver_name", "timestamp_utc", "packet_length", "packet_schema", "raw"] + [name for name, _ in self.FORZA_FIELDS]
        placeholders = ", ".join(["?"] * len(columns))
        values = [
            self.lap_id,
            self.track_id,
            self.driver_name,
            timestamp_utc,
            len(raw),
            schema_name,
            sqlite3.Binary(raw)]
        try:
            lap = int(parsed.get("lap_number", 0) or 0)
        except Exception:
            lap = 0
        payload = self._update_runtime_telemetry(timestamp_utc, parsed)
        x = payload["position_x"]
        z = payload["position_z"]
        self._log_telemetry(timestamp_utc, payload)
        with self._lap_positions_lock:
            self._current_lap_positions.append(
                {
                    "timestamp_utc": timestamp_utc,
                    "position_x": payload["position_x"],
                    "position_z": payload["position_z"],
                    "speed": payload["speed"],
                    "yaw": payload["yaw"],
                    "segment": self.segment,
                }
            )
        self._lap_sample_count += 1
        self._lap_max_segment = max(self._lap_max_segment, int(self.segment))

        for name, _ in self.FORZA_FIELDS:
            values.append(parsed.get(name, 0))
        
        if self.DEBUG:
            print((x, z))
        if self.segment < len(self.segments):
            distance = np.sqrt((x-self.segments[self.segment][0])**2 + (z-self.segments[self.segment][1])**2)
            if distance <= self.preplay_distance:
                self._speak_previous_lap_response(self.lap, self.segment)
            if distance <= self.threshold:
                if distance <= self.distance:
                    self.distance = distance
                    self.in_zone = True
                else:
                    print(f'Start processing {self.segment}')
                    self._speak_previous_lap_response(self.lap, self.segment)
                    print('Queued model answer ...')
                    self._enqueue_model_run(
                        self.lap_id,
                        self.lap,
                        self.segment,
                        persist_context=False,
                    )
                    self._log_event("segment_complete", {"lap": self.lap, "lap_id": self.lap_id, "segment": self.segment})
                    self.segment += 1
                    self._lap_max_segment = max(self._lap_max_segment, int(self.segment))
                    self.distance = np.inf
                    self.in_zone = False
        
        if lap != self.lap:
            try:
                last_lap_time = float(parsed.get("last_lap", 0) or 0.0)
            except Exception:
                last_lap_time = 0.0
            lap_completed = last_lap_time > 0
            discard_reason = "lap_switch_without_completion"
            if lap_completed and not self._is_lap_switch_completion_valid(lap):
                lap_completed = False
                discard_reason = "lap_switch_invalid_completion"
            if lap_completed:
                self.last_lap_time = last_lap_time
                optimal_lap_time = None
                if self.track_id is not None:
                    metadata_path = Path("main_live_pipeline/metadata.json")
                    data = json.loads(metadata_path.read_text(encoding="utf-8"))
                    tracks = data.get("tracks") if isinstance(data.get("tracks"), dict) else data
                    if isinstance(tracks, dict) and self.track_id in tracks:
                        optimal_lap_time = tracks[self.track_id].get("time")
                if optimal_lap_time is None or last_lap_time < optimal_lap_time:
                    self.update_optimal_lap_time(self.lap_id, last_lap_time)
                    if self.DEBUG:
                        print(f"New lap record: {last_lap_time}")
                    self.best_lap_time = last_lap_time
                    self._log_event("lap_record", {"lap_id": self.lap_id, "lap": self.lap, "time": last_lap_time})
                print(f'Start processing {self.segment}')
                print('Queued model answer ...')
                self._enqueue_model_run(
                    self.lap_id,
                    self.lap,
                    self.segment,
                    persist_context=True,
                )
                self._log_event("lap_complete", {"lap_id": self.lap_id, "lap": self.lap, "time": last_lap_time})
            else:
                self._delete_lap_samples(
                    self.lap_id,
                    reason=discard_reason,
                    conn=conn,
                )
            self.lap = lap
            self._start_new_lap(lap, reset_session=False)
            values[0] = self.lap_id
            values[1] = self.track_id
            if lap_completed:
                self._speak_previous_lap_response(self.lap, self.segment)

        conn.execute(f"INSERT INTO telemetry_samples ({', '.join(columns)}) VALUES ({placeholders})",values,)
        conn.commit()

    def _log_telemetry(self, timestamp_utc: str, payload: dict) -> None:
        entry = {
            "timestamp_utc": timestamp_utc,
            "data": payload,
        }
        with self._telemetry_lock:
            self._telemetry_seq += 1
            entry["id"] = self._telemetry_seq
            self._telemetry_buffer.append(entry)
        try:
            self.telemetry_queue.put_nowait(entry)
        except queue.Full:
            pass

    def _log_event(self, event_type: str, data: dict) -> None:
        entry = {
            "type": event_type,
            "data": data,
            "timestamp_utc": (datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"),
        }
        with self._events_lock:
            self._events_seq += 1
            entry["id"] = self._events_seq
            self._events.append(entry)
            if len(self._events) > self._events_max:
                self._events = self._events[-self._events_max:]
