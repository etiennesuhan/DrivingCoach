import datetime
import socket
import sqlite3
import struct
import time
import ast
import numpy as np
import threading
import queue

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
    def __init__(self, DB_FILE):
        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT = 5300
        self.DB_FILE = resolve_repo_path(DB_FILE)
        self.MAX_PACKETS_PER_SECOND = 10
        self.MIN_SAVE_INTERVAL = 1.0 / self.MAX_PACKETS_PER_SECOND
        self.FORZA_FIELDS = self.load_tuples(BASE_DIR / "forza_fields.txt")
        self.SCHEMAS = SCHEMAS = [("fh5_dash_324", self.FORZA_FIELDS, struct.calcsize("<" + "".join(fmt for _, fmt in self.FORZA_FIELDS)))]
        self.BASE_STRUCT_FORMAT = "<" + "".join(fmt for _, fmt in self.FORZA_FIELDS)
        self.distance = np.inf
        self.threshold = 15
        self.segment = 0
        self.in_zone = False
        self.TRACK_2_SEGMENT_POINTS = [(610, 2485), (635, 2780), (525, 2720), (880, 2790)]
        self.lap = 0
        self.preprocessing = Preprocessing(self.DB_FILE, "data/track2_good.db")
        self.model = Model()
        self.voice = Voice()
        self.model_queue = queue.Queue()
        self.model_worker = threading.Thread(target=self._model_worker, name="ModelWorker", daemon=True)
        self.model_worker.start()
        self.voice_queue = queue.Queue()
        self.voice_worker = threading.Thread(target=self._voice_worker, name="VoiceWorker", daemon=True)
        self.voice_worker.start()
        self._responses_lock = threading.Lock()
        self._model_responses = {}
        self._spoken_for_lap = set()
        
        
    def run(self):
        conn = self.init_db(self.DB_FILE)
        sock = self.create_socket(self.UDP_IP, self.UDP_PORT)
        print(f"Lausche auf UDP {self.UDP_IP}:{self.UDP_PORT} ...")
        last_saved = 0.0

        try:
            while True:
                data, _ = sock.recvfrom(512)
                parsed_result = self.parse_telemetry(data)
                if parsed_result is None:
                    continue
                schema_name, parsed = parsed_result
                if parsed.get("engine_max_rpm", 0) == 0:
                    continue
                now = time.monotonic()
                if now - last_saved < self.MIN_SAVE_INTERVAL:
                    continue
                last_saved = now

                timestamp_utc = (datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z")
                self.insert_sample(conn, timestamp_utc, data, schema_name, parsed)
        except KeyboardInterrupt:
            print("\nBeende Listener...")
        finally:
            conn.close()
            sock.close()

    def _enqueue_model_run(self, lap: int, segment: int) -> None:
        self.model_queue.put((lap, segment))

    def _model_worker(self) -> None:
        while True:
            lap, segment = self.model_queue.get()
            try:
                md_text = self.preprocessing.run()
                if not md_text or not md_text.strip():
                    print("No markdown data available; skipping model run.")
                    continue
                response_text = self.model.run(lap, segment, md_text=md_text)
                self._store_model_response(lap, segment, response_text)
            except Exception as exc:
                print(f"Model worker error: {exc}")
            finally:
                self.model_queue.task_done()

    def _enqueue_voice(self, text: str) -> None:
        if not text or not text.strip():
            return
        self.voice_queue.put(text.strip())

    def _voice_worker(self) -> None:
        while True:
            text = self.voice_queue.get()
            try:
                self.voice.run(text)
            except Exception as exc:
                print(f"Voice error: {exc}")
            finally:
                self.voice_queue.task_done()

    def _store_model_response(self, lap: int, segment: int, response_text: str | None) -> None:
        if not response_text or not response_text.strip():
            return
        key = (lap, segment)
        with self._responses_lock:
            self._model_responses[key] = response_text.strip()

    def _speak_previous_lap_response(self, current_lap: int, segment: int) -> None:
        if current_lap <= 0:
            return
        previous_key = (current_lap - 1, segment)
        with self._responses_lock:
            text = self._model_responses.get(previous_key)
            if text is None:
                return
            spoken_key = (current_lap, segment)
            if spoken_key in self._spoken_for_lap:
                return
            self._spoken_for_lap.add(spoken_key)
        self._enqueue_voice(text)

    def load_tuples(self, path: str | Path):
        return ast.literal_eval("[" + Path(path).read_text(encoding="utf-8").strip().strip().rstrip(",") + "]")

    def create_socket(self, ip: str, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        return sock


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
                ("timestamp_utc", "TEXT"),
                ("packet_length", "INTEGER"),
                ("packet_schema", "TEXT"),
                ("raw", "BLOB")
            ]
            + [(name, db_types[fmt]) for name, fmt in self.FORZA_FIELDS]
        )
        return conn


    def parse_telemetry(self, data: bytes) -> tuple[str, dict] | None:
        for schema_name, fields, size in self.SCHEMAS:
            if len(data) != size:
                continue
            fmt = "<" + "".join(fmt for _, fmt in fields)
            values = struct.unpack_from(fmt, data, 0)
            return schema_name, {name: value for (name, _), value in zip(fields, values)}
        return None


    def insert_sample(
        self, conn: sqlite3.Connection, timestamp_utc: str, raw: bytes, schema_name: str, parsed: dict
    ) -> None:
        
        columns = ["timestamp_utc", "packet_length", "packet_schema", "raw"] + [name for name, _ in self.FORZA_FIELDS]
        placeholders = ", ".join(["?"] * len(columns))
        values = [
            timestamp_utc,
            len(raw),
            schema_name,
            sqlite3.Binary(raw)]
        
        for name, _ in self.FORZA_FIELDS:
            values.append(parsed.get(name, 0))
        if self.segment <= 3:
            distance = np.sqrt((values[65]-self.TRACK_2_SEGMENT_POINTS[self.segment][0])**2 + (values[67]-self.TRACK_2_SEGMENT_POINTS[self.segment][1])**2) 
            if distance <= self.threshold:
                if distance <= self.distance:
                    self.distance = distance
                    self.in_zone = True
                else:
                    print(f'Start processing {self.segment}')
                    self._speak_previous_lap_response(self.lap, self.segment)
                    print('Queued model answer ...')
                    self._enqueue_model_run(self.lap, self.segment)
                    self.segment += 1
                    self.distance = np.inf
                    self.in_zone = False
        
        if values[82] != self.lap:
            print(f'Start processing {self.segment}')
            print('Queued model answer ...')
            self._enqueue_model_run(self.lap, self.segment)
            self.lap += 1
            self.segment = 0
            self._speak_previous_lap_response(self.lap, self.segment)

        conn.execute(f"INSERT INTO telemetry_samples ({', '.join(columns)}) VALUES ({placeholders})",values,)
        conn.commit()
