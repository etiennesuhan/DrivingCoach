import queue
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    TestClient = None


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "main_live_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


if "preprocessing" not in sys.modules:
    preprocessing_module = types.ModuleType("preprocessing")

    class _FakePreprocessing:  # pragma: no cover - stub only
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return ""

    preprocessing_module.Preprocessing = _FakePreprocessing
    sys.modules["preprocessing"] = preprocessing_module


if "model" not in sys.modules:
    model_module = types.ModuleType("model")

    class _FakeModel:  # pragma: no cover - stub only
        def run(self, *args, **kwargs):
            return None

    model_module.Model = _FakeModel
    sys.modules["model"] = model_module


if "voice" not in sys.modules:
    voice_module = types.ModuleType("voice")

    class _FakeVoice:  # pragma: no cover - stub only
        def synthesize_to_wav(self, *args, **kwargs):
            return Path("dummy.wav")

        def play_wav(self, *args, **kwargs):
            return None

    voice_module.Voice = _FakeVoice
    sys.modules["voice"] = voice_module


if TestClient is not None:
    from api import create_app  # noqa: E402
else:  # pragma: no cover - environment dependent
    create_app = None


class _DummyListener:
    def __init__(self, db_file: Path):
        self.DB_FILE = db_file
        self.track_id = "[1, 1]"
        self.driver_name = "Etienne"
        self.lap_id = None
        self._race_active = False
        self.telemetry_queue = queue.Queue()
        self._events = queue.Queue()
        self._feedback = queue.Queue()

    def set_driver_name(self, name: str | None):
        normalized = (name or "").strip()
        self.driver_name = normalized if normalized else "Etienne"

    def get_status(self):
        return {"track_id": self.track_id, "driver_name": self.driver_name}

    def get_feedback(self, **_kwargs):
        return []

    def get_events(self, **_kwargs):
        return []

    def get_live_positions(self, **_kwargs):
        return []

    def get_current_lap_positions(self, **_kwargs):
        return []

    def ack_feedback(self, _feedback_id: int):
        return True

    def set_paused(self, _paused: bool):
        return None

    def reset_session(self):
        return None

    def _log_event(self, _event_type: str, _data: dict):
        return None


def _create_test_db(path: Path) -> None:
    conn = sqlite3.connect(path.as_posix())
    try:
        conn.execute(
            """
            CREATE TABLE telemetry_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lap_id INTEGER NOT NULL,
                track_id TEXT,
                driver_name TEXT,
                timestamp_utc TEXT NOT NULL,
                lap_number INTEGER NOT NULL,
                position_x REAL NOT NULL,
                position_z REAL NOT NULL,
                speed REAL NOT NULL,
                yaw REAL NOT NULL
            )
            """
        )
        rows = [
            (1, "[1, 1]", "Etienne", "2026-02-16T10:00:00.000Z", 1, 10.0, 20.0, 30.0, 0.1),
            (1, "[1, 1]", "", "2026-02-16T10:00:00.100Z", 1, 11.0, 21.0, 31.0, 0.1),
            (2, "[1, 1]", "Max", "2026-02-16T10:01:00.000Z", 2, 12.0, 22.0, 32.0, 0.2),
        ]
        conn.executemany(
            """
            INSERT INTO telemetry_samples (
                lap_id, track_id, driver_name, timestamp_utc, lap_number, position_x, position_z, speed, yaw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


@unittest.skipIf(TestClient is None or create_app is None, "fastapi is not installed")
class ApiDriverNameTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        _create_test_db(self.db_path)
        self.listener = _DummyListener(self.db_path)
        self.app = create_app(self.listener)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self._tmp.cleanup()

    def test_driver_endpoint_get_and_put(self):
        response = self.client.get("/driver")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["driver_name"], "Etienne")

        updated = self.client.put("/driver", json={"driver_name": "Anna"})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["driver_name"], "Anna")
        self.assertEqual(self.listener.driver_name, "Anna")

    def test_laps_include_driver_name(self):
        response = self.client.get("/laps")
        self.assertEqual(response.status_code, 200)
        laps = response.json()["laps"]
        by_lap_id = {item["lap_id"]: item for item in laps}
        self.assertEqual(by_lap_id[1]["driver_name"], "Etienne")
        self.assertEqual(by_lap_id[2]["driver_name"], "Max")


if __name__ == "__main__":
    unittest.main()
