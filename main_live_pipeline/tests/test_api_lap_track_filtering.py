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
    def __init__(self, db_file: Path, track_id: str | None):
        self.DB_FILE = db_file
        self.track_id = track_id
        self.lap_id = None
        self._race_active = False
        self.telemetry_queue = queue.Queue()
        self._events = queue.Queue()
        self._feedback = queue.Queue()

    def get_status(self):
        return {"track_id": self.track_id}

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
            (1, "[1, 1]", "2026-02-16T10:00:00.000Z", 1, 10.0, 20.0, 30.0, 0.1),
            (1, "[1, 1]", "2026-02-16T10:00:00.100Z", 1, 11.0, 21.0, 31.0, 0.1),
            (2, "[1, 1]", "2026-02-16T10:01:00.000Z", 2, 12.0, 22.0, 32.0, 0.2),
            (3, "[2, 2]", "2026-02-16T10:02:00.000Z", 1, 13.0, 23.0, 33.0, 0.3),
        ]
        conn.executemany(
            """
            INSERT INTO telemetry_samples (
                lap_id, track_id, timestamp_utc, lap_number, position_x, position_z, speed, yaw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


@unittest.skipIf(TestClient is None or create_app is None, "fastapi is not installed")
class ApiLapTrackFilteringTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        _create_test_db(self.db_path)
        self.listener = _DummyListener(self.db_path, track_id="[1, 1]")
        self.app = create_app(self.listener)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self._tmp.cleanup()

    def test_laps_default_uses_current_track_filter(self):
        response = self.client.get("/laps")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        laps = payload["laps"]
        lap_ids = sorted(item["lap_id"] for item in laps)
        self.assertEqual(lap_ids, [1, 2])
        self.assertTrue(all(item["track_id"] == "[1, 1]" for item in laps))

    def test_compare_rejects_laps_from_different_tracks(self):
        response = self.client.get("/compare", params={"lap_a": 1, "lap_b": 3})
        self.assertEqual(response.status_code, 400)
        self.assertIn("different tracks", response.json()["detail"])

    def test_compare_allows_laps_from_same_track(self):
        response = self.client.get("/compare", params={"lap_a": 1, "lap_b": 2})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["track_id"], "[1, 1]")
        self.assertEqual(payload["lap_a"]["track_id"], "[1, 1]")
        self.assertEqual(payload["lap_b"]["track_id"], "[1, 1]")

    def test_compare_rejects_when_requested_track_does_not_match(self):
        response = self.client.get(
            "/compare",
            params={"lap_a": 1, "lap_b": 2, "track_id": "[2, 2]"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("requested track", response.json()["detail"])

    def test_delete_lap_removes_it_from_lap_overview(self):
        delete_response = self.client.delete("/laps/2")
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json()["lap_id"], 2)

        response = self.client.get("/laps")
        self.assertEqual(response.status_code, 200)
        lap_ids = sorted(item["lap_id"] for item in response.json()["laps"])
        self.assertEqual(lap_ids, [1])

    def test_delete_active_lap_is_rejected(self):
        self.listener._race_active = True
        self.listener.lap_id = 1

        response = self.client.delete("/laps/1")
        self.assertEqual(response.status_code, 409)
        self.assertIn("active lap", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
