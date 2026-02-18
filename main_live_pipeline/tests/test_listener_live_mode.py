import math
import queue
import sys
import threading
import types
import unittest
import json
from collections import deque
from pathlib import Path


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


from listener import Listener  # noqa: E402


def _build_listener_for_runtime_tests() -> Listener:
    listener = Listener.__new__(Listener)
    listener.DB_FILE = ":memory:"
    listener.last_packet_utc = None
    listener._packet_times = deque(maxlen=200)
    listener.packet_rate = 0.0
    listener.segments = [[100.0, 200.0], [300.0, 400.0]]
    listener.segment = 0
    listener.current_segment_target = None
    listener.current_distance_to_segment = None
    listener.last_position = None
    listener.last_speed = None
    listener.last_yaw = None
    listener.lap_id = 7
    listener.lap = 0
    listener.track_id = "[100, 200]"
    listener._telemetry_lock = threading.Lock()
    listener._telemetry_buffer = deque(maxlen=5000)
    listener._telemetry_seq = 0
    listener.telemetry_queue = queue.Queue(maxsize=100)
    listener._lap_positions_lock = threading.Lock()
    listener._current_lap_positions = []
    listener.distance = math.inf
    listener.in_zone = False
    listener.INCOMPLETE_LAP_SEGMENT_TOLERANCE = 1
    listener.INCOMPLETE_LAP_MIN_SAMPLES = 80
    listener._lap_sample_count = 0
    listener._lap_max_segment = 0
    listener._last_race_time = None
    listener._last_race_lap_number = None
    listener._spoken_for_lap = set()
    listener._responses_lock = threading.Lock()
    listener._model_responses = {}
    listener._response_log_lock = threading.Lock()
    listener._response_log = []
    listener._response_seq = 0
    listener._response_log_max = 1000
    listener.tts_enabled = True
    listener.tts_queue = queue.Queue()
    listener.DEBUG = False
    listener._get_next_lap_id = lambda: 42
    listener._log_event = lambda *_args, **_kwargs: None
    listener.track_name = None
    listener.optimal_lap_id = None
    return listener


class ListenerLiveModeTests(unittest.TestCase):
    def test_free_roam_ingest_appends_telemetry_buffer_item(self):
        listener = _build_listener_for_runtime_tests()
        sample = {
            "position_x": 120.5,
            "position_z": 245.0,
            "speed": 32.0,
            "yaw": -0.35,
            "lap_number": 0,
        }

        listener._ingest_free_roam_sample("2026-02-16T12:00:00.000Z", sample)

        self.assertEqual(listener._telemetry_seq, 1)
        self.assertEqual(len(listener._telemetry_buffer), 1)
        entry = listener._telemetry_buffer[-1]
        self.assertEqual(entry["id"], 1)
        self.assertEqual(entry["data"]["position_x"], 120.5)
        self.assertEqual(entry["data"]["position_z"], 245.0)
        self.assertEqual(entry["data"]["speed"], 32.0)

    def test_free_roam_updates_runtime_fields(self):
        listener = _build_listener_for_runtime_tests()
        sample = {
            "position_x": 140.0,
            "position_z": 210.0,
            "speed": 15.5,
            "yaw": 0.12,
            "lap_number": 0,
        }

        listener._ingest_free_roam_sample("2026-02-16T12:00:00.000Z", sample)

        self.assertEqual(listener.last_position, [140.0, 210.0])
        self.assertEqual(listener.last_speed, 15.5)
        self.assertEqual(listener.last_yaw, 0.12)
        self.assertEqual(listener.current_segment_target, [100.0, 200.0])
        self.assertAlmostEqual(
            listener.current_distance_to_segment,
            math.sqrt((140.0 - 100.0) ** 2 + (210.0 - 200.0) ** 2),
            places=5,
        )

    def test_free_roam_path_does_not_call_db_insert(self):
        listener = _build_listener_for_runtime_tests()
        listener.insert_sample_called = False

        def _mark_called(*_args, **_kwargs):
            listener.insert_sample_called = True

        listener.insert_sample = _mark_called
        sample = {
            "position_x": 1.0,
            "position_z": 2.0,
            "speed": 3.0,
            "lap_number": 0,
        }

        listener._ingest_free_roam_sample("2026-02-16T12:00:00.000Z", sample)

        self.assertFalse(listener.insert_sample_called)
        self.assertEqual(listener._telemetry_seq, 1)

    def test_start_new_lap_keeps_telemetry_sequence(self):
        listener = _build_listener_for_runtime_tests()
        sample = {
            "position_x": 10.0,
            "position_z": 20.0,
            "speed": 5.0,
            "lap_number": 0,
        }
        listener._ingest_free_roam_sample("2026-02-16T12:00:00.000Z", sample)
        listener._ingest_free_roam_sample("2026-02-16T12:00:00.100Z", sample)
        seq_before = listener._telemetry_seq

        listener._start_new_lap(lap_number=1, reset_session=True)

        self.assertEqual(listener._telemetry_seq, seq_before)

    def test_short_menu_pause_does_not_end_active_race(self):
        listener = _build_listener_for_runtime_tests()
        listener._race_active = True
        listener.lap = 3
        listener.lap_id = 99
        listener._last_race_time = 125.0
        listener._last_race_lap_number = 3
        events = []
        listener._log_event = lambda event_type, data: events.append((event_type, data))

        listener._update_race_activity_state(
            in_race=False,
            parsed={"distance_traveled": 0, "engine_max_rpm": 0},
        )

        self.assertTrue(listener._race_active)
        self.assertEqual(events, [])

    def test_long_menu_pause_still_keeps_active_race(self):
        listener = _build_listener_for_runtime_tests()
        listener._race_active = True
        listener.lap = 5
        listener.lap_id = 123
        listener._last_race_time = 200.0
        listener._last_race_lap_number = 5
        events = []
        listener._log_event = lambda event_type, data: events.append((event_type, data))

        listener._update_race_activity_state(
            in_race=False,
            parsed={"distance_traveled": 0, "engine_max_rpm": 0},
        )

        self.assertTrue(listener._race_active)
        self.assertEqual(events, [])

    def test_open_world_packet_ends_active_race_immediately(self):
        listener = _build_listener_for_runtime_tests()
        listener._race_active = True
        listener.lap = 2
        listener.lap_id = 55
        listener._last_race_time = 80.0
        listener._last_race_lap_number = 2
        events = []
        listener._log_event = lambda event_type, data: events.append((event_type, data))

        listener._update_race_activity_state(
            in_race=False,
            parsed={"distance_traveled": 0, "engine_max_rpm": 6500},
        )

        self.assertFalse(listener._race_active)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "race_end")

    def test_race_time_rewind_marks_new_race_context(self):
        listener = _build_listener_for_runtime_tests()
        listener._race_active = True
        listener.lap = 4
        listener.lap_id = 77
        listener._last_race_time = 156.0
        listener._last_race_lap_number = 4
        events = []
        listener._log_event = lambda event_type, data: events.append((event_type, data))

        listener._update_race_activity_state(
            in_race=True,
            parsed={
                "distance_traveled": 12.0,
                "engine_max_rpm": 7000.0,
                "current_race_time": 2.5,
                "lap_number": 1,
            },
        )

        self.assertFalse(listener._race_active)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "race_end")

    def test_race_restart_in_lap_zero_is_detected(self):
        listener = _build_listener_for_runtime_tests()
        listener._race_active = True
        listener.lap = 0
        listener.lap_id = 88
        listener._last_race_time = 18.0
        listener._last_race_lap_number = 0
        events = []
        listener._log_event = lambda event_type, data: events.append((event_type, data))

        listener._update_race_activity_state(
            in_race=True,
            parsed={
                "distance_traveled": 8.0,
                "engine_max_rpm": 6900.0,
                "current_race_time": 0.4,
                "lap_number": 0,
            },
        )

        self.assertFalse(listener._race_active)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "race_end")

    def test_drop_current_lap_if_incomplete_discards_low_progress(self):
        listener = _build_listener_for_runtime_tests()
        listener.lap_id = 66
        listener._lap_sample_count = 20
        listener._lap_max_segment = 1
        listener.segments = [[1, 1], [2, 2], [3, 3], [4, 4]]
        discarded = []
        listener._delete_lap_samples = lambda lap_id, reason, conn=None: discarded.append((lap_id, reason))

        listener._drop_current_lap_if_incomplete(reason="unit_test")

        self.assertEqual(discarded, [(66, "unit_test")])

    def test_drop_current_lap_if_incomplete_keeps_nearly_complete(self):
        listener = _build_listener_for_runtime_tests()
        listener.lap_id = 67
        listener._lap_sample_count = 140
        listener._lap_max_segment = 3
        listener.segments = [[1, 1], [2, 2], [3, 3], [4, 4]]
        discarded = []
        listener._delete_lap_samples = lambda lap_id, reason, conn=None: discarded.append((lap_id, reason))

        listener._drop_current_lap_if_incomplete(reason="unit_test")

        self.assertEqual(discarded, [])

    def test_lap_switch_completion_validation_rejects_short_restart_lap(self):
        listener = _build_listener_for_runtime_tests()
        listener.lap_id = 13
        listener.lap = 0
        listener._lap_sample_count = 61
        listener._lap_max_segment = 1
        listener.segments = [[1, 1], [2, 2], [3, 3], [4, 4]]

        self.assertFalse(listener._is_lap_switch_completion_valid(next_lap=1))

    def test_lap_switch_completion_validation_accepts_forward_complete_lap(self):
        listener = _build_listener_for_runtime_tests()
        listener.lap_id = 68
        listener.lap = 2
        listener._lap_sample_count = 140
        listener._lap_max_segment = 3
        listener.segments = [[1, 1], [2, 2], [3, 3], [4, 4]]

        self.assertTrue(listener._is_lap_switch_completion_valid(next_lap=3))

    def test_store_model_response_logs_feedback_entry(self):
        listener = _build_listener_for_runtime_tests()

        listener._store_model_response(
            lap_id=21,
            lap=2,
            segment=1,
            response_text='{"status": true, "message": "Ausgang frueher aufs Gas."}',
        )

        self.assertEqual(len(listener._response_log), 1)
        entry = listener._response_log[0]
        self.assertEqual(entry["lap_id"], 21)
        self.assertEqual(entry["lap"], 2)
        self.assertEqual(entry["segment"], 1)
        self.assertTrue(entry["status"])
        self.assertEqual(entry["message"], "Ausgang frueher aufs Gas.")
        self.assertEqual(listener._model_responses[(2, 1)], "Ausgang frueher aufs Gas.")

    def test_store_model_response_enqueues_backend_tts(self):
        listener = _build_listener_for_runtime_tests()

        listener._store_model_response(
            lap_id=22,
            lap=3,
            segment=0,
            response_text='{"status": true, "message": "Bremspunkt spaeter setzen."}',
        )

        self.assertFalse(listener.tts_queue.empty())
        self.assertEqual(listener.tts_queue.qsize(), 1)

    def test_get_tts_audio_path_uses_active_voice_suffix(self):
        listener = _build_listener_for_runtime_tests()
        listener._tts_lock = threading.Lock()
        listener._tts_cache = {}
        listener._tts_dir = ROOT / "data" / "test_tts_cache"
        listener._tts_dir.mkdir(parents=True, exist_ok=True)
        listener._model_responses[(4, 2)] = "Saubere Linie, weiter so."

        class _Voice:
            audio_suffix = ".mp3"

            def synthesize_to_wav(self, _text, out_path):
                path = Path(out_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"ID3")
                return path

        listener.voice = _Voice()

        out_path = listener.get_tts_audio_path(
            lap=4,
            segment=2,
            synthesize_if_missing=True,
        )

        self.assertIsNotNone(out_path)
        self.assertEqual(Path(out_path).suffix, ".mp3")
        Path(out_path).unlink(missing_ok=True)

    def test_parse_model_payload_extracts_json_from_markdown_block(self):
        listener = _build_listener_for_runtime_tests()
        raw = """```json
{
  "status": false,
  "message": "Good throttle use, speed carried well."
}
```
Explanation text that should be ignored."""

        payload = listener._parse_model_payload(raw)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], False)
        self.assertEqual(payload["message"], "Good throttle use, speed carried well.")
        self.assertEqual(payload["parsed"], True)

    def test_normalize_feedback_language_translates_known_english_phrase(self):
        listener = _build_listener_for_runtime_tests()
        out = listener._normalize_feedback_language(
            "Good throttle use, speed carried well."
        )
        self.assertEqual(
            out,
            "Gute Gasannahme, die Geschwindigkeit wurde sauber mitgenommen.",
        )

    def test_identify_track_id_uses_relaxed_distance_when_race_active(self):
        listener = _build_listener_for_runtime_tests()
        listener._race_active = True
        listener.track_id = None
        listener.segments = [[0, 0]]
        listener._log_event = lambda *_args, **_kwargs: None

        metadata_path = ROOT / "main_live_pipeline" / "metadata.json"
        original = metadata_path.read_text(encoding="utf-8")
        test_data = {
            "tracks": {
                "[931, 2603]": {
                    "name": "Bola Ocho, Rundstrecke",
                    "lap": 7,
                    "segments": [[610, 2485], [635, 2780]],
                }
            }
        }
        try:
            metadata_path.write_text(
                json.dumps(test_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            listener.identify_track_id([844.78, 2503.92])  # ~132m away
            self.assertEqual(listener.track_id, "[931, 2603]")
            self.assertEqual(listener.optimal_lap_id, 7)
            self.assertEqual(listener.track_name, "Bola Ocho, Rundstrecke")
            self.assertEqual(listener.segments, [[610, 2485], [635, 2780]])
        finally:
            metadata_path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
