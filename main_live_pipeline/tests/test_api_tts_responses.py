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
    def __init__(self, audio_path: Path, preview_path: Path):
        self._audio_path = audio_path
        self._preview_path = preview_path

    def get_tts_audio_path(self, **_kwargs):
        return self._audio_path

    def synthesize_preview_tts(self, _text: str):
        return self._preview_path


@unittest.skipIf(TestClient is None or create_app is None, "fastapi is not installed")
class ApiTtsResponsesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp_dir = Path(self._tmp.name)
        self.audio_path = tmp_dir / "lap1_seg0.mp3"
        self.preview_path = tmp_dir / "preview.wav"
        self.audio_path.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")
        self.preview_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        self.listener = _DummyListener(
            audio_path=self.audio_path,
            preview_path=self.preview_path,
        )
        self.app = create_app(self.listener)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self._tmp.cleanup()

    def test_tts_audio_response_is_audio_without_attachment(self):
        response = self.client.get(
            "/tts/audio",
            params={"lap": 1, "segment": 0},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("content-type", "").startswith("audio/"))
        self.assertNotIn("attachment", response.headers.get("content-disposition", "").lower())
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_tts_preview_response_is_audio_without_attachment(self):
        response = self.client.get(
            "/tts/preview",
            params={"text": "Hallo"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("content-type", "").startswith("audio/"))
        self.assertNotIn("attachment", response.headers.get("content-disposition", "").lower())
        self.assertEqual(response.headers.get("cache-control"), "no-store")


if __name__ == "__main__":
    unittest.main()
