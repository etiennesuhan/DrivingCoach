from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from scipy.io import wavfile
import sounddevice as sd

try:
    import edge_tts  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    edge_tts = None

try:
    from soprano.utils.streaming import play_stream  # type: ignore
    from soprano import SopranoTTS  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    play_stream = None
    SopranoTTS = None


class Voice():
    def __init__(self):
        preferred_backend = os.getenv("FH5_TTS_BACKEND", "edge").strip().lower()
        self.voice_name = os.getenv("FH5_TTS_VOICE", "de-DE-KatjaNeural")
        self.backend = self._resolve_backend(preferred_backend)
        self.audio_suffix = ".mp3" if self.backend == "edge" else ".wav"
        self.model = None
        if self.backend == "soprano":
            self.model = SopranoTTS(
                backend="transformers",
                device="cpu",
                cache_size_mb=100,
                decoder_batch_size=1,
            )

    def _resolve_backend(self, preferred_backend: str) -> str:
        if preferred_backend in ("edge", "edge_tts") and edge_tts is not None:
            return "edge"
        if preferred_backend == "soprano" and SopranoTTS is not None:
            return "soprano"
        if edge_tts is not None:
            return "edge"
        if SopranoTTS is not None:
            return "soprano"
        raise RuntimeError("No TTS backend available. Install edge-tts or soprano.")

    async def _edge_synthesize(self, text_to_read_out: str, out_path: Path) -> None:
        communicator = edge_tts.Communicate(text_to_read_out, self.voice_name)
        await communicator.save(out_path.as_posix())

    def run(self, text_to_read_out: str):
        text = (text_to_read_out or "").strip()
        if not text:
            return
        if self.backend == "soprano" and self.model is not None and play_stream is not None:
            stream = self.model.infer_stream(text, chunk_size=1)
            play_stream(stream)
            return

        temp_path = Path(gettempdir()) / f"fh5_voice_{uuid4().hex}{self.audio_suffix}"
        generated_path = self.synthesize_to_wav(text, temp_path)
        try:
            self.play_wav(generated_path)
        finally:
            try:
                generated_path.unlink(missing_ok=True)
            except Exception:
                pass

    def synthesize_to_wav(self, text_to_read_out: str, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        text = (text_to_read_out or "").strip()
        if not text:
            raise ValueError("text_to_read_out must not be empty")

        if self.backend == "edge":
            if out_path.suffix.lower() not in (".mp3", ".wav"):
                out_path = out_path.with_suffix(".mp3")
            asyncio.run(self._edge_synthesize(text, out_path))
            return out_path

        if self.model is None:
            raise RuntimeError("Soprano backend selected but model is not initialized")
        self.model.infer(text, out_path=out_path.as_posix())
        return out_path

    def play_wav(self, path: str | Path) -> None:
        wav_path = Path(path)
        if wav_path.suffix.lower() != ".wav":
            raise RuntimeError("Local playback supports only WAV; use browser playback for non-WAV audio.")
        sr, audio = wavfile.read(wav_path.as_posix())
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        sd.play(audio, sr)
        sd.wait()
