from __future__ import annotations

import asyncio
import os
import subprocess
import sys
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
        preferred_backend = os.getenv("FH5_TTS_BACKEND", "auto").strip().lower()
        configured_voice = os.getenv("FH5_TTS_VOICE", "de-DE-KatjaNeural").strip()
        self.voice_name = self._resolve_german_voice(configured_voice)
        self.sapi_voice_name = os.getenv("FH5_SAPI_VOICE", "de-DE").strip()
        self.soprano_model_path = (
            os.getenv("FH5_SOPRANO_MODEL_PATH", "").strip()
            or os.getenv("FH5_SOPRANO_MODEL", "").strip()
        )
        self.backend = self._resolve_backend(preferred_backend)
        self.audio_suffix = ".mp3" if self.backend == "edge" else ".wav"
        self.model = None
        if self.backend == "soprano":
            self.model = self._create_soprano_model()

    def _resolve_backend(self, preferred_backend: str) -> str:
        if preferred_backend in ("auto", ""):
            if edge_tts is not None:
                return "edge"
            if self._sapi_available():
                return "sapi"
            if SopranoTTS is not None:
                return "soprano"
            raise RuntimeError("No TTS backend available. Install edge-tts or soprano.")

        if preferred_backend in ("edge", "edge_tts") and edge_tts is not None:
            return "edge"
        if preferred_backend in ("sapi", "windows", "system") and self._sapi_available():
            return "sapi"
        if preferred_backend == "soprano" and SopranoTTS is not None:
            return "soprano"
        raise RuntimeError(
            f"Requested TTS backend '{preferred_backend}' is not available. "
            "Use FH5_TTS_BACKEND=auto, install edge-tts, or configure a working backend."
        )

    def _resolve_german_voice(self, voice_name: str) -> str:
        normalized = (voice_name or "").strip()
        if normalized.lower().startswith("de-"):
            return normalized
        return "de-DE-KatjaNeural"

    def _sapi_available(self) -> bool:
        return sys.platform.startswith("win")

    def _create_soprano_model(self):
        kwargs = {
            "backend": "transformers",
            "device": "cpu",
            "cache_size_mb": 100,
            "decoder_batch_size": 1,
        }
        if self.soprano_model_path:
            kwargs["model_path"] = self.soprano_model_path
        return SopranoTTS(**kwargs)

    def _sapi_synthesize(self, text_to_read_out: str, out_path: Path) -> None:
        script = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $text = $env:FH5_SAPI_TEXT
  $outPath = $env:FH5_SAPI_OUT_PATH
  $preferred = $env:FH5_SAPI_VOICE
  $voices = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo }
  $selected = $null
  if ($preferred) {
    $selected = $voices | Where-Object { $_.Name -eq $preferred -or $_.Culture.Name -eq $preferred } | Select-Object -First 1
  }
  if (-not $selected) {
    $selected = $voices | Where-Object { $_.Culture.Name -like 'de-*' } | Select-Object -First 1
  }
  if (-not $selected) {
    $selected = $voices | Select-Object -First 1
  }
  if ($selected) {
    $s.SelectVoice($selected.Name)
  }
  $s.SetOutputToWaveFile($outPath)
  $s.Speak($text)
}
finally {
  $s.Dispose()
}
"""
        env = os.environ.copy()
        env["FH5_SAPI_TEXT"] = text_to_read_out
        env["FH5_SAPI_OUT_PATH"] = out_path.as_posix()
        env["FH5_SAPI_VOICE"] = self.sapi_voice_name
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"SAPI TTS failed: {err or 'unknown error'}")

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
        if self.backend == "sapi":
            if out_path.suffix.lower() != ".wav":
                out_path = out_path.with_suffix(".wav")
            try:
                self._sapi_synthesize(text, out_path)
            except Exception as exc:
                print(f"SAPI TTS failed, fallback to soprano: {exc}")
                if SopranoTTS is None:
                    raise
                if self.model is None:
                    self.model = self._create_soprano_model()
                self.model.infer(text, out_path=out_path.as_posix())
                self.backend = "soprano"
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
