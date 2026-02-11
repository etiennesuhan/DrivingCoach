from pathlib import Path
from soprano.utils.streaming import play_stream
from soprano import SopranoTTS
from scipy.io import wavfile
import sounddevice as sd


class Voice():
    def __init__(self):
        self.model = SopranoTTS(backend='transformers', device='cpu', cache_size_mb=100, decoder_batch_size=1)

    def run(self, text_to_read_out: str):
        stream = self.model.infer_stream(text_to_read_out, chunk_size=1)
        play_stream(stream)

    def synthesize_to_wav(self, text_to_read_out: str, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.infer(text_to_read_out, out_path=out_path.as_posix())
        return out_path

    def play_wav(self, path: str | Path) -> None:
        wav_path = Path(path)
        sr, audio = wavfile.read(wav_path.as_posix())
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        sd.play(audio, sr)
        sd.wait()
