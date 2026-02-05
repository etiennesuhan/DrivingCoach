from soprano.utils.streaming import play_stream
from soprano import SopranoTTS


class Voice():
    def run(self, text_to_read_out: str):
        model = SopranoTTS(backend='transformers', device='cpu', cache_size_mb=100, decoder_batch_size=1)
        stream = model.infer_stream(text_to_read_out, chunk_size=1)
        play_stream(stream)