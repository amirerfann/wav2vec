# wav2vec2_corrector.py

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class Wav2Vec2Corrector:
    def __init__(self, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def correct(self, audio_path: str) -> str:
        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # تبدیل به mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # تبدیل به numpy برای processor
        waveform_np = waveform.numpy()

        inputs = self.processor(
            waveform_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription
