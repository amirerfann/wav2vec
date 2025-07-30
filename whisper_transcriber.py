# whisper_transcriber.py

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os

class WhisperTranscriber:
    def __init__(self, model_id="models/whisper-small", language="fa"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_id, language=language, task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()
        self.language = language

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        forced_ids = self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")

        with torch.no_grad():
            output_ids = self.model.generate(inputs, forced_decoder_ids=forced_ids)

        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text
