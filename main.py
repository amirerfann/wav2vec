# main.py

from whisper_transcriber import WhisperTranscriber
from wav2vec2_corrector import Wav2Vec2Corrector
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and refine Persian audio")
    parser.add_argument("--audio", type=str, default="audios/sample 2.wav", help="Path to input audio file")
    args = parser.parse_args()

    whisper = WhisperTranscriber()
    # wav2vec2 = Wav2Vec2Corrector()

    print("\n🔹 Whisper output:")
    whisper_result = whisper.transcribe(args.audio)
    print(whisper_result)

    # print("\n🔹 Wav2Vec2 refined output:")
    # corrected_result = wav2vec2.correct(args.audio)
    # print(corrected_result)

    corrector = Wav2Vec2Corrector()
    corrected_result = corrector.correct(args.audio)

    print("\n🔹 Wav2Vec2 refined output:")
    print(corrected_result)
