import os
import time
import csv
from pathlib import Path
import torchaudio
from whisper_transcriber import WhisperTranscriber
from wav2vec2_corrector import Wav2Vec2Corrector
from evaluate import load

CLIP_DIR = Path("mini_corpus/clips")
TSV_PATH = Path("mini_corpus/validated.tsv")
OUTPUT_CSV = Path("evaluation_results.csv")

def load_ground_truths(tsv_path):
    gt_map = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:  # حذف header
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            file_name, text = parts[1], parts[2]
            gt_map[file_name] = text
    return gt_map

wer_metric = load("wer")
def wer(reference, hypothesis):
    return wer_metric.compute(predictions=[hypothesis], references=[reference])

def run_batch_evaluation():
    gt_map = load_ground_truths(TSV_PATH)
    whisper = WhisperTranscriber()
    wav2vec = Wav2Vec2Corrector()

    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "file_name", "true_text",
            "whisper_text", "whisper_runtime", "whisper_wer",
            "wav2vec2_text", "wav2vec2_runtime", "wav2vec2_wer"
        ])

        for audio_file in CLIP_DIR.glob("*.mp3"):
            file_name = audio_file.name
            if file_name not in gt_map:
                print(f"⚠️ Skipping: {file_name} (no ground truth)")
                continue
            true_text = gt_map[file_name]

            try:
                # Whisper
                start = time.time()
                whisper_text = whisper.transcribe(str(audio_file))
                whisper_time = time.time() - start
                whisper_error = wer(true_text, whisper_text)

                # Wav2Vec2
                start = time.time()
                wav2vec2_text = wav2vec.correct(str(audio_file))
                wav2vec2_time = time.time() - start
                wav2vec2_error = wer(true_text, wav2vec2_text)

                writer.writerow([
                    file_name, true_text,
                    whisper_text, round(whisper_time, 2), round(whisper_error, 3),
                    wav2vec2_text, round(wav2vec2_time, 2), round(wav2vec2_error, 3)
                ])
                print(f"✅ Done: {file_name}")
            except Exception as e:
                print(f"❌ Error with {file_name}: {e}")


if __name__ == "__main__":
    run_batch_evaluation()

