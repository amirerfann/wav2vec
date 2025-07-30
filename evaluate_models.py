import time
import os
from pathlib import Path
from whisper_transcriber import WhisperTranscriber
from wav2vec2_corrector import Wav2Vec2Corrector
from evaluate import load

AUDIO_DIR = Path("audios")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

wer_metric = load("wer")

def wer(reference, hypothesis):
    return wer_metric.compute(predictions=[hypothesis], references=[reference])

def load_ground_truth(audio_path: Path) -> str:
    gt_path = audio_path.with_suffix(".txt")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth file for: {audio_path.name}")
    return gt_path.read_text(encoding="utf-8").strip()

def run_evaluation(audio_path: Path):
    whisper = WhisperTranscriber()
    wav2vec = Wav2Vec2Corrector()

    ground_truth = load_ground_truth(audio_path)

    start = time.time()
    whisper_out = whisper.transcribe(str(audio_path))
    whisper_time = time.time() - start
    whisper_wer = wer(ground_truth, whisper_out)

    start = time.time()
    wav2vec_out = wav2vec.correct(str(audio_path))
    wav2vec_time = time.time() - start
    wav2vec_wer = wer(ground_truth, wav2vec_out)

    report = f"""
## 📄 Evaluation Report for `{audio_path.name}`

**Ground Truth**  
{ground_truth}

---

### 🔹 Whisper
- ⏱ Time: `{whisper_time:.2f}s`
- 📝 Output: `{whisper_out}`
- ❌ WER: `{whisper_wer:.3f}`

---

### 🔹 Wav2Vec2
- ⏱ Time: `{wav2vec_time:.2f}s`
- 📝 Output: `{wav2vec_out}`
- ❌ WER: `{wav2vec_wer:.3f}`
"""

    out_path = REPORT_DIR / f"{audio_path.stem}.md"
    out_path.write_text(report.strip(), encoding="utf-8")
    print(f"✅ Report saved: {out_path}")

if __name__ == "__main__":
    wav_files = list(AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        print("⚠️ No .wav files found in audios/")
        exit(1)

    for audio_file in wav_files:
        try:
            run_evaluation(audio_file)
        except Exception as e:
            print(f"❌ Failed to process {audio_file.name}: {e}")
