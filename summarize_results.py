import csv
import time
from pathlib import Path

INPUT_CSV = Path("evaluation_results.csv")
OUTPUT_SUMMARY = Path("evaluation_summary.csv")

def summarize():
    start_all = time.time()

    whisper_times, whisper_wers = [], []
    wav2vec2_times, wav2vec2_wers = [], []

    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                whisper_times.append(float(row["whisper_runtime"]))
                whisper_wers.append(float(row["whisper_wer"]))
                wav2vec2_times.append(float(row["wav2vec2_runtime"]))
                wav2vec2_wers.append(float(row["wav2vec2_wer"]))
            except:
                continue

    total_files = len(whisper_times)

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    summary = {
        "total_files": total_files,
        "whisper_mean_runtime": avg(whisper_times),
        "whisper_mean_wer": avg(whisper_wers),
        "wav2vec2_mean_runtime": avg(wav2vec2_times),
        "wav2vec2_mean_wer": avg(wav2vec2_wers),
    }

    with open(OUTPUT_SUMMARY, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print(f"✅ Summary saved to {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    summarize()
