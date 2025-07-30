# Persian Speech-to-Text Transcription System

A dual-model pipeline for converting Persian audio to text using **Whisper** and **Wav2Vec2**. Designed for high-accuracy transcription of Farsi speech in WAV format.

---

## 📌 Project Features

- 🎙 Converts `.wav` Persian audio to text
- 🤖 Uses `Whisper` for multilingual transcription
- 🧬 Uses `Wav2Vec2` (fine-tuned on Persian) for correction/refinement
- 📝 Generates detailed WER-based performance reports for all audio files in the `audios/` folder

---

## 📂 File Structure

```
.
├── main.py                     # Main entry to run whisper and wav2vec2 transcription
├── whisper_transcriber.py     # Whisper-based transcriber
├── wav2vec2_corrector.py      # Wav2Vec2-based corrector
├── evaluate_models.py         # Accuracy report (WER), runtime benchmarks
├── requirements.txt           # Required Python packages
├── audios/                    # Input audio files (.wav) and ground truths (.txt)
├── reports/                   # Per-file evaluation markdowns generated here
└── README.md
```

---

## 🧪 Requirements

Install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:

```text
torch
transformers
torchaudio
evaluate
```

---

## 🚀 Usage

### 1. Basic Transcription

Run transcription using both models:

```bash
python3 main.py --audio "audios/sample.wav"
```

### 2. Batch Evaluation

To evaluate **all** `.wav` files in the `audios/` directory:

```bash
python3 evaluate_models.py
```

Each `.wav` file must have a matching `.txt` ground truth file with the same name. For example:

```
audios/sample.wav
└── audios/sample.txt
```

Results will be saved as individual `.md` reports in the `reports/` folder.

---

## 📊 Sample Evaluation Report Format

```
## 📄 Evaluation Report for `sample.wav`

**Ground Truth**  
قرص دیفن هیدرامین هر دوازده ساعت یک عدد

---

### 🔹 Whisper
- ⏱ Time: `2.14s`
- 📝 Output: `قرص دیفن هید رمین هر دواز ده ساعت یک اتد`
- ❌ WER: `0.909`

---

### 🔹 Wav2Vec2
- ⏱ Time: `1.99s`
- 📝 Output: `قوث دیفنهیدزامین هر دوازده ساعت یکاردز به مدت پنج بوز`
- ❌ WER: `0.455`
```

### WER Meaning:

- **WER (Word Error Rate)** is a ratio: `#errors / #words` in reference
- Lower = better. `WER: 0.0` means perfect match.

---

## 💡 Notes

- Whisper is fast and general-purpose, but may struggle on Persian.
- Wav2Vec2 is slower but more accurate for Farsi audio.
- Evaluation uses HuggingFace `evaluate` and `wer` metric.

---

## 🔧 TODO / Suggestions

-

