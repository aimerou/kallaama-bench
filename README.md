# KallaamaBench 🎙️

**The first open TTS benchmark for the Wolof language**

> *Kallaama* — "language" in Wolof

KallaamaBench provides a standardised evaluation protocol for Wolof TTS systems.
Participants synthesise audio from 1602 text prompts; the evaluation script
scores submissions against hidden reference data.

---

## TTS Leaderboard

*Last updated: 2026-05-12*

| Rank | Team | System | Human MOS ↑ | UTMOS ↑ | WER ↓ | SNR ↑ (dB) | RTF ↓ | SpkSim ↓ | F0Corr ↑ | Date |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| — | *Baseline* | *VITS (10 h Wolof)* | *3.1 ± 0.5* | *3.12* | *3.21* | *0.38* | *18.4* | *0.09* | *0.62* | 2026-01-15 |

---

## Metrics

| Metric | Tool | What it measures |
|---|---|---|
| **UTMOS** ↑ | UTMOSv2 | Predicted naturalness (1–5) |
| **WER** ↓ | Omnilingual ASR + jiwer | Intelligibility |
| **SNR** ↑ | SpeechBrain SepFormer | Signal cleanliness (dB) |
| **SpkSim** ↑ | ECAPA-TDNN cosine similarity | Voice similarity to reference speaker |
| **F0Corr** ↑ | pYIN + Pearson correlation | Pitch dynamics match |
| **RTF** ↓ | Self-reported | Speed of synthesis |

Top-5 systems are evaluated monthly by native Wolof speakers (CMOS, scale −3 to +3).

---

## Repository layout

```
kallaamabench/
├── evaluate.py              # Evaluation script
├── SUBMISSION_FORMAT.md
├── README.md
│
├── data/
│   ├── corpus.csv           # 1602 Wolof sentences (filename|transcription|speaker|gender)
│   ├── groundtruth.npz       # Precomputed speaker embeddings + F0 stats
│   └── rtf_scores.json
│
└── dev/
    ├── corpus_dev.csv        # 100-sentence dev subset
    ├── audio/                # Reference audio for dev set (WAV, 16 kHz mono)
    └── groundtruth_dev.npz
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Synthesise the test sentences

```python
import pandas as pd, soundfile as sf
from your_tts_model import TTSModel

model  = TTSModel.load("your_checkpoint")
corpus = pd.read_csv("data/corpus.csv", sep="|")

os.makedirs("my_submission/audio", exist_ok=True)
for _, row in corpus.iterrows():
    audio, sr = model.synthesise(row["transcription"])
    sf.write(f"my_submission/audio/{row['filename']}", audio, sr)
```

### 3. Create `metadata.json`

```json
{
    "team"            : "MyTeam",
    "system"          : "FastSpeech2 + HiFi-GAN",
    "submission_date" : "2026-05-12",
    "rtf"             : 0.038,
    "contact"         : "me@example.com"
}
```

### 4. Validate locally on the dev set

```bash
python evaluate.py evaluate \
    --submission  my_submission/ \
    --groundtruth  dev/groundtruth_dev.npz \
    --corpus      dev/corpus_dev.csv \
    --output      dev_results.json
```

### 5. Submit

Zip your submission and open a **New Submission** issue in this repository.
See [SUBMISSION_FORMAT.md](SUBMISSION_FORMAT.md) for details.

---

## Data policy

- The test set audio (1602 recordings) is **never released**.
- `data/corpus.csv` and `data/groundtruth.npz` are public.
- `groundtruth.npz` contains speaker embeddings (192-dim) and F0 histogram statistics only — raw audio cannot be reconstructed.

---

## Citation

```bibtex
@misc{kallaamabench2026,
  title   = {KallaamaBench: A TTS Evaluation Benchmark for Wolof},
  author  = {Your Name and Collaborators},
  year    = {2026},
  url     = {https://github.com/yourorg/kallaamabench}
}
```
