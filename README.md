# KallaamaBench 🎙️

**The first open TTS benchmark for the Wolof language**

> *Kallaama* — "language" in Wolof

KallaamaBench provides a standardised evaluation protocol for Wolof TTS systems.
Participants synthesise audio from 1602 text prompts; the evaluation script
scores submissions against hidden reference data.

---

## TTS Leaderboard

*Last updated: 2026-05-25*

| Rank | System | MOS↑ | UTMOS↑ | WER↓ | SNR↑ (dB) | RTF↓ | SpkSim↑ | F0 Corr↑ | Date |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | *Oolel-Voices* | *4.17* | *3.01* | *0.61* | *170.74* | *0.64* | *0.51* | *0.78* | 2026-05-25 |
| 2 | *Mira-WO* | *3.61* | *2.57* | *0.66* | *12.33* | *0.18* | *0.48* | *0.69* | 2026-05-25 |
| 3 | *OmniVoice* | *3.56* | *2.84* | *0.69* | *100.00* | *0.15* | *0.67* | *0.85* | 2026-05-25 |
| 4 | *Galsen AI TTS* | *3.36* | *2.81* | *0.71* | *53.94* | *0.34* | *0.58* | *0.82* | 2026-05-25 |
| 5 | *Adia-TTS* | *3.02* | *2.45* | *0.72* | *29.02* | *1.42* | *-* | *-* | 2026-05-25 |
| 6 | *Mira-EN* | *-* | *2.37* | *1.05* | *11.40* | *0.18* | *0.34* | *0.77* | 2026-05-25 |

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

Top-5 systems are evaluated monthly by native Wolof speakers to deliver the mean opinion score (MOS).

---

## Repository layout

```
kallaamabench/
├── data/
│   ├── corpus.csv            # Metadata of 1602 Wolof sentences (filename|transcription|speaker|gender)
│   ├── groundtruth.npz       # Precomputed speaker embeddings + F0 stats
│   └── references            # Reference audio samples used for voice cloning (WAV, 16 kHz mono)
│
├── models/                   # Inference scripts for each evaluated model
│
├── results/                  # JSON files containing evaluation results
│
├── utils/
│    ├── copy_mos_files.csv   # Copy files used for MOS evaluation
│    └── verbalizer           # Script used for text normalization (dates, currencies, etc.)
│
├── README.md
├── SUBMISSION_FORMAT.md      # Instructions to submit a new TTS model to the leaderboard
├── evaluate.py               # Evaluation script
└── requirements.txt          # Package requirements with specific versions

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
