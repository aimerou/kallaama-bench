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
| - | *Adia-TTS* | 3.02 | 2.45 | 0.72 | 61.27 | 1.42 | - | - | 2026-05-25 |
| - | *GalsenAI-TTS* | 3.36 | 2.81 | 0.71 | 60.37 | 0.34 | 0.58 | 0.82 | 2026-05-25 |
| - | *Oolel-Voices* | **4.17** | **3.01** | **0.61** | 58.20 | 0.64 | 0.51 | 0.78 | 2026-05-25 |
| - | *OmniVoice* | 3.56 | 2.84 | 0.69 | 58.64 | **0.15** | **0.67** | **0.85** | 2026-05-25 |
| - | *OmniVoice-FT* | **4.17** | 2.68 | 0.67 | **67.69** | **0.15** | 0.57 | **0.85** | 2026-07-14 |
| - | *OmniVoice-FT-CS* | 4.10 | 2.78 | **0.61** | 65.05 | **0.15** | 0.63 | 0.84 | 2026-07-14 |
| - | *MiraTTS* | - | 2.37 | 1.05 | 53.16 | 0.18 | 0.34 | 0.77 | 2026-05-25 |
| - | *MiraTTS-FT* | 3.71 | 2.50 | 0.67 | 59.68 | 0.18 | 0.47 | 0.79 | 2026-07-14 |
| - | *MiraTTS-FT-CS* | 3.23 | 2.50 | 0.72 | 57.88 | 0.18 | 0.46 | 0.81 | 2026-07-14 |

---

## Metrics

| Metric | Tool | What it measures |
|---|---|---|
| **UTMOS** ↑ | UTMOSv2 | Predicted naturalness (1–5) |
| **WER** ↓ | Omnilingual ASR + jiwer | Intelligibility |
| **SNR** ↑ | PyAnnote Brouhaha | Signal cleanliness (dB) |
| **SpkSim** ↑ | ECAPA-TDNN cosine similarity | Voice similarity to reference speaker |
| **F0Corr** ↑ | pYIN + Pearson correlation | Pitch dynamics match |
| **RTF** ↓ | Self-reported | Speed of synthesis |

Top-5 systems are evaluated monthly by native Wolof speakers to deliver the mean opinion score (MOS).

---

## Repository layout

```
kallaamabench/
├── data/
│   ├── corpus.csv            # Metadata of Wolof test utterances (filename|transcription|speaker|gender)
│   ├── groundtruth.npz       # Precomputed speaker embeddings + F0 stats
│   └── references            # Reference audio samples used for voice cloning (WAV, 16 kHz mono)
│
├── inference/                # Inference scripts for each evaluated model
│
├── results/                  # JSON files containing evaluation results
│
├── utils/
│    ├── evaluation           # Python scripts used for MOS evaluation
│    └── hf-utils             # Python script to donwload data from Hugging Face
│
├── README.md
├── SUBMISSION_FORMAT.md      # Instructions to submit a new TTS model to the leaderboard
├── evaluate.py               # Evaluation script
└── requirements.txt          # Package dependecies with specific versions

```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Synthesise the test sentences

```python
import time
import torchaudio
import pandas as pd
from my_tts_model import TTSModel

submission_dir = "submissions/my_tts_model"
os.makedirs(submission_dir, exist_ok=True)

model  = TTSModel.load("my_checkpoint")
corpus = pd.read_csv("data/corpus.csv", sep="|", header=0, on_bad_lines="skip", dtype=str)

rtf_values = []
for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Generating audio"):
    start_time     = time.time()
    audio, sr      = model.synthesise(row["transcription"])
    torchaudio.save(os.path.join(submission_dir, row["filename"]), audio, sr)
    rtf_values.append((time.time() - start_time) / audio_duration)
```

### 3. Create `metadata.json` in the submission folder

```json
{
    "team"            : "MyTeam",
    "system"          : "FastSpeech2 + HiFi-GAN",
    "submission_date" : "2026-05-25",
    "rtf"             : 0.038,
    "contact"         : "me@example.com"
}
```

### 4. Validate locally on the test set

```bash
python evaluate.py evaluate \
    --submission    submissions/my_tts_model/ \
    --groundtruth   data/groundtruth.npz \
    --corpus        data/corpus.csv \
    --output        results/my_tts_model/my_tts_model.json
```

### 5. Submit

Zip your submission and open a **New Submission** issue in this repository.
See [SUBMISSION_FORMAT.md](SUBMISSION_FORMAT.md) for details.

---

## Data policy

- The test set audio (1602 recordings) is released only in its compressed Numpy format.
- `data/corpus.csv` and `data/groundtruth.npz` are public.
- `groundtruth.npz` contains speaker embeddings (192-dim) and F0 histogram statistics only — raw audio cannot be reconstructed.

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

You are free to share and adapt under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
