# KallaamaBench 🎙️

**The first open TTS benchmark for the Wolof language**

> *Kallaama* — "language" in Wolof

KallaamaBench provides a standardised, blind evaluation protocol for
Wolof Text-to-Speech systems.  Participants synthesise audio from a fixed
set of 1 529 text prompts; a server-side evaluation script scores the
submissions against hidden reference data.  No reference audio is ever released.

---

## Leaderboard

*Last updated: 2026-05-12 · Human MOS scores updated monthly*

> ↑ = higher is better · ↓ = lower is better · — = pending human evaluation

| Rank | Team | System | UTMOS ↑ | WER ↓ | SNR ↑ (dB) | SpkSim ↑ | F0Corr ↑ | RTF ↓ | Human MOS ↑ | Date |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| — | *Baseline* | *VITS (10 h Wolof)* | *3.21* | *0.38* | *18.4* | *0.71* | *0.62* | *0.09* | *3.1 ± 0.5* | 2026-01-15 |

*The leaderboard will be populated once the first external submissions are received.*

To add your result, follow the [submission instructions](#submitting).

---

## Metrics

### Automated (instant)

| Metric | Tool | What it measures |
|---|---|---|
| **UTMOS** ↑ | [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | Predicted naturalness (proxy MOS, 1–5 scale) |
| **WER** ↓ | Whisper (`CAYTU/whosper-large-v2`) + jiwer | Intelligibility — how well ASR can transcribe the synthesis |
| **SNR** ↑ | SpeechBrain SepFormer | Signal cleanliness (dB); higher = less noise |
| **SpkSim** ↑ | ECAPA-TDNN cosine similarity | How close the synthesised voice is to the reference speaker |
| **F0Corr** ↑ | pYIN + Pearson histogram correlation | How well pitch dynamics match the reference recording |
| **RTF** ↓ | Self-reported | Real-Time Factor; speed of synthesis |

### Human (monthly)

Top-5 systems on automated metrics are evaluated monthly by a panel of **native
Wolof speakers** using CMOS (Comparative Mean Opinion Score, scale −3 to +3
relative to the baseline).  The absolute MOS shown in the leaderboard is
computed from a separate MOS round run at each quarterly release.

---

## Wolof-specific design choices

- **Phonetic coverage**: the 1 529 test sentences were selected to maximise
  coverage of Wolof phonemes, including ejectives (ƴ, ñ), prenasalised
  consonants, the ATR vowel distinction, and tonal patterns.
- **Code-switching**: ~8 % of sentences contain French loanwords, reflecting
  natural Dakar Wolof speech.
- **ASR proxy**: `CAYTU/whosper-large-v2` is a fine-tuned Whisper model for
  Wolof.  Because ASR coverage of Wolof is still limited, WER should be
  interpreted as an approximation; human MOS is the primary quality signal.
- **Speaker identity**: all reference recordings come from a single professional
  native speaker.  SpkSim measures voice consistency with that speaker.

---

## Repository layout

```
kallaamabench/
├── evaluate.py              # Official evaluation script (this repo)
├── SUBMISSION_FORMAT.md     # How to format your submission
├── README.md                # This file
│
├── data/
│   ├── corpus.txt           # 1 529 Wolof sentences (one per line)
│   ├── references.npz       # Precomputed speaker embeddings + F0 stats
│   └── rtf_scores.json      # Optional RTF records by model name
│
└── dev/
    ├── corpus_dev.txt        # 100-sentence dev subset
    ├── audio/                # Reference audio for dev set (WAV, 16 kHz mono)
    └── references_dev.npz    # Precomputed vectors for dev set
```

> **The `data/references.npz` file contains precomputed vectors only.**
> No raw test-set audio is distributed.  See [Data Policy](#data-policy).

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Synthesise the test sentences

```python
import torch, soundfile as sf
from your_tts_model import TTSModel

model = TTSModel.load("your_checkpoint")

with open("data/corpus.txt") as f:
    sentences = [l.strip() for l in f]

os.makedirs("my_submission/audio", exist_ok=True)
for i, text in enumerate(sentences, start=1):
    audio, sr = model.synthesise(text)
    sf.write(f"my_submission/audio/{i:04d}.wav", audio, sr)
```

### 3. Create `metadata.json`

```json
{
    "team"            : "MyTeam",
    "system"          : "FastSpeech2 + HiFi-GAN, 20 h Wolof",
    "submission_date" : "2026-05-12",
    "rtf"             : 0.038,
    "contact"         : "me@example.com"
}
```

### 4. Validate locally on the dev set

```bash
python evaluate.py evaluate \
    --submission  my_submission/ \
    --references  dev/references_dev.npz \
    --corpus      dev/corpus_dev.txt \
    --output      dev_results.json

cat dev_results.json
```

### 5. Submit

```bash
zip -r my_submission.zip my_submission/
```

Open a **New Submission** issue in this repository and attach the archive.
See [SUBMISSION_FORMAT.md](SUBMISSION_FORMAT.md) for full details.

---

## Submitting

1. Read [SUBMISSION_FORMAT.md](SUBMISSION_FORMAT.md) carefully.
2. Validate your submission locally on the dev set (step 4 above).
3. Open a GitHub Issue using the **"New Submission"** template.
4. Attach `my_submission.zip` (≤ 2 GB).
5. The evaluation bot will score your submission within **24 hours** and open
   a PR to update the leaderboard.

**Limits:** 3 submissions per team per 30-day window.

---

## Precomputing reference vectors (organisers only)

If the reference audio changes (e.g. a new split), regenerate `references.npz`:

```bash
python evaluate.py precompute \
    --audio-folder  /path/to/reference_audio/ \
    --corpus        data/corpus.txt \
    --output-npz    data/references.npz
```

This extracts ECAPA-TDNN speaker embeddings and pYIN F0 histograms for all
1 529 samples and stores them in a compressed NumPy archive.  Participants
never need to run this command.

---

## Data policy

- The **test set audio** (1 529 reference recordings) is **never released**.
- `data/corpus.txt` (text prompts) and `data/references.npz` (precomputed
  vectors) are public.
- `references.npz` contains compressed speaker embeddings (192-dim float32)
  and F0 histogram statistics.  The raw waveforms cannot be reconstructed
  from these vectors.
- Submissions are stored privately and used solely for scoring.

---

## Citation

If you use KallaamaBench in your research, please cite:

```bibtex
@misc{kallaamabench2026,
  title   = {KallaamaBench: A TTS Evaluation Benchmark for Wolof},
  author  = {Your Name and Collaborators},
  year    = {2026},
  url     = {https://github.com/yourorg/kallaamabench}
}
```

---

## Contact

Open a GitHub Issue for questions about the benchmark protocol, evaluation
script, or submission process.

---

## Acknowledgements

KallaamaBench builds on [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2),
[SpeechBrain](https://speechbrain.github.io/),
[Transformers](https://github.com/huggingface/transformers), and
[librosa](https://librosa.org/).  The Wolof ASR model (`CAYTU/whosper-large-v2`)
is used with permission from its authors.
