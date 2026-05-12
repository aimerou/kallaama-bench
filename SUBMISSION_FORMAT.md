# KallaamaBench — Submission Format

This document describes the exact folder structure and file conventions
required to submit to **KallaamaBench**, the first TTS benchmark for the
Wolof language (1 529 test samples).

---

## Folder structure

```
my_submission/
├── metadata.json
└── audio/
    ├── 0001.wav
    ├── 0002.wav
    ├── …
    └── 1529.wav
```

Every submission **must** contain exactly these two things: a `metadata.json`
file at the root and an `audio/` subfolder with exactly **1 529 `.wav` files**.

---

## `metadata.json`

```json
{
    "team"             : "YourTeamName",
    "system"           : "Short description of your TTS system",
    "submission_date"  : "YYYY-MM-DD",
    "rtf"              : 0.042,
    "contact"          : "yourname@example.com",
    "language_model"   : "optional — LM used for TTS, e.g. none",
    "acoustic_model"   : "optional — e.g. FastSpeech2",
    "vocoder"          : "optional — e.g. HiFi-GAN",
    "training_data_h"  : 10.5,
    "notes"            : "optional free-text notes about your system"
}
```

### Required fields

| Field | Type | Description |
|---|---|---|
| `team` | string | Your team or organisation name |
| `system` | string | One-line description of the TTS system |
| `submission_date` | string | ISO 8601 date `YYYY-MM-DD` |
| `rtf` | float | Real-Time Factor measured on your hardware. See RTF section below. |
| `contact` | string | Contact e-mail of the submitting author |

### Optional fields

All other fields are optional but strongly encouraged to support reproducibility
and to appear in the leaderboard system column.

---

## Audio files

| Property | Requirement |
|---|---|
| Format | 16-bit PCM WAV (`.wav`) |
| Sample rate | 16 000 Hz |
| Channels | Mono (1 channel) |
| Naming | Zero-padded 4-digit index matching the corpus line number: `0001.wav` … `1529.wav` |
| Count | Exactly **1 529** files |

> **Important:** file `0001.wav` must correspond to line 1 of `corpus.txt`,
> `0002.wav` to line 2, and so on.  Misaligned files will produce incorrect
> WER and F0 scores without any warning from the evaluation script.

### Converting audio to the required format

```bash
# Using ffmpeg (recommended)
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav

# Using sox
sox input.wav -r 16000 -c 1 -b 16 output.wav
```

---

## RTF (Real-Time Factor)

RTF = synthesis\_time / audio\_duration.

You are responsible for measuring RTF on your own hardware and reporting it
honestly in `metadata.json`.  Please also describe your hardware in `notes`
(e.g. `"notes": "RTF measured on a single NVIDIA A100 40 GB"`).

A helper script is provided to measure RTF if you generate audio through a
Python function:

```python
import time, librosa

def measure_rtf(synthesise_fn, text_list):
    """synthesise_fn(text) -> (np.ndarray, sample_rate)"""
    total_synth_time = 0.0
    total_audio_dur  = 0.0
    for text in text_list:
        t0 = time.perf_counter()
        audio, sr = synthesise_fn(text)
        total_synth_time += time.perf_counter() - t0
        total_audio_dur  += len(audio) / sr
    return total_synth_time / total_audio_dur
```

---

## Submission limits

- Maximum **3 submissions per team per rolling 30-day window**.
- Each submission must correspond to a *distinct* system configuration.
  Submitting the same system with minor post-processing tweaks to game the
  leaderboard is not permitted.
- Teams that reach the top-5 on automated metrics will be contacted for a
  **human MOS evaluation** round.

---

## Submitting

Package your submission as a `.zip` archive:

```bash
zip -r my_submission.zip my_submission/
```

Then open a GitHub Issue in this repository using the **"New Submission"**
template and attach the archive.  The evaluation bot will process your
submission within 24 hours and open a PR to update the leaderboard.

---

## Dev set

A dev set of **100 samples** (audio + transcripts) is available under `dev/`
in this repository.  Use it to sanity-check your pipeline before submitting.

```
dev/
├── corpus_dev.txt       # 100 reference transcripts
├── audio/               # 100 reference WAV files (same format as test set)
└── references_dev.npz   # Precomputed speaker embeddings + F0 stats
```

Run the evaluation script on the dev set locally before submitting:

```bash
python evaluate.py evaluate \
    --submission  path/to/my_dev_submission/ \
    --references  dev/references_dev.npz \
    --corpus      dev/corpus_dev.txt \
    --output      dev_results.json
```

---

## Data policy

The **test set audio** (reference recordings) is never released.  Only the
text prompts (`data/corpus.txt`) and the precomputed reference vectors
(`data/references.npz`) are public.  This design prevents systems from
overfitting directly to the reference waveforms.

The precomputed vectors contain:
- Speaker embeddings (ECAPA-TDNN, 192-dim)
- F0 distribution statistics (mean, std, 50-bin histogram with bin edges)

No raw audio from the test set can be reconstructed from these vectors.
