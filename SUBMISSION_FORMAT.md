# KallaamaBench — Submission Format

---

## Folder structure

```
my_submission/
├── metadata.json
└── audio/
    ├── audio_0001.wav
    ├── audio_0002.wav
    └── … (1 529 files total)
```

---

## `metadata.json`

```json
{
    "team"            : "YourTeamName",
    "system"          : "Short description of your TTS system",
    "submission_date" : "YYYY-MM-DD",
    "rtf"             : 0.042,
    "contact"         : "yourname@example.com",
    "notes"           : "optional — hardware, training data, etc."
}
```

Required fields: `team`, `system`, `submission_date`, `rtf`, `contact`.

---

## Audio files

| Property | Requirement |
|---|---|
| Format | WAV (16-bit PCM) |
| Sample rate | 16 000 Hz |
| Channels | Mono |
| Naming | `audio_xxxx.wav` |
| Count | Exactly **1 529** files |

Each file must correspond to the matching row in `corpus.csv` by filename.
Misaligned files will silently produce incorrect WER and F0 scores.

```bash
# Convert to 16-bit PCM WAV with ffmpeg
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

---

## RTF

RTF = synthesis_time / audio_duration. Measure on your own hardware and report
it in `metadata.json`. Include your hardware specs in `notes`.

---

## Dev set

Validate locally before submitting:

```bash
python evaluate.py evaluate \
    --submission  path/to/my_dev_submission/ \
    --references  dev/references_dev.npz \
    --corpus      dev/corpus_dev.csv \
    --output      dev_results.json
```

---

## Submitting

```bash
zip -r my_submission.zip my_submission/
```

Open a **New Submission** issue and attach the archive. Results within 24 hours.
**Limit:** 3 submissions per team per 30-day window.
