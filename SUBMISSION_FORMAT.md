# KallaamaBench — Submission Format

---

## Folder structure

```
my_submission/
├── metadata.json
└── audio/
    ├── audio_00000.wav
    ├── audio_00001.wav
    └── … (1602 files total)
```

---

## `metadata.json`

```json
{
    "team"            : "YourTeamName",
    "system"          : "Short description of your TTS system",
    "submission_date" : "YYYY-MM-DD",
    "contact"         : "yourname@example.com",
    "rtf"             : 0.042,
    "notes"           : "optional — hardware, training data, etc."
}
```

Required fields: `team`, `system`, `submission_date`.

---

## Audio files

| Property | Requirement |
|---|---|
| Format | WAV (16-bit PCM) |
| Sample rate | 16 000 Hz |
| Channels | Mono |
| Naming | `audio_xxxx.wav` |
| Count | Exactly **1602** files |

Each file must correspond to the matching row in `corpus.csv` by filename.
Misaligned files will silently produce incorrect WER and F0 scores.

```bash
# Convert to 16-bit PCM WAV with ffmpeg
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

---

## Submitting

```bash
zip -r my_submission.zip my_submission/
```

Open a **New Submission** issue. Upload your zip file submission to Google Drive and attach the public URL. Results within three days.
**Limit:** 3 submissions per team per 30-day window.
