#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "librosa",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///
"""
TTS Dataset Clustering: Morphology + F0 + Duration Features
============================================================
Extracts morphological, F0, and duration features for every sample in a TTS
dataset, runs K-Means (K=6) per speaker, saves per-speaker cluster plots, and
writes the medoid of each cluster to a JSON file.

Dataset structure expected
--------------------------
  data/metadata.csv   – pipe-separated: filename | transcription | speaker | gender
  data/audio/         – audio files (wav / mp3 / …)

Usage
-----
uv run cluster_tts_samples.py \
    --metadata  ../../data/corpus.csv \
    --audiodir  ../../data/audio \
    --outdir    ../../data/mos-evaluation/clustering/ \
    [--n-clusters 6] \
    [--delimiter "|"]
"""

import argparse
import csv
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ── Morphological feature definitions (from counter_morphology.py) ────────────

GEMINATE_CONSONANTS = [
    "pp", "tt", "cc", "kk", "bb", "dd", "jj", "gg",
    "mm", "nn", "ññ", "ŋŋ", "ww", "ll", "yy",
]

PRENASALISED_CONSONANTS = [
    "mp", "nt", "nc", "nk", "nq", "mb", "nd", "nj", "ng",
]

# Long vowels checked before ATR vowels to avoid double-counting
LONG_VOWELS = ["ii", "uu", "éé", "óó", "ee", "oo", "aa"]

# ATR vowels: single occurrences NOT part of a long-vowel digraph
ATR_VOWELS = ["é", "ó", "ë"]


# ── Morphology helpers ────────────────────────────────────────────────────────

def count_non_overlapping(text: str, pattern: str) -> int:
    return len(re.findall(re.escape(pattern), text, flags=re.IGNORECASE))


def count_atr_vowels(text: str) -> dict:
    masked = text
    for lv in ["éé", "óó"]:
        masked = masked.replace(lv, "XX")
    return {v: count_non_overlapping(masked, v) for v in ATR_VOWELS}


def extract_morphology(sentence: str) -> dict:
    gem   = {g: count_non_overlapping(sentence, g) for g in GEMINATE_CONSONANTS}
    pre   = {p: count_non_overlapping(sentence, p) for p in PRENASALISED_CONSONANTS}
    lv    = {v: count_non_overlapping(sentence, v) for v in LONG_VOWELS}
    atr   = count_atr_vowels(sentence)
    total = sum(gem.values()) + sum(pre.values()) + sum(lv.values()) + sum(atr.values())
    return {
        "geminate_consonants":    gem,
        "prenasalised_consonants": pre,
        "long_vowels":            lv,
        "atr_vowels":             atr,
        "total":                  total,
        # Aggregate counts used as features
        "n_geminates":    sum(gem.values()),
        "n_prenasalised": sum(pre.values()),
        "n_long_vowels":  sum(lv.values()),
        "n_atr_vowels":   sum(atr.values()),
    }


# ── Audio feature extraction ──────────────────────────────────────────────────

def extract_audio_features(audio_path: Path) -> dict | None:
    """
    Returns:
        duration_s      – total duration in seconds
        f0_mean         – mean F0 (Hz) over voiced frames
        f0_std          – std  F0 (Hz) over voiced frames
        f0_min          – min  F0 (Hz) over voiced frames
        f0_max          – max  F0 (Hz) over voiced frames
        voiced_fraction – fraction of frames that are voiced
    Returns None if the file cannot be loaded.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:
        print(f"  [WARN] Cannot load {audio_path.name}: {exc}", file=sys.stderr)
        return None

    duration_s = librosa.get_duration(y=y, sr=sr)

    # F0 via pyin (probabilistic YIN) – returns f0 in Hz, voiced flag
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    except Exception:
        voiced_f0 = np.array([])

    if voiced_f0.size > 0:
        f0_mean = float(np.mean(voiced_f0))
        f0_std  = float(np.std(voiced_f0))
        f0_min  = float(np.min(voiced_f0))
        f0_max  = float(np.max(voiced_f0))
    else:
        f0_mean = f0_std = f0_min = f0_max = 0.0

    total_frames   = f0.shape[0] if "f0" in dir() and f0 is not None else 1
    voiced_fraction = float(voiced_f0.size / max(total_frames, 1))

    return {
        "duration_s":       duration_s,
        "f0_mean":          f0_mean,
        "f0_std":           f0_std,
        "f0_min":           f0_min,
        "f0_max":           f0_max,
        "voiced_fraction":  voiced_fraction,
    }


# ── Feature vector assembly ───────────────────────────────────────────────────

FEATURE_KEYS = [
    # morphological
    "n_geminates", "n_prenasalised", "n_long_vowels", "n_atr_vowels",
    # acoustic
    "duration_s", "f0_mean", "f0_std", "f0_min", "f0_max", "voiced_fraction",
]


def build_feature_vector(morph: dict, audio: dict) -> np.ndarray:
    merged = {**morph, **audio}
    return np.array([merged.get(k, 0.0) for k in FEATURE_KEYS], dtype=float)


# ── Medoid selection ──────────────────────────────────────────────────────────

def find_medoid(X: np.ndarray, labels: np.ndarray, cluster_id: int) -> int:
    """Return the index (into X) of the medoid of a cluster."""
    mask = labels == cluster_id
    indices = np.where(mask)[0]
    cluster_X = X[mask]
    centroid = cluster_X.mean(axis=0)
    dists = np.linalg.norm(cluster_X - centroid, axis=1)
    return int(indices[np.argmin(dists)])


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    medoid_indices: list[int],
    speaker: str,
    n_clusters: int,
    out_path: Path,
) -> None:
    """2-D PCA projection coloured by cluster; medoids marked with a star."""
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)

    ev = pca.explained_variance_ratio_
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            color=cmap(k), alpha=0.65, s=40, label=f"Cluster {k}",
        )

    # Highlight medoids
    for k, idx in enumerate(medoid_indices):
        ax.scatter(
            X2[idx, 0], X2[idx, 1],
            color=cmap(k), marker="*", s=260, edgecolors="black", linewidths=0.8,
            zorder=5,
        )

    ax.set_title(f"Speaker: {speaker}  –  K-Means (K={n_clusters})\n"
                 f"PCA explains {ev[0]:.1%} + {ev[1]:.1%} of variance")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best", fontsize=8, markerscale=0.8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster TTS samples by morphology + F0 + duration features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--metadata",    type=Path, required=True,
                   help="Path to metadata CSV (pipe-separated).")
    p.add_argument("--audiodir",    type=Path, required=True,
                   help="Directory containing audio files.")
    p.add_argument("--outdir",      type=Path, required=True,
                   help="Output directory for plots and JSON.")
    p.add_argument("--n-clusters",  type=int, default=6,
                   help="Number of K-Means clusters per speaker (default: 6).")
    p.add_argument("--delimiter",   default="|",
                   help="CSV delimiter (default: pipe |).")
    p.add_argument("--encoding",    default="utf-8")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    K = args.n_clusters

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    if not args.metadata.exists():
        sys.exit(f"Error: metadata file not found – {args.metadata}")

    rows: list[dict] = []
    with args.metadata.open(encoding=args.encoding, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=args.delimiter)
        required = {"filename", "transcription", "speaker", "gender"}
        missing  = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"Missing columns in CSV: {', '.join(sorted(missing))}\n"
                     f"Found: {', '.join(reader.fieldnames or [])}")
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {args.metadata}")

    # ── 2. Extract features ───────────────────────────────────────────────────
    samples: list[dict] = []
    total = len(rows)
    for i, row in enumerate(rows):
        filename     = row["filename"].strip()
        transcription = row["transcription"].strip()
        speaker      = row["speaker"].strip()
        gender       = row["gender"].strip()

        # Audio file – try the filename as-is, then with common extensions
        audio_path = args.audiodir / filename
        if not audio_path.exists():
            for ext in (".wav", ".mp3", ".flac", ".ogg"):
                candidate = args.audiodir / (Path(filename).stem + ext)
                if candidate.exists():
                    audio_path = candidate
                    break

        print(f"  [{i+1}/{total}] {filename}", end="  ", flush=True)

        morph = extract_morphology(transcription)
        audio = extract_audio_features(audio_path) if audio_path.exists() else None
        if audio is None:
            print("[audio missing – zeros used]")
            audio = {k: 0.0 for k in
                     ("duration_s", "f0_mean", "f0_std", "f0_min", "f0_max", "voiced_fraction")}
        else:
            print(f"dur={audio['duration_s']:.2f}s  f0={audio['f0_mean']:.1f}Hz")

        samples.append({
            "filename":      filename,
            "transcription": transcription,
            "speaker":       speaker,
            "gender":        gender,
            "morphology":    morph,
            "audio":         audio,
            "feature_vector": build_feature_vector(morph, audio).tolist(),
        })

    # ── 3. Group by speaker ───────────────────────────────────────────────────
    by_speaker: dict[str, list[dict]] = {}
    for s in samples:
        by_speaker.setdefault(s["speaker"], []).append(s)

    speakers = sorted(by_speaker.keys())
    print(f"\nFound {len(speakers)} speaker(s): {', '.join(speakers)}")

    # ── 4. K-Means per speaker ────────────────────────────────────────────────
    all_medoids: list[dict] = []

    for speaker in speakers:
        sp_samples = by_speaker[speaker]
        n = len(sp_samples)
        k = min(K, n)          # guard: can't have more clusters than samples
        print(f"\n── Speaker '{speaker}' ({n} samples, K={k}) ──")

        X = np.array([s["feature_vector"] for s in sp_samples])

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # Medoid per cluster
        medoid_indices: list[int] = []
        for cluster_id in range(k):
            med_idx = find_medoid(X_scaled, labels, cluster_id)
            medoid_indices.append(med_idx)
            med_sample = sp_samples[med_idx]
            print(f"  Cluster {cluster_id}: medoid → {med_sample['filename']}")

            all_medoids.append({
                "speaker":       speaker,
                "gender":        med_sample["gender"],
                "cluster":       int(cluster_id),
                "filename":      med_sample["filename"],
                "transcription": med_sample["transcription"],
                "morphology":    med_sample["morphology"],
                "audio":         med_sample["audio"],
            })

        # Attach cluster labels back to samples (for reference)
        for idx, s in enumerate(sp_samples):
            s["cluster"] = int(labels[idx])
            s["is_medoid"] = (idx in medoid_indices)

        # Plot
        plot_path = plots_dir / f"clusters_{speaker}.png"
        plot_clusters(X_scaled, labels, medoid_indices, speaker, k, plot_path)
        print(f"  Plot saved → {plot_path}")

    # ── 5. Save medoids JSON ──────────────────────────────────────────────────
    json_path = args.outdir / "medoids.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(all_medoids, fh, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(all_medoids)} medoid(s) saved → {json_path}")
    print(f"✓ Cluster plots saved in  → {plots_dir}/")


if __name__ == "__main__":
    main()