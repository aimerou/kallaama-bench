"""
KallaamaBench — Official Evaluation Script
==========================================
Metrics: UTMOS · WER · SNR · RTF · Speaker Similarity · F0 Correlation

Usage
-----
python evaluate.py \
    --submission path/to/submission/ \
    --references data/references.npz \
    --corpus     data/corpus.txt \
    --rtf-json   data/rtf_scores.json \
    --output     results.json

The script expects the submission folder to follow the layout described in
SUBMISSION_FORMAT.md.  It scores submitted audio against:
  - data/references.npz   : precomputed speaker embeddings + F0 statistics
  - data/corpus.txt        : reference transcripts (one per line, 1529 lines)
"""

import os
import json
import argparse
import warnings
import torch
import librosa
import numpy as np
import utmosv2
from jiwer import wer
from scipy.stats import pearsonr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.inference.speaker import EncoderClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WOLOF_ASR_REPO   = "CAYTU/whosper-large-v2"
DENOISER_NAME    = "speechbrain/sepformer-wham16k-enhancement"
SPKR_MODEL_NAME  = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SR        = 16_000
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_models():
    """Load all models once and return them in a dict."""
    print("[KallaamaBench] Loading models …")

    utmos_model = utmosv2.create_model(pretrained=True)

    asr_processor = WhisperProcessor.from_pretrained(WOLOF_ASR_REPO)
    asr_model     = WhisperForConditionalGeneration.from_pretrained(WOLOF_ASR_REPO).to(DEVICE)

    denoiser = separator.from_hparams(
        source=DENOISER_NAME, run_opts={"device": str(DEVICE)}
    )

    spkr_encoder = EncoderClassifier.from_hparams(
        source=SPKR_MODEL_NAME, run_opts={"device": str(DEVICE)}
    )

    return dict(
        utmos=utmos_model,
        asr_processor=asr_processor,
        asr_model=asr_model,
        denoiser=denoiser,
        spkr_encoder=spkr_encoder,
    )


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_wer(audio: np.ndarray, sr: int, reference: str, models: dict) -> float:
    """ASR-based Word Error Rate."""
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    inputs    = models["asr_processor"](audio_16k, sampling_rate=TARGET_SR, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = models["asr_model"].generate(**inputs)
    hypothesis = models["asr_processor"].batch_decode(ids, skip_special_tokens=True)[0]
    return wer(reference, hypothesis)


def _snr_power(signal: np.ndarray, noise: np.ndarray) -> float:
    n      = min(len(signal), len(noise))
    s_pwr  = np.mean(signal[:n] ** 2)
    n_pwr  = np.mean(noise[:n]  ** 2)
    return 10 * np.log10(s_pwr / n_pwr) if n_pwr > 0 else 100.0


def compute_snr(audio: np.ndarray, models: dict) -> float:
    """Signal-to-Noise Ratio via SepFormer source separation."""
    t = torch.tensor(audio).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        result          = models["denoiser"].separate_batch(t)
        noise_estimate  = result.permute(0, 2, 1)[0][0]
    return _snr_power(audio, noise_estimate.cpu().numpy())


def compute_speaker_similarity(audio: np.ndarray, sr: int,
                                ref_embedding: np.ndarray, models: dict) -> float:
    """
    Cosine similarity between the submitted sample's speaker embedding and the
    reference embedding pre-extracted from the KallaamaBench dataset.

    Returns a value in [-1, 1]; higher is more similar.
    """
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    t         = torch.tensor(audio_16k).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        emb = models["spkr_encoder"].encode_batch(t).squeeze().cpu().numpy()

    # Cosine similarity
    num   = np.dot(emb, ref_embedding)
    denom = (np.linalg.norm(emb) * np.linalg.norm(ref_embedding)) + 1e-9
    return float(num / denom)


def extract_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract F0 contour using pYIN.  Voiced frames only; returns array of Hz values.
    Unvoiced frames are dropped so that the correlation is computed on voiced regions.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    # Keep only voiced frames that are not NaN
    voiced = f0[voiced_flag & ~np.isnan(f0)]
    return voiced if len(voiced) > 0 else np.array([0.0])


def compute_f0_correlation(audio: np.ndarray, sr: int,
                            ref_f0_stats: dict) -> float:
    """
    Pearson correlation between the submitted sample's F0 sequence and the
    reference F0 sequence stored in references.npz.

    Because the two sequences will generally differ in length, we compare
    their distributions by computing a histogram correlation (50 bins over
    the common Hz range) — a standard approach when alignment is unavailable.

    Returns a value in [-1, 1].
    """
    hyp_f0 = extract_f0(audio, sr)

    ref_mean  = ref_f0_stats["mean"]
    ref_std   = ref_f0_stats["std"]
    ref_hist  = ref_f0_stats["hist"]   # pre-computed 50-bin histogram
    ref_edges = ref_f0_stats["edges"]  # bin edges

    if len(hyp_f0) < 2:
        return 0.0

    hyp_hist, _ = np.histogram(hyp_f0, bins=ref_edges)
    hyp_hist    = hyp_hist.astype(float)

    # Normalise both histograms to probability distributions
    ref_hist_norm = ref_hist / (ref_hist.sum() + 1e-9)
    hyp_hist_norm = hyp_hist / (hyp_hist.sum() + 1e-9)

    corr, _ = pearsonr(ref_hist_norm, hyp_hist_norm)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_utmos(audio_path: str, models: dict) -> float:
    return float(models["utmos"].predict(input_path=audio_path))


# ---------------------------------------------------------------------------
# Reference data helpers
# ---------------------------------------------------------------------------

def load_references(references_npz: str) -> dict:
    """
    Load the precomputed reference vectors from references.npz.

    Expected keys
    -------------
    speaker_embeddings : float32 array of shape (N, D)
        ECAPA-TDNN speaker embeddings, one per sample.
    f0_means           : float32 array of shape (N,)
    f0_stds            : float32 array of shape (N,)
    f0_hists           : float32 array of shape (N, 50)
    f0_edges           : float32 array of shape (N, 51)
    """
    data = np.load(references_npz)
    return {
        "speaker_embeddings": data["speaker_embeddings"],
        "f0_means"          : data["f0_means"],
        "f0_stds"           : data["f0_stds"],
        "f0_hists"          : data["f0_hists"],
        "f0_edges"          : data["f0_edges"],
    }


# ---------------------------------------------------------------------------
# Precompute reference vectors (utility — run once by benchmark organisers)
# ---------------------------------------------------------------------------

def precompute_references(audio_folder: str, corpus_file: str,
                           output_npz: str, spkr_model_name: str = SPKR_MODEL_NAME):
    """
    Precompute and save speaker embeddings and F0 statistics for all reference
    audio files.  Run once by the benchmark organisers; participants never
    call this function.

    Usage
    -----
        python evaluate.py --precompute \\
            --audio-folder data/reference_audio/ \\
            --corpus       data/corpus.txt \\
            --output-npz   data/references.npz
    """
    print("[KallaamaBench] Precomputing reference vectors …")

    spkr_encoder = EncoderClassifier.from_hparams(
        source=spkr_model_name, run_opts={"device": str(DEVICE)}
    )

    with open(corpus_file, encoding="utf-8") as f:
        sentences = [l.strip() for l in f.readlines()]

    audio_files = sorted([fn for fn in os.listdir(audio_folder) if fn.endswith(".wav")])
    assert len(audio_files) == len(sentences), (
        f"Mismatch: {len(audio_files)} audio files vs {len(sentences)} sentences"
    )

    spk_embs, f0_means, f0_stds, f0_hists, f0_edges_all = [], [], [], [], []

    N_BINS   = 50
    F0_MIN   = librosa.note_to_hz("C2")
    F0_MAX   = librosa.note_to_hz("C7")
    bin_edges = np.linspace(F0_MIN, F0_MAX, N_BINS + 1)

    for i, fn in enumerate(audio_files):
        path            = os.path.join(audio_folder, fn)
        audio, sr       = librosa.load(path, sr=None)
        audio_16k       = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        # Speaker embedding
        t   = torch.tensor(audio_16k).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            emb = spkr_encoder.encode_batch(t).squeeze().cpu().numpy()
        spk_embs.append(emb)

        # F0
        f0_voiced = extract_f0(audio_16k, TARGET_SR)
        f0_means.append(float(np.mean(f0_voiced)))
        f0_stds.append(float(np.std(f0_voiced)))
        hist, _ = np.histogram(f0_voiced, bins=bin_edges)
        f0_hists.append(hist.astype(np.float32))
        f0_edges_all.append(bin_edges.astype(np.float32))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(audio_files)} done")

    np.savez_compressed(
        output_npz,
        speaker_embeddings = np.stack(spk_embs).astype(np.float32),
        f0_means           = np.array(f0_means,        dtype=np.float32),
        f0_stds            = np.array(f0_stds,         dtype=np.float32),
        f0_hists           = np.stack(f0_hists).astype(np.float32),
        f0_edges           = np.stack(f0_edges_all).astype(np.float32),
    )
    print(f"[KallaamaBench] Saved reference vectors → {output_npz}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(submission_dir: str, references_npz: str, corpus_file: str,
             rtf_json: str, output_json: str):

    # -- Load metadata
    meta_path = os.path.join(submission_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    team_name  = meta["team"]
    model_name = meta.get("system", "unknown")
    print(f"[KallaamaBench] Evaluating submission: {team_name} / {model_name}")

    # -- Load corpus
    with open(corpus_file, encoding="utf-8") as f:
        sentences = [l.strip() for l in f.readlines()]

    # -- Load references
    refs = load_references(references_npz)

    # -- Load RTF (if provided in submission)
    rtf = meta.get("rtf", None)
    if rtf is None and rtf_json and os.path.exists(rtf_json):
        with open(rtf_json) as f:
            rtf_data = json.load(f)
        for item in rtf_data:
            if item.get("model_name") == model_name:
                rtf = item["rtf"]
                break

    # -- Load audio files
    audio_dir  = os.path.join(submission_dir, "audio")
    audio_files = sorted([fn for fn in os.listdir(audio_dir) if fn.endswith(".wav")])

    assert len(audio_files) == len(sentences), (
        f"Expected {len(sentences)} audio files, got {len(audio_files)}"
    )

    # -- Load models
    models = load_models()

    # -- Score loop
    utmos_scores, wer_scores, snr_scores, spk_sim_scores, f0_corr_scores = \
        [], [], [], [], []

    for i, fn in enumerate(audio_files):
        path        = os.path.join(audio_dir, fn)
        audio, sr   = librosa.load(path, sr=None)
        reference   = sentences[i]

        utmos = compute_utmos(path, models)
        wer_  = compute_wer(audio, sr, reference, models)
        snr   = compute_snr(audio, models)
        spk   = compute_speaker_similarity(
            audio, sr,
            ref_embedding=refs["speaker_embeddings"][i],
            models=models,
        )
        f0c = compute_f0_correlation(
            audio, sr,
            ref_f0_stats={
                "mean" : refs["f0_means"][i],
                "std"  : refs["f0_stds"][i],
                "hist" : refs["f0_hists"][i],
                "edges": refs["f0_edges"][i],
            },
        )

        utmos_scores.append(utmos)
        wer_scores.append(wer_)
        snr_scores.append(snr)
        spk_sim_scores.append(spk)
        f0_corr_scores.append(f0c)

        print(
            f"[{i+1:04d}/{len(audio_files)}] {fn} | "
            f"UTMOS={utmos:.3f}  WER={wer_:.3f}  SNR={snr:.1f}dB  "
            f"SpkSim={spk:.3f}  F0Corr={f0c:.3f}"
        )

    # -- Aggregate
    def _mean(lst): return float(np.mean(lst)) if lst else 0.0

    result = {
        "team"               : team_name,
        "system"             : model_name,
        "submission_date"    : meta.get("submission_date", ""),
        "n_samples"          : len(audio_files),
        "UTMOS"              : _mean(utmos_scores),
        "WER"                : _mean(wer_scores),
        "SNR_dB"             : _mean(snr_scores),
        "SpeakerSimilarity"  : _mean(spk_sim_scores),
        "F0Correlation"      : _mean(f0_corr_scores),
        "RTF"                : rtf,
    }

    # -- Append to output file
    existing = []
    if os.path.exists(output_json):
        with open(output_json) as f:
            existing = json.load(f)
    existing.append(result)
    with open(output_json, "w") as f:
        json.dump(existing, f, indent=4)

    print("\n[KallaamaBench] ── Final scores ──────────────────────────────────")
    for k, v in result.items():
        print(f"  {k:<22}: {v}")
    print(f"[KallaamaBench] Saved → {output_json}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="KallaamaBench evaluation script")
    sub = p.add_subparsers(dest="command")

    # evaluate subcommand
    ev = sub.add_parser("evaluate", help="Score a participant submission")
    ev.add_argument("--submission",  required=True, help="Path to submission folder")
    ev.add_argument("--references",  required=True, help="Path to references.npz")
    ev.add_argument("--corpus",      required=True, help="Path to corpus.txt")
    ev.add_argument("--rtf-json",    default=None,  help="Optional external RTF JSON")
    ev.add_argument("--output",      default="results.json", help="Output JSON path")

    # precompute subcommand (organisers only)
    pc = sub.add_parser("precompute", help="Precompute reference vectors (organisers only)")
    pc.add_argument("--audio-folder", required=True)
    pc.add_argument("--corpus",       required=True)
    pc.add_argument("--output-npz",   default="data/references.npz")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "evaluate":
        evaluate(
            submission_dir = args.submission,
            references_npz = args.references,
            corpus_file    = args.corpus,
            rtf_json       = args.rtf_json,
            output_json    = args.output,
        )
    elif args.command == "precompute":
        precompute_references(
            audio_folder = args.audio_folder,
            corpus_file  = args.corpus,
            output_npz   = args.output_npz,
        )
    else:
        print("Use 'evaluate' or 'precompute'. Run with --help for details.")
