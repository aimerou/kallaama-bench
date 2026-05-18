"""
KallaamaBench — Official Evaluation Script
==========================================
Metrics: UTMOS · WER · SNR · RTF · Speaker Similarity · F0 Correlation

Usage
-----
python evaluate.py \\
    --submission path/to/submission/ \\
    --references data/references/ \\
    --corpus     data/corpus.csv \\
    --rtf-json   data/rtf_scores.json \\
    --output     results.json

The script expects the submission folder to follow the layout described in
SUBMISSION_FORMAT.md.  It scores submitted audio against:
  - data/references/          : folder of per-speaker WAV files (<speaker>.wav)
  - data/corpus.csv           : pipe-separated CSV with columns:
                                  filename | transcription | speaker | gender
"""

import gc
import os
import json
import shutil
import argparse
import tempfile
import warnings
import torch
import librosa
import torchaudio
import numpy as np
import pandas as pd
import utmosv2
from jiwer import wer
from scipy.stats import pearsonr
from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.inference.speaker import EncoderClassifier
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DENOISER_NAME    = "speechbrain/sepformer-wham16k-enhancement"
SPKR_MODEL_NAME  = "speechbrain/spkrec-ecapa-voxceleb"
ASR_MODEL_CARD   = "omnilingual/asr-wolof"   # override via --asr-model
LANGUAGE_CODE    = "wo"                       # override via --language-code
TARGET_SR        = 16_000
CHUNK_DURATION_S = 30                         # seconds per ASR chunk
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def free_gpu():
    """Release all cached GPU memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def iter_chunks(audio_array: np.ndarray, chunk_duration_s: int, sample_rate: int,
                min_chunk_s: float = 0.5):
    """Yield fixed-size chunks of an audio array, skipping very short trailing chunks."""
    chunk_size = chunk_duration_s * sample_rate
    min_samples = int(min_chunk_s * sample_rate)
    for start in range(0, len(audio_array), chunk_size):
        chunk = audio_array[start: start + chunk_size]
        if len(chunk) < min_samples:
            break
        yield start, chunk


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_corpus(corpus_csv: str) -> pd.DataFrame:
    """
    Load a pipe-separated CSV with columns: filename, transcription, speaker, gender.

    Returns a DataFrame indexed by filename (stem) for easy lookup.
    """
    df = pd.read_csv(
        corpus_csv,
        sep="|",
        header=0,
        on_bad_lines="skip",
        dtype=str,
    )

    required = {"filename", "transcription", "speaker", "gender"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus CSV is missing columns: {missing}")

    df = df.dropna(subset=["filename", "transcription", "speaker"])
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Reference audio helpers
# ---------------------------------------------------------------------------

def load_speaker_embedding(speaker: str, references_dir: str,
                            spkr_encoder: "EncoderClassifier") -> np.ndarray:
    """
    Load the WAV file for *speaker* from *references_dir* and return its
    ECAPA-TDNN embedding as a numpy array.

    The reference file is expected at:
        <references_dir>/<speaker>.wav
    """
    ref_path = os.path.join(references_dir, f"{speaker}.wav")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"No reference audio found for speaker '{speaker}' at {ref_path}"
        )

    audio, sr = librosa.load(ref_path, sr=None)
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    t = torch.tensor(audio_16k).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        emb = spkr_encoder.encode_batch(t).squeeze().cpu().numpy()
    return emb


def load_f0_stats_from_audio(speaker: str, references_dir: str) -> dict:
    """
    Compute F0 statistics on-the-fly from the reference WAV for *speaker*.

    Returns a dict with keys: mean, std, hist, edges  (matching the shape
    expected by compute_f0_correlation).
    """
    ref_path = os.path.join(references_dir, f"{speaker}.wav")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"No reference audio found for speaker '{speaker}' at {ref_path}"
        )

    audio, sr = librosa.load(ref_path, sr=None)
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    f0_voiced = extract_f0(audio_16k, TARGET_SR)

    N_BINS    = 50
    F0_MIN    = librosa.note_to_hz("C2")
    F0_MAX    = librosa.note_to_hz("C7")
    bin_edges = np.linspace(F0_MIN, F0_MAX, N_BINS + 1)

    hist, _ = np.histogram(f0_voiced, bins=bin_edges)

    return {
        "mean" : float(np.mean(f0_voiced)),
        "std"  : float(np.std(f0_voiced)),
        "hist" : hist.astype(np.float32),
        "edges": bin_edges.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(asr_model_card: str = ASR_MODEL_CARD) -> dict:
    """Load all models once and return them in a dict."""
    print("[KallaamaBench] Loading models …")

    utmos_model = utmosv2.create_model(pretrained=True)

    asr_pipeline = ASRInferencePipeline(model_card=asr_model_card, device=str(DEVICE))

    denoiser = separator.from_hparams(
        source=DENOISER_NAME, run_opts={"device": str(DEVICE)}
    )

    spkr_encoder = EncoderClassifier.from_hparams(
        source=SPKR_MODEL_NAME, run_opts={"device": str(DEVICE)}
    )

    return dict(
        utmos=utmos_model,
        asr_pipeline=asr_pipeline,
        denoiser=denoiser,
        spkr_encoder=spkr_encoder,
    )


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def transcribe_chunked(audio_array: np.ndarray, asr_pipeline, language_code: str,
                        audio_path_hint: str = "chunk.wav") -> str:
    """
    Transcribe audio by splitting into fixed-size chunks, writing each to a
    temp WAV, running the Omnilingual ASR pipeline, then joining hypotheses.
    """
    chunks = list(iter_chunks(audio_array, CHUNK_DURATION_S, TARGET_SR))
    if len(chunks) <= 1:
        chunks = [(0, audio_array)]

    tmp_dir    = tempfile.mkdtemp(prefix="asr_chunks_")
    hypotheses = []

    try:
        for idx, (_, chunk) in enumerate(chunks):
            tmp_path     = os.path.join(tmp_dir, f"chunk_{idx:04d}.wav")
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).float()
            torchaudio.save(tmp_path, chunk_tensor, TARGET_SR)

            try:
                hyp = asr_pipeline.transcribe([tmp_path], lang=[language_code])[0]
                hypotheses.append(hyp.strip())
            finally:
                os.remove(tmp_path)
                free_gpu()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return " ".join(hypotheses)


def compute_wer(audio: np.ndarray, sr: int, reference: str,
                models: dict, language_code: str) -> float:
    """ASR-based Word Error Rate using Omnilingual ASR."""
    audio_16k  = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    hypothesis = transcribe_chunked(audio_16k, models["asr_pipeline"], language_code)
    return wer(reference, hypothesis)


def _snr_power(signal: np.ndarray, noise: np.ndarray) -> float:
    n     = min(len(signal), len(noise))
    s_pwr = np.mean(signal[:n] ** 2)
    n_pwr = np.mean(noise[:n]  ** 2)
    return 10 * np.log10(s_pwr / n_pwr) if n_pwr > 0 else 100.0


def compute_snr(audio: np.ndarray, models: dict) -> float:
    """Signal-to-Noise Ratio via SepFormer source separation."""
    t = torch.tensor(audio).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        result         = models["denoiser"].separate_batch(t)
        noise_estimate = result.permute(0, 2, 1)[0][0]
    return _snr_power(audio, noise_estimate.cpu().numpy())


def compute_speaker_similarity(audio: np.ndarray, sr: int,
                                ref_embedding: np.ndarray, models: dict) -> float:
    """
    Cosine similarity between the submitted sample's speaker embedding and the
    reference embedding extracted from the per-speaker reference audio.

    Returns a value in [-1, 1]; higher is more similar.
    """
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    t         = torch.tensor(audio_16k).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        emb = models["spkr_encoder"].encode_batch(t).squeeze().cpu().numpy()

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
    voiced = f0[voiced_flag & ~np.isnan(f0)]
    return voiced if len(voiced) > 0 else np.array([0.0])


def compute_f0_correlation(audio: np.ndarray, sr: int, ref_f0_stats: dict) -> float:
    """
    Pearson correlation between the submitted sample's F0 histogram and the
    reference F0 histogram derived from the per-speaker reference audio.

    Returns a value in [-1, 1].
    """
    hyp_f0    = extract_f0(audio, sr)
    ref_hist  = ref_f0_stats["hist"]
    ref_edges = ref_f0_stats["edges"]

    if len(hyp_f0) < 2:
        return 0.0

    hyp_hist, _ = np.histogram(hyp_f0, bins=ref_edges)
    hyp_hist     = hyp_hist.astype(float)

    ref_hist_norm = ref_hist / (ref_hist.sum() + 1e-9)
    hyp_hist_norm = hyp_hist / (hyp_hist.sum() + 1e-9)

    corr, _ = pearsonr(ref_hist_norm, hyp_hist_norm)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_utmos(audio_path: str, models: dict) -> float:
    return float(models["utmos"].predict(input_path=audio_path))


# ---------------------------------------------------------------------------
# Speaker reference cache  (compute once per unique speaker)
# ---------------------------------------------------------------------------

class SpeakerReferenceCache:
    """
    Lazily computes and caches speaker embeddings and F0 stats from the
    per-speaker WAV files in *references_dir*.  Each unique speaker is
    processed at most once per evaluation run.
    """

    def __init__(self, references_dir: str, spkr_encoder):
        self._dir          = references_dir
        self._spkr_encoder = spkr_encoder
        self._embeddings   = {}   # speaker -> np.ndarray
        self._f0_stats     = {}   # speaker -> dict

    def get_embedding(self, speaker: str) -> np.ndarray:
        if speaker not in self._embeddings:
            self._embeddings[speaker] = load_speaker_embedding(
                speaker, self._dir, self._spkr_encoder
            )
        return self._embeddings[speaker]

    def get_f0_stats(self, speaker: str) -> dict:
        if speaker not in self._f0_stats:
            self._f0_stats[speaker] = load_f0_stats_from_audio(speaker, self._dir)
        return self._f0_stats[speaker]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(submission_dir: str, references_dir: str, corpus_csv: str,
             rtf_json: str, output_json: str,
             asr_model_card: str = ASR_MODEL_CARD,
             language_code: str  = LANGUAGE_CODE):

    # -- Load metadata
    meta_path = os.path.join(submission_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    team_name  = meta["team"]
    model_name = meta.get("system", "unknown")
    print(f"[KallaamaBench] Evaluating submission: {team_name} / {model_name}")

    # -- Load corpus CSV (filename | transcription | speaker | gender)
    corpus_df = load_corpus(corpus_csv)
    print(f"[KallaamaBench] Corpus: {len(corpus_df)} entries loaded from {corpus_csv}")

    # Build a lookup: filename stem -> row
    corpus_index = {
        os.path.splitext(row["filename"])[0]: row
        for _, row in corpus_df.iterrows()
    }

    # -- Load RTF (from submission metadata or external JSON)
    rtf = meta.get("rtf", None)
    if rtf is None and rtf_json and os.path.exists(rtf_json):
        with open(rtf_json) as f:
            rtf_data = json.load(f)
        for item in rtf_data:
            if item.get("model_name") == model_name:
                rtf = item["rtf"]
                break

    # -- Discover submitted audio files
    audio_dir   = os.path.join(submission_dir, "audio")
    audio_files = sorted([fn for fn in os.listdir(audio_dir) if fn.endswith(".wav")])
    print(f"[KallaamaBench] Found {len(audio_files)} audio file(s) in submission.")

    # -- Load models
    models = load_models(asr_model_card)

    # -- Speaker reference cache (loads reference WAVs on demand)
    ref_cache = SpeakerReferenceCache(references_dir, models["spkr_encoder"])

    # -- Score loop
    utmos_scores, wer_scores, snr_scores, spk_sim_scores, f0_corr_scores = \
        [], [], [], [], []
    skipped = []

    for i, fn in enumerate(audio_files):
        stem = os.path.splitext(fn)[0]

        if stem not in corpus_index:
            warnings.warn(f"No corpus entry for '{fn}' — skipping.")
            skipped.append(fn)
            continue

        row       = corpus_index[stem]
        reference = row["transcription"]
        speaker   = row["speaker"]

        path      = os.path.join(audio_dir, fn)
        audio, sr = librosa.load(path, sr=None)

        try:
            ref_emb  = ref_cache.get_embedding(speaker)
            f0_stats = ref_cache.get_f0_stats(speaker)
        except FileNotFoundError as exc:
            warnings.warn(str(exc) + " — skipping speaker similarity and F0 metrics.")
            ref_emb  = None
            f0_stats = None

        utmos = compute_utmos(path, models)
        wer_  = compute_wer(audio, sr, reference, models, language_code)
        snr   = compute_snr(audio, models)
        spk   = (
            compute_speaker_similarity(audio, sr, ref_emb, models)
            if ref_emb is not None else float("nan")
        )
        f0c = (
            compute_f0_correlation(audio, sr, f0_stats)
            if f0_stats is not None else float("nan")
        )

        utmos_scores.append(utmos)
        wer_scores.append(wer_)
        snr_scores.append(snr)
        spk_sim_scores.append(spk)
        f0_corr_scores.append(f0c)

        free_gpu()

        print(
            f"[{i+1:04d}/{len(audio_files)}] {fn} (spk={speaker}, gender={row['gender']}) | "
            f"UTMOS={utmos:.3f}  WER={wer_:.3f}  SNR={snr:.1f}dB  "
            f"SpkSim={spk:.3f}  F0Corr={f0c:.3f}"
        )

    # -- Aggregate (ignore NaN entries for speaker-level metrics)
    def _mean(lst):
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr)) if len(arr) > 0 else 0.0

    result = {
        "team"              : team_name,
        "system"            : model_name,
        "submission_date"   : meta.get("submission_date", ""),
        "n_samples"         : len(utmos_scores),
        "n_skipped"         : len(skipped),
        "UTMOS"             : _mean(utmos_scores),
        "WER"               : _mean(wer_scores),
        "SNR_dB"            : _mean(snr_scores),
        "SpeakerSimilarity" : _mean(spk_sim_scores),
        "F0Correlation"     : _mean(f0_corr_scores),
        "RTF"               : rtf,
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

    ev = sub.add_parser("evaluate", help="Score a participant submission")
    ev.add_argument("--submission",    required=True,
                    help="Path to submission folder")
    ev.add_argument("--references",    required=True,
                    help="Path to folder of per-speaker reference WAV files "
                         "(e.g. data/references/)")
    ev.add_argument("--corpus",        required=True,
                    help="Path to pipe-separated CSV corpus "
                         "(columns: filename|transcription|speaker|gender)")
    ev.add_argument("--rtf-json",      default=None,
                    help="Optional external RTF JSON")
    ev.add_argument("--output",        default="results.json",
                    help="Output JSON path")
    ev.add_argument("--asr-model",     default=ASR_MODEL_CARD,
                    help="Omnilingual ASR model card")
    ev.add_argument("--language-code", default=LANGUAGE_CODE,
                    help="Language code passed to the ASR pipeline (e.g. 'wo')")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "evaluate":
        evaluate(
            submission_dir = args.submission,
            references_dir = args.references,
            corpus_csv     = args.corpus,
            rtf_json       = args.rtf_json,
            output_json    = args.output,
            asr_model_card = args.asr_model,
            language_code  = args.language_code,
        )
    else:
        print("Use 'evaluate'. Run with --help for details.")