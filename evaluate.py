"""
KallaamaBench — Official Evaluation Script
==========================================
Metrics: UTMOS · WER · SNR · RTF · Speaker Similarity · F0 Correlation

Requirements
------------
pip install omnilingual_asr==0.2.0

Usage
-----
python evaluate.py precompute \
    --audio-folder data/audio/ \
    --corpus       data/corpus.csv \
    --output-npz   data/groundtruth.npz

python evaluate.py evaluate \
    --submission submissions/adia/ \
    --groundtruth data/groundtruth.npz \
    --corpus     data/corpus.csv \
    --output     results/adia/results_adia.json

Note: data/references/ contains per-speaker WAV files used by TTS models
as conditioning audio during synthesis; they are NOT used by this script.
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
ASR_MODEL_CARD   = "omniASR_LLM_1B"             # override via --asr-model
LANGUAGE_CODE    = "wol_Latn"                   # override via --language-code
TARGET_SR        = 16_000
CHUNK_DURATION_S = 10                           # seconds per ASR chunk
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
    chunk_size  = chunk_duration_s * sample_rate
    min_samples = int(min_chunk_s * sample_rate)
    for start in range(0, len(audio_array), chunk_size):
        chunk = audio_array[start: start + chunk_size]
        if len(chunk) < min_samples:
            break
        yield start, chunk


# ---------------------------------------------------------------------------
# Checkpoint helpers  (one JSONL line per completed sample)
# ---------------------------------------------------------------------------

def _checkpoint_path(output_json: str) -> str:
    """Return the .jsonl checkpoint file path derived from the output JSON path."""
    base, _ = os.path.splitext(output_json)
    return base + ".checkpoint.jsonl"


def load_checkpoint(output_json: str) -> dict:
    """
    Load previously scored samples from the checkpoint file.

    Returns a dict mapping audio filename stem -> per-sample score dict,
    so the main loop can skip already-completed files.
    """
    path = _checkpoint_path(output_json)
    completed: dict = {}
    if not os.path.exists(path):
        return completed
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed[record["stem"]] = record
            except (json.JSONDecodeError, KeyError):
                pass  # ignore corrupted lines
    if completed:
        print(f"[KallaamaBench] Resuming: {len(completed)} sample(s) already scored "
              f"(checkpoint: {path})")
    return completed


def append_checkpoint(output_json: str, stem: str, scores: dict) -> None:
    """Append one completed sample's scores to the checkpoint file (atomic-ish)."""
    path   = _checkpoint_path(output_json)
    record = {"stem": stem, **scores}
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def delete_checkpoint(output_json: str) -> None:
    """Remove the checkpoint file after a successful, complete evaluation run."""
    path = _checkpoint_path(output_json)
    if os.path.exists(path):
        os.remove(path)
        print(f"[KallaamaBench] Checkpoint removed: {path}")


# ---------------------------------------------------------------------------
# Precompute checkpoint helpers
#
# Unlike the evaluate checkpoint (which stores scalar scores), the precompute
# checkpoint must persist numpy arrays (speaker embeddings, F0 histograms …).
# Each JSONL line stores one file's arrays as base64-encoded raw bytes so that
# the file remains plain-text and human-inspectable.
# ---------------------------------------------------------------------------

def _precompute_checkpoint_path(output_npz: str) -> str:
    base, _ = os.path.splitext(output_npz)
    return base + ".checkpoint.jsonl"


def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Serialize a numpy array to a base64 string (dtype + shape preserved)."""
    import base64, io
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_ndarray(s: str) -> np.ndarray:
    """Deserialize a numpy array from a base64 string."""
    import base64, io
    return np.load(io.BytesIO(base64.b64decode(s)))


def load_precompute_checkpoint(output_npz: str) -> dict:
    """
    Return a dict mapping filename stem -> dict of precomputed arrays/scalars
    for every file already processed in a previous (interrupted) run.
    """
    path = _precompute_checkpoint_path(output_npz)
    completed: dict = {}
    if not os.path.exists(path):
        return completed
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                stem   = record["stem"]
                completed[stem] = {
                    "spk_emb"  : _b64_to_ndarray(record["spk_emb"]),
                    "f0_mean"  : record["f0_mean"],
                    "f0_std"   : record["f0_std"],
                    "f0_hist"  : _b64_to_ndarray(record["f0_hist"]),
                    "f0_edges" : _b64_to_ndarray(record["f0_edges"]),
                }
            except (json.JSONDecodeError, KeyError, Exception):
                pass  # skip corrupted lines
    if completed:
        print(f"[KallaamaBench] Resuming precompute: {len(completed)} file(s) already "
              f"processed (checkpoint: {path})")
    return completed


def append_precompute_checkpoint(output_npz: str, stem: str,
                                  spk_emb: np.ndarray, f0_mean: float,
                                  f0_std: float, f0_hist: np.ndarray,
                                  f0_edges: np.ndarray) -> None:
    """Persist one file's precomputed data immediately after processing."""
    path   = _precompute_checkpoint_path(output_npz)
    record = {
        "stem"    : stem,
        "spk_emb" : _ndarray_to_b64(spk_emb),
        "f0_mean" : f0_mean,
        "f0_std"  : f0_std,
        "f0_hist" : _ndarray_to_b64(f0_hist),
        "f0_edges": _ndarray_to_b64(f0_edges),
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def delete_precompute_checkpoint(output_npz: str) -> None:
    """Remove the precompute checkpoint after a successful run."""
    path = _precompute_checkpoint_path(output_npz)
    if os.path.exists(path):
        os.remove(path)
        print(f"[KallaamaBench] Precompute checkpoint removed: {path}")


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_corpus(corpus_csv: str) -> pd.DataFrame:
    """
    Load a pipe-separated CSV with columns: filename, transcription, speaker, gender.

    Rows are returned in file order; the ordering must match groundtruth.npz.
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
# Reference data helpers
# ---------------------------------------------------------------------------

def load_groundtruth(groundtruth_npz: str) -> dict:
    """
    Load the precomputed reference vectors from groundtruth.npz.

    Expected keys
    -------------
    speaker_embeddings : float32 array of shape (N, D)
        ECAPA-TDNN speaker embeddings, one per corpus row.
    f0_means           : float32 array of shape (N,)
    f0_stds            : float32 array of shape (N,)
    f0_hists           : float32 array of shape (N, 50)
    f0_edges           : float32 array of shape (N, 51)
    """
    data = np.load(groundtruth_npz)
    return {
        "speaker_embeddings": data["speaker_embeddings"],
        "f0_means"          : data["f0_means"],
        "f0_stds"           : data["f0_stds"],
        "f0_hists"          : data["f0_hists"],
        "f0_edges"          : data["f0_edges"],
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(asr_model_card: str = ASR_MODEL_CARD) -> dict:
    """Load all models once and return them in a dict."""
    print("[KallaamaBench] Loading models ...")

    utmos_model  = utmosv2.create_model(pretrained=True)
    asr_pipeline = ASRInferencePipeline(model_card=asr_model_card, device=str(DEVICE))
    denoiser     = separator.from_hparams(
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

def transcribe_chunked(audio_array: np.ndarray, asr_pipeline,
                        language_code: str) -> str:
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


def compute_wer(audio: np.ndarray, reference: str,
                models: dict, language_code: str) -> float:
    """ASR-based Word Error Rate using Omnilingual ASR."""
    hypothesis = transcribe_chunked(audio, models["asr_pipeline"], language_code)
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
    return _snr_power(t.cpu().numpy(), noise_estimate.cpu().numpy())


def compute_speaker_similarity(audio: np.ndarray, ref_embedding: np.ndarray, models: dict) -> float:
    """
    Cosine similarity between the submitted sample's speaker embedding and the
    reference embedding loaded from groundtruth.npz.

    Returns a value in [-1, 1]; higher is more similar.
    """
    t         = torch.tensor(audio).unsqueeze(0).float().to(DEVICE)
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
    reference F0 histogram stored in groundtruth.npz.

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
# Precompute reference vectors (utility — run once by benchmark organisers)
# ---------------------------------------------------------------------------

def precompute_groundtruth(audio_folder: str, corpus_csv: str,
                           output_npz: str,
                           spkr_model_name: str = SPKR_MODEL_NAME):
    """
    Precompute and save speaker embeddings and F0 statistics for all reference
    audio files.  Run once by the benchmark organisers; participants never
    call this function.

    The audio files must be sorted in the same order as the corpus CSV rows.

    Resumable: if interrupted, re-running with the same arguments picks up
    where it left off using a '<output_npz>.checkpoint.jsonl' sidecar file.
    """
    print("[KallaamaBench] Precomputing groundtruth vectors ...")

    corpus_df   = load_corpus(corpus_csv)
    audio_files = sorted([fn for fn in os.listdir(audio_folder) if fn.endswith(".wav")])

    assert len(audio_files) == len(corpus_df), (
        f"Mismatch: {len(audio_files)} audio files vs {len(corpus_df)} corpus rows"
    )

    N_BINS    = 50
    F0_MIN    = librosa.note_to_hz("C2")
    F0_MAX    = librosa.note_to_hz("C7")
    bin_edges = np.linspace(F0_MIN, F0_MAX, N_BINS + 1)

    # -- Load checkpoint: files already processed in a previous interrupted run
    completed = load_precompute_checkpoint(output_npz)

    remaining = [fn for fn in audio_files
                 if os.path.splitext(fn)[0] not in completed]

    # Only load the heavy encoder model if there is actual work left to do
    if remaining:
        spkr_encoder = EncoderClassifier.from_hparams(
            source=spkr_model_name, run_opts={"device": str(DEVICE)}
        )
    else:
        spkr_encoder = None
        print("[KallaamaBench] All files already processed; skipping model loading.")

    # Collect results in corpus order (important: .npz row order must match corpus)
    spk_embs, f0_means, f0_stds, f0_hists, f0_edges_all = [], [], [], [], []

    for i, fn in enumerate(audio_files):
        stem = os.path.splitext(fn)[0]

        # -- Resume: replay cached result without reprocessing
        if stem in completed:
            rec = completed[stem]
            spk_embs.append(rec["spk_emb"])
            f0_means.append(rec["f0_mean"])
            f0_stds.append(rec["f0_std"])
            f0_hists.append(rec["f0_hist"])
            f0_edges_all.append(rec["f0_edges"])
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(audio_files)} done (from checkpoint)")
            continue

        # -- Process new file
        path      = os.path.join(audio_folder, fn)
        audio, sr = librosa.load(path, sr=None)
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        # Speaker embedding
        t = torch.tensor(audio_16k).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            emb = spkr_encoder.encode_batch(t).squeeze().cpu().numpy()

        # F0
        f0_voiced = extract_f0(audio_16k, TARGET_SR)
        f0_mean   = float(np.mean(f0_voiced))
        f0_std    = float(np.std(f0_voiced))
        hist, _   = np.histogram(f0_voiced, bins=bin_edges)
        f0_hist   = hist.astype(np.float32)
        f0_edges  = bin_edges.astype(np.float32)

        spk_embs.append(emb)
        f0_means.append(f0_mean)
        f0_stds.append(f0_std)
        f0_hists.append(f0_hist)
        f0_edges_all.append(f0_edges)

        # -- Persist immediately so a crash doesn't lose this file's work
        append_precompute_checkpoint(
            output_npz, stem, emb, f0_mean, f0_std, f0_hist, f0_edges
        )

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(audio_files)} done")

    np.savez_compressed(
        output_npz,
        speaker_embeddings = np.stack(spk_embs).astype(np.float32),
        f0_means           = np.array(f0_means,  dtype=np.float32),
        f0_stds            = np.array(f0_stds,   dtype=np.float32),
        f0_hists           = np.stack(f0_hists).astype(np.float32),
        f0_edges           = np.stack(f0_edges_all).astype(np.float32),
    )
    print(f"[KallaamaBench] Saved reference vectors -> {output_npz}")

    # Clean up checkpoint now that the .npz is safely written
    delete_precompute_checkpoint(output_npz)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(submission_dir: str, groundtruth_npz: str, corpus_csv: str,
             output_json: str, asr_model_card: str = ASR_MODEL_CARD,
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

    # Build a lookup: filename stem -> (row_index, row)
    corpus_index = {
        os.path.splitext(row["filename"])[0]: (idx, row)
        for idx, row in corpus_df.iterrows()
    }

    # -- Load groundtruth vectors (speaker embeddings + F0 stats)
    groundtruth = load_groundtruth(groundtruth_npz)

    # -- Load RTF (from submission metadata)
    rtf = meta.get("rtf", None)
    if rtf is None:
        rtf = 0.0

    # -- Discover submitted audio files
    audio_dir   = os.path.join(submission_dir, "audio")
    audio_files = sorted([fn for fn in os.listdir(audio_dir) if fn.endswith(".wav")])
    print(f"[KallaamaBench] Found {len(audio_files)} audio file(s) in submission.")

    # -- Load checkpoint (already-scored samples from a previous interrupted run)
    completed = load_checkpoint(output_json)

    # -- Load models (skip if everything is already done)
    remaining = [fn for fn in audio_files
                 if os.path.splitext(fn)[0] not in completed
                 and os.path.splitext(fn)[0] in corpus_index]
    if remaining:
        models = load_models(asr_model_card)
    else:
        models = {}
        print("[KallaamaBench] All samples already scored; skipping model loading.")

    # -- Score loop
    utmos_scores, wer_scores, snr_scores, spk_sim_scores, f0_corr_scores = \
        [], [], [], [], []
    skipped = []

    # Re-populate score lists from checkpoint so aggregates are correct
    for record in completed.values():
        utmos_scores.append(record["utmos"])
        wer_scores.append(record["wer"])
        snr_scores.append(record["snr"])
        spk_sim_scores.append(record["spk_sim"])
        f0_corr_scores.append(record["f0_corr"])

    for i, fn in enumerate(audio_files):
        stem = os.path.splitext(fn)[0]

        if stem not in corpus_index:
            warnings.warn(f"No corpus entry for '{fn}' -- skipping.")
            skipped.append(fn)
            continue

        # -- Resume: skip already-completed samples
        if stem in completed:
            row_idx, row = corpus_index[stem]
            rec = completed[stem]
            print(
                f"[{i+1:04d}/{len(audio_files)}] {fn} SKIPPED (already scored) | "
                f"UTMOS={rec['utmos']:.3f}  WER={rec['wer']:.3f}  "
                f"SNR={rec['snr']:.1f}dB  SpkSim={rec['spk_sim']:.3f}  "
                f"F0Corr={rec['f0_corr']:.3f}"
            )
            continue

        row_idx, row = corpus_index[stem]
        reference    = row["transcription"]
        speaker      = row["speaker"]

        path      = os.path.join(audio_dir, fn)
        audio, sr = librosa.load(path, sr=None)
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        print(f"Original SR: {sr} Hz, Resampled SR: {TARGET_SR} Hz, Duration: {len(audio_16k)/TARGET_SR:.2f} s")

        utmos = compute_utmos(path, models)
        wer_  = compute_wer(audio_16k, reference, models, language_code)
        snr   = compute_snr(audio_16k, models)
        spk   = compute_speaker_similarity(
            audio_16k,
            ref_embedding=groundtruth["speaker_embeddings"][row_idx],
            models=models,
        )
        f0c = compute_f0_correlation(
            audio_16k, TARGET_SR,
            ref_f0_stats={
                "mean" : groundtruth["f0_means"][row_idx],
                "std"  : groundtruth["f0_stds"][row_idx],
                "hist" : groundtruth["f0_hists"][row_idx],
                "edges": groundtruth["f0_edges"][row_idx],
            },
        )

        utmos_scores.append(utmos)
        wer_scores.append(wer_)
        snr_scores.append(snr)
        spk_sim_scores.append(spk)
        f0_corr_scores.append(f0c)

        # -- Persist this sample's scores immediately so a crash won't lose them
        append_checkpoint(output_json, stem, {
            "utmos"  : utmos,
            "wer"    : wer_,
            "snr"    : snr,
            "spk_sim": spk,
            "f0_corr": f0c,
        })

        free_gpu()

        print(
            f"[{i+1:04d}/{len(audio_files)}] {fn} "
            f"(spk={speaker}, gender={row['gender']}) | "
            f"UTMOS={utmos:.3f}  WER={wer_:.3f}  SNR={snr:.1f}dB  "
            f"SpkSim={spk:.3f}  F0Corr={f0c:.3f}"
        )

    # -- Aggregate
    def _mean(lst): return float(np.mean(lst)) if lst else 0.0

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

    print("\n[KallaamaBench] -- Final scores ------------------------------------------")
    for k, v in result.items():
        print(f"  {k:<22}: {v}")
    print(f"[KallaamaBench] Saved -> {output_json}")

    # All samples scored successfully — the checkpoint is no longer needed.
    delete_checkpoint(output_json)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p   = argparse.ArgumentParser(description="KallaamaBench evaluation script")
    sub = p.add_subparsers(dest="command")

    # evaluate subcommand
    ev = sub.add_parser("evaluate", help="Score a participant submission")
    ev.add_argument("--submission",    required=True,
                    help="Path to submission folder")
    ev.add_argument("--groundtruth",    required=True,
                    help="Path to groundtruth.npz")
    ev.add_argument("--corpus",        required=True,
                    help="Path to pipe-separated CSV corpus "
                         "(columns: filename|transcription|speaker|gender)")
    ev.add_argument("--output",        default="results.json",
                    help="Output JSON path")
    ev.add_argument("--asr-model",     default=ASR_MODEL_CARD,
                    help="Omnilingual ASR model card")
    ev.add_argument("--language-code", default=LANGUAGE_CODE,
                    help="Language code passed to the ASR pipeline (e.g. 'wo')")

    # precompute subcommand (organisers only)
    pc = sub.add_parser("precompute", help="Precompute reference vectors (organisers only)")
    pc.add_argument("--audio-folder", required=True,
                    help="Folder of reference audio WAV files")
    pc.add_argument("--corpus",       required=True,
                    help="Path to pipe-separated CSV corpus")
    pc.add_argument("--output-npz",   default="data/groundtruth.npz",
                    help="Output .npz path")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "evaluate":
        evaluate(
            submission_dir = args.submission,
            groundtruth_npz = args.groundtruth,
            corpus_csv     = args.corpus,
            output_json    = args.output,
            asr_model_card = args.asr_model,
            language_code  = args.language_code,
        )
    elif args.command == "precompute":
        precompute_groundtruth(
            audio_folder = args.audio_folder,
            corpus_csv   = args.corpus,
            output_npz   = args.output_npz,
        )
    else:
        print("Use 'evaluate' or 'precompute'. Run with --help for details.")