#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
# ]
# ///
"""
WER leaderboard plot for the README.

Reads every results/<system>/results_*.json, sorts systems from the lowest
(best) to the highest WER and draws a horizontal bar chart. The groundtruth
recordings are drawn as a reference line, not as a bar. Writes a light and a
dark variant so the README can pick one with <picture>.

Usage (from the repository root)
-----
uv run utils/leaderboard/plot_wer.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# results/<folder> -> name shown on the README leaderboard
DISPLAY_NAMES = {
    "adia": "Adia-TTS",
    "galsenai": "GalsenAI-TTS",
    "oolel": "Oolel-Voices",
    "omnivoice": "OmniVoice",
    "omnivoice-ft": "OmniVoice-FT",
    "omnivoice-ft-cs": "OmniVoice-FT-CS",
    "mira": "MiraTTS",
    "mira-ft": "MiraTTS-FT",
    "mira-ft-cs": "MiraTTS-FT-CS",
    "kiriku": "Kiriku-TTS",
    "pocket-tts": "Pocket-TTS",
}

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "bar": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33322f", "bar": "#3987e5"},
}


def load_wer(results_dir):
    wer = {}
    for path in sorted(Path(results_dir).glob("*/results_*.json")):
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
        entry = entry[0] if isinstance(entry, list) else entry
        wer[path.parent.name] = entry["WER"]
    return wer


def plot(systems, groundtruth_wer, theme, out_path):
    c = THEMES[theme]
    names = [DISPLAY_NAMES.get(k, k) for k, _ in systems]
    values = [v for _, v in systems]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(systems) + 1.3), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    y = range(len(systems))
    ax.barh(y, values, height=0.6, color=c["bar"], zorder=2)
    ax.invert_yaxis()  # best system on top
    ax.set_yticks(list(y), names, color=c["text"], fontsize=10)

    xmax = max(values) * 1.12
    ax.set_xlim(0, xmax)
    for i, v in enumerate(values):
        ax.text(v + xmax * 0.01, i, f"{v:.2f}", va="center", ha="left", color=c["text"], fontsize=9)

    if groundtruth_wer is not None:
        ax.axvline(groundtruth_wer, color=c["muted"], linewidth=1, zorder=3)
        ax.text(groundtruth_wer + xmax * 0.008, -0.75, f"Ground truth {groundtruth_wer:.2f}", ha="left", va="bottom",
                color=c["muted"], fontsize=8.5)

    ax.set_xlabel("WER (lower is better)", color=c["muted"], fontsize=9.5)
    ax.tick_params(axis="x", colors=c["muted"], labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=c["grid"], linewidth=1, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("KallaamaBench WER leaderboard", loc="left", color=c["text"], fontsize=12, pad=22)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=c["surface"])
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot the KallaamaBench WER leaderboard.")
    parser.add_argument("--results-dir", default="results/", help="Folder of per-system results (default: %(default)s)")
    parser.add_argument("--output-dir", default="assets/", help="Where to write the PNGs (default: %(default)s)")
    args = parser.parse_args()

    wer = load_wer(args.results_dir)
    groundtruth_wer = wer.pop("groundtruth", None)
    systems = sorted(wer.items(), key=lambda kv: kv[1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        plot(systems, groundtruth_wer, theme, out_dir / f"wer_leaderboard_{theme}.png")


if __name__ == "__main__":
    main()
