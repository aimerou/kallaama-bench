"""
Wolof orthography normalizer using a causal language model.

Usage:
    python normalizer.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "Fix the orthography of this Wolof social media text. "
    "Apply standard rules: correct vowels with diacritics, restore geminates, "
    "fix French-influenced spellings. "
    "Keep mixed French/English as-is. "
    "Reply only with <CORRECTION>corrected text</CORRECTION>."
)

_CORRECTION_RE = re.compile(r"<CORRECTION>(.*?)</CORRECTION>", re.DOTALL)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NormalizerConfig:
    """Hyper-parameters for the WolofNormalizer."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 512
    temperature: float = 0.1


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class WolofNormalizer:
    """
    Correct the orthography of Wolof social-media text.

    The model and tokenizer are injected at construction time so they can be
    loaded once and shared across multiple normalizer instances or calls.

    Args:
        model:     A pre-loaded ``AutoModelForCausalLM`` (or compatible) model.
        tokenizer: The matching ``AutoTokenizer``.
        config:    Optional ``NormalizerConfig``; defaults are used if omitted.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: NormalizerConfig | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or NormalizerConfig()
        self._device: str = str(next(model.parameters()).device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """
        Return the orthographically corrected Wolof text.

        Args:
            text: Raw Wolof social-media text to correct.

        Returns:
            Corrected text extracted from the model's ``<CORRECTION>`` tag,
            or the full model output if the tag is absent.
        """
        raw_output = self._generate(text)
        return self._extract_correction(raw_output)

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Convenience wrapper – normalizes each text independently."""
        return [self.normalize(t) for t in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self._config.system_prompt},
            {"role": "user",   "content": text},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, text: str) -> str:
        prompt = self._build_prompt(text)
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._device)

        generated_ids = self._model.generate(
            inputs.input_ids,
            max_new_tokens=self._config.max_new_tokens,
            do_sample=True,       # required for temperature to take effect;
            temperature=self._config.temperature,  # without this flag HuggingFace
                                  # falls back to greedy decoding and ignores temperature
        )

        # Strip the prompt tokens from the output
        new_ids = [
            output[len(inp):]
            for inp, output in zip(inputs.input_ids, generated_ids)
        ]
        return self._tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]

    @staticmethod
    def _extract_correction(raw: str) -> str:
        match = _CORRECTION_RE.search(raw)
        return match.group(1).strip() if match else raw.strip()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load and return the causal LM and its tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def main() -> None:
    model_name = "soynade-research/Oolel-Corrector"

    print(f"Loading model '{model_name}' …")
    model, tokenizer = load_model(model_name)

    config = NormalizerConfig(
        max_new_tokens=512,
        temperature=0.1,
    )
    normalizer = WolofNormalizer(model, tokenizer, config)

    samples = [
        "man mom khawma sakh li loumou done. Niom gneup dem naniou dakar ngir vacances scolaires yi",
        "Dinaa dem ecole bii jang wolof ak francais",
    ]

    for text in samples:
        result = normalizer.normalize(text)
        print(f"\nInput : {text}")
        print(f"Output: {result}")


if __name__ == "__main__":
    main()