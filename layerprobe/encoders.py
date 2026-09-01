# -*- coding: utf-8 -*-
"""Extracting one sentence vector per transformer layer.

The encoder is used strictly as a frozen feature extractor: a batch of texts
goes in, an array of shape ``(n_layers, n_examples, hidden_size)`` comes out,
where layer 0 is the embedding layer and layer ``L`` the final one.  Nothing
here is fine-tuned -- the whole point of the study is to compare the
representations the pretrained model already has.

Two implementations:

:class:`HFEncoder`
    The real thing, wrapping any Hugging Face encoder that can return hidden
    states (``xlm-roberta-base`` by default).

:class:`SyntheticEncoder`
    A deterministic stand-in that needs no downloads.  It fabricates a
    layer profile in which language identity dominates the bottom layers,
    is suppressed in the middle, and partly returns at the top.  That shape
    is a *simulation* chosen to exercise every code path offline -- it is
    not evidence about any real model.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Sequence

import numpy as np

from layerprobe.config import EncoderConfig


POOLINGS = ("mean", "cls", "max", "last_token")


def _pool(states: "np.ndarray | object", mask, pooling: str):
    """Pool ``(batch, seq, hidden)`` token states down to ``(batch, hidden)``.

    Works on torch tensors; ``mask`` is the attention mask so that padding
    never contributes to a mean or a max.
    """

    import torch

    if pooling == "last_token":
        # The only sound single-token choice for a causal model: with
        # unidirectional attention the final real token is the only position
        # that has seen the whole sentence.  Found from the mask rather than
        # by taking [:, -1], so it is correct under either padding side.
        idx = _last_real_index(mask)
        return states[torch.arange(states.shape[0], device=states.device), idx]

    mask = mask.unsqueeze(-1).to(states.dtype)
    if pooling == "cls":
        return states[:, 0, :]
    if pooling == "mean":
        summed = (states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    if pooling == "max":
        neg_inf = torch.finfo(states.dtype).min
        return (states.masked_fill(mask == 0, neg_inf)).max(dim=1).values
    raise ValueError(f"unknown pooling {pooling!r}; expected one of {POOLINGS}")


def _last_real_index(mask):
    """Index of the final unmasked token in each row, for either padding side.

    Right padding gives ``[1,1,1,0,0]`` and left padding ``[0,0,1,1,1]``;
    reversing and taking the first hit locates the last real token in both.
    """

    import torch

    flipped = torch.flip(mask, dims=[1])
    from_end = torch.argmax(flipped.to(torch.int32), dim=1)
    idx = mask.shape[1] - 1 - from_end
    # An all-zero row (no real tokens) would otherwise index the last column.
    return torch.where(mask.sum(dim=1) > 0, idx, torch.zeros_like(idx))


class BaseEncoder:
    """Interface shared by the real and the offline encoder."""

    hidden_size: int
    num_layers: int  # transformer blocks; total layer count is this + 1

    @property
    def layer_ids(self) -> List[int]:
        """All available layer indices, embeddings (0) through final."""

        return list(range(self.num_layers + 1))

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HFEncoder(BaseEncoder):
    """Frozen Hugging Face encoder returning every hidden layer.

    ``model`` and ``tokenizer`` can be supplied directly, which is how the
    test-suite exercises this class without reaching the network.
    """

    def __init__(self, cfg: EncoderConfig, model=None, tokenizer=None):
        import torch

        self.cfg = cfg
        self.torch = torch
        if cfg.device:
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is None or tokenizer is None:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            kwargs = {"trust_remote_code": cfg.trust_remote_code}
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name, **kwargs)
            model_config = AutoConfig.from_pretrained(
                cfg.model_name, output_hidden_states=True, **kwargs
            )
            model = model or AutoModel.from_pretrained(
                cfg.model_name, config=model_config, dtype=self._dtype(), **kwargs
            )
        self.tokenizer = tokenizer
        self.model = model
        # Some checkpoints default to not returning hidden states; insist.
        self.model.config.output_hidden_states = True
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._prepare_tokenizer()
        self.hidden_size = int(self._config_value("hidden_size"))
        self.num_layers = int(self._config_value("num_hidden_layers"))
        self._check_pooling()

    def _config_value(self, name: str):
        """Read a field that multimodal configs nest under ``text_config``."""

        config = self.model.config
        if hasattr(config, name):
            return getattr(config, name)
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, name):
            return getattr(text_config, name)
        raise AttributeError(f"{self.cfg.model_name} config exposes no {name!r}")

    def _dtype(self):
        """Resolve ``encoder.dtype`` into a torch dtype (``None`` == leave alone)."""

        torch = self.torch
        name = (self.cfg.dtype or "auto").lower()
        if name == "auto":
            # bf16 halves memory and is ~free on a modern GPU; on CPU it is
            # markedly slower than fp32, so only opt in where it pays.
            if self.cfg.half_precision and self.device.type == "cuda":
                return torch.bfloat16
            return None
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]

    def _prepare_tokenizer(self) -> None:
        """Make a decoder-only tokenizer usable for batched feature extraction.

        Causal LMs are shipped for generation: most have no pad token at all,
        and several pad on the left.  Batching without a pad token raises;
        padding side is harmless here because every pooling mode consults the
        attention mask rather than assuming a position.
        """

        if getattr(self.tokenizer, "pad_token", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is None:
                raise ValueError(
                    f"{self.cfg.model_name} has neither a pad nor an eos token; "
                    "batched extraction needs one to pad with"
                )
            self.tokenizer.pad_token = eos

    def _check_pooling(self) -> None:
        """Warn when the pooling choice does not suit the architecture."""

        if self.cfg.pooling != "cls":
            return
        if self.is_causal:
            raise ValueError(
                f"pooling='cls' is meaningless for the decoder-only model "
                f"{self.cfg.model_name}: with causal attention the first token has "
                "seen nothing but itself. Use 'last_token' or 'mean'."
            )

    @property
    def is_causal(self) -> bool:
        """Whether the model attends unidirectionally (a decoder-only LLM)."""

        config = self.model.config
        if getattr(config, "is_decoder", False):
            return True
        architectures = getattr(config, "architectures", None) or []
        if any("ForCausalLM" in str(a) for a in architectures):
            return True
        # Fall back to the model class name: AutoModel strips the LM head, so
        # a bare LlamaModel/Qwen3Model/Gemma3TextModel still reaches here.
        name = type(self.model).__name__
        return any(tag in name for tag in ("Llama", "Qwen", "Gemma", "Mistral", "GPT", "Falcon"))

    def fertility(self, texts: Sequence[str]) -> dict:
        """Tokens per character, and how much of the corpus gets truncated.

        A confound that bites hard when comparing models: XLM-R's
        SentencePiece vocabulary was built with Amharic and Hausa in it,
        while an English-centric BPE can spend several tokens per character
        on the same script.  At a fixed ``max_length`` the two models
        therefore see *different amounts of text*, and a layer comparison
        across models silently becomes a comparison of how much got cut off.
        Reported per language so that bias is visible rather than assumed
        away.
        """

        lengths, truncated, chars = [], 0, 0
        for text in texts:
            # truncation off on purpose: the whole point is to see how much
            # would have been cut, which a truncated count cannot show.
            ids = self.tokenizer(str(text), add_special_tokens=True, truncation=False)["input_ids"]
            lengths.append(len(ids))
            chars += max(1, len(str(text)))
            if len(ids) > self.cfg.max_length:
                truncated += 1
        if not lengths:
            return {}
        return {
            "tokens_per_char": float(sum(lengths) / chars),
            "mean_tokens": float(np.mean(lengths)),
            "p95_tokens": float(np.percentile(lengths, 95)),
            "max_length": int(self.cfg.max_length),
            "truncated_fraction": float(truncated / len(lengths)),
        }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), self.cfg.batch_size):
            batch = [str(t) for t in texts[start : start + self.cfg.batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden_states = self.model(**encoded, output_hidden_states=True).hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    f"{self.cfg.model_name} returned no hidden states; the study needs an "
                    "encoder that exposes every layer"
                )
            mask = encoded["attention_mask"]
            pooled = torch.stack(
                [_pool(layer, mask, self.cfg.pooling) for layer in hidden_states], dim=0
            )
            outputs.append(pooled.float().cpu().numpy())
        if not outputs:
            return np.zeros((self.num_layers + 1, 0, self.hidden_size), dtype=np.float32)
        return np.concatenate(outputs, axis=1)


class SyntheticEncoder(BaseEncoder):
    """Offline stand-in with a deterministic, hand-built layer profile.

    Every token maps to a fixed pseudo-random vector via a hash, so the same
    text always yields the same representation and no state is downloaded.
    The synthetic corpus marks its tokens (see
    ``layerprobe.data._load_synthetic_language``), which lets this encoder
    build each layer as a mixture of a *shared* emotion component
    (``e{k}`` markers) and a *language* component (the ``L{language}`` tag),
    over a bed of filler noise.  The mixing weight varies with depth to mimic
    the U-shaped language-neutrality curve reported for multilingual
    encoders: language identity dominates at the bottom, is suppressed in the
    middle, and partly returns at the top.
    """

    #: How much of the pooled vector is filler that carries no signal.
    FILLER_WEIGHT = 0.35

    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self.hidden_size = cfg.synthetic_hidden_size
        self.num_layers = cfg.synthetic_num_layers
        self._cache: dict[str, np.ndarray] = {}
        # A fixed per-layer rotation stands in for the transformer's
        # per-layer feature basis, so that no two layers are collinear.
        self._rotations = np.stack(
            [
                np.random.default_rng(1000 + layer)
                .normal(size=(self.hidden_size, self.hidden_size))
                .astype(np.float32)
                / np.sqrt(self.hidden_size)
                for layer in range(self.num_layers + 1)
            ]
        )

    def _token_vector(self, token: str) -> np.ndarray:
        cached = self._cache.get(token)
        if cached is None:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            cached = rng.normal(size=self.hidden_size).astype(np.float32)
            self._cache[token] = cached
        return cached

    def _language_weight(self, layer: int) -> float:
        """How strongly language-specific tokens colour a given layer."""

        depth = layer / max(1, self.num_layers)
        # 1.0 at the embeddings, a minimum around two thirds depth, rising again.
        return float(0.15 + 0.85 * (2.0 * depth - 1.3) ** 2)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        n_layers = self.num_layers + 1
        out = np.zeros((n_layers, len(texts), self.hidden_size), dtype=np.float32)
        zero = np.zeros(self.hidden_size, np.float32)
        for i, text in enumerate(texts):
            tokens = str(text).split() or ["<empty>"]

            def _mean(chosen: List[str]) -> np.ndarray:
                if not chosen:
                    return zero
                return np.mean([self._token_vector(t) for t in chosen], axis=0)

            # emotion markers (shared across languages), the language tag
            # (constant within a language), and per-language filler noise
            shared_vec = _mean([t for t in tokens if t.startswith("e")])
            lang_vec = _mean([t for t in tokens if t.startswith("L")])
            filler_vec = _mean([t for t in tokens if not t.startswith(("e", "L"))])
            # Noise is seeded from the text, so the same sentence always gets
            # the same representation and the feature cache stays valid.
            digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
            noise_rng = np.random.default_rng(int.from_bytes(digest, "little"))
            noise = noise_rng.normal(
                scale=self.cfg.synthetic_noise, size=(n_layers, self.hidden_size)
            ).astype(np.float32)
            for layer in range(n_layers):
                w = self._language_weight(layer)
                mixed = (1.0 - w) * shared_vec + w * lang_vec + self.FILLER_WEIGHT * filler_vec
                out[layer, i] = mixed @ self._rotations[layer] + noise[layer]
        return out


def build_encoder(cfg: EncoderConfig) -> BaseEncoder:
    """Instantiate the encoder named by the config."""

    if cfg.model_name == "synthetic":
        return SyntheticEncoder(cfg)
    return HFEncoder(cfg)


def select_layers(features: np.ndarray, layers: Optional[Iterable[int]]) -> np.ndarray:
    """Keep only the requested layers of a ``(n_layers, n, d)`` array."""

    if layers is None:
        return features
    idx = list(layers)
    return features[idx]
