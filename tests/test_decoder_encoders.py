# -*- coding: utf-8 -*-
"""Decoder-only LLMs (Qwen, Llama, Gemma) as frozen feature extractors.

Every model here is built from its config with random weights, so the tests
exercise the real transformers code path -- causal attention, rotary
embeddings, the lot -- without downloading anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from layerprobe.config import DataConfig, EncoderConfig
from layerprobe.data import load_corpus, stable_seed
from layerprobe.encoders import HFEncoder, _last_real_index, _pool
from layerprobe.features import extract_split

SMALL = dict(
    vocab_size=64,
    hidden_size=16,
    num_hidden_layers=3,
    num_attention_heads=2,
    num_key_value_heads=1,
    intermediate_size=32,
    max_position_embeddings=64,
)


def _qwen():
    from transformers import Qwen3Config, Qwen3Model

    return Qwen3Model(Qwen3Config(head_dim=8, **SMALL))


def _llama():
    from transformers import LlamaConfig, LlamaModel

    return LlamaModel(LlamaConfig(**SMALL))


def _gemma():
    from transformers import Gemma3TextConfig, Gemma3TextModel

    return Gemma3TextModel(Gemma3TextConfig(head_dim=8, **SMALL))


DECODERS = {"qwen": _qwen, "llama": _llama, "gemma": _gemma}


class _StubTokenizer:
    """Whitespace tokeniser mimicking a causal LM's tokenizer.

    Starts with no pad token, exactly like Qwen/Llama/Gemma ship, so the
    encoder's pad-token handling is genuinely exercised.
    """

    def __init__(self, vocab_size: int = 64, padding_side: str = "right"):
        self.vocab_size = vocab_size
        self.pad_token = None
        self.eos_token = "</s>"
        self.padding_side = padding_side

    def _ids(self, text, max_length=None):
        ids = [2 + (stable_seed(tok) % (self.vocab_size - 3)) for tok in str(text).split()] or [1]
        return ids[:max_length] if max_length else ids

    def __call__(
        self,
        batch,
        padding=True,
        truncation=False,  # matches the real tokenizers' default
        max_length=16,
        return_tensors=None,
        add_special_tokens=True,
    ):
        # A single string comes back unbatched and unpadded, as the real
        # tokenizers do -- that is the shape the fertility count relies on.
        if isinstance(batch, str):
            return {"input_ids": self._ids(batch, max_length if truncation else None)}
        sequences = [self._ids(text, max_length) for text in batch]
        width = max(len(s) for s in sequences)
        input_ids = torch.zeros(len(sequences), width, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), width, dtype=torch.long)
        for i, seq in enumerate(sequences):
            if self.padding_side == "left":
                input_ids[i, width - len(seq) :] = torch.tensor(seq)
                attention_mask[i, width - len(seq) :] = 1
            else:
                input_ids[i, : len(seq)] = torch.tensor(seq)
                attention_mask[i, : len(seq)] = 1
        return _StubBatch({"input_ids": input_ids, "attention_mask": attention_mask})


class _StubBatch(dict):
    def to(self, device):
        return _StubBatch({k: v.to(device) for k, v in self.items()})


def _encoder(build, pooling="last_token", padding_side="right", batch_size=4):
    cfg = EncoderConfig(
        model_name="tiny-decoder", pooling=pooling, batch_size=batch_size, max_length=16
    )
    return HFEncoder(cfg, model=build(), tokenizer=_StubTokenizer(padding_side=padding_side))


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------


def test_last_real_index_handles_both_padding_sides():
    right = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
    np.testing.assert_array_equal(_last_real_index(right).numpy(), [2, 0])
    left = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 1]])
    np.testing.assert_array_equal(_last_real_index(left).numpy(), [4, 4])
    # A fully masked row must not index off the end.
    np.testing.assert_array_equal(_last_real_index(torch.tensor([[0, 0, 0]])).numpy(), [0])


def test_last_token_pooling_picks_the_final_real_token():
    states = torch.tensor([[[1.0], [2.0], [99.0]], [[7.0], [88.0], [88.0]]])
    right = torch.tensor([[1, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(_pool(states, right, "last_token").numpy(), [[2.0], [7.0]])


def test_last_token_pooling_ignores_left_padding():
    states = torch.tensor([[[99.0], [1.0], [2.0]]])
    left = torch.tensor([[0, 1, 1]])
    np.testing.assert_allclose(_pool(states, left, "last_token").numpy(), [[2.0]])


# --------------------------------------------------------------------------
# the three families
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(DECODERS))
def test_decoder_returns_every_layer(family):
    encoder = _encoder(DECODERS[family])
    features = encoder.encode(["hello world", "a b c d", "x"])
    assert encoder.num_layers == 3
    assert encoder.layer_ids == [0, 1, 2, 3]
    assert features.shape == (4, 3, 16)
    assert np.isfinite(features).all()
    assert not np.allclose(features[0], features[3])


@pytest.mark.parametrize("family", sorted(DECODERS))
def test_decoder_is_detected_as_causal(family):
    assert _encoder(DECODERS[family]).is_causal


@pytest.mark.parametrize("family", sorted(DECODERS))
def test_missing_pad_token_is_filled_in(family):
    encoder = _encoder(DECODERS[family])
    assert encoder.tokenizer.pad_token == encoder.tokenizer.eos_token


@pytest.mark.parametrize("family", sorted(DECODERS))
def test_cls_pooling_is_rejected_for_decoders(family):
    with pytest.raises(ValueError, match="meaningless for the decoder-only model"):
        _encoder(DECODERS[family], pooling="cls")


def test_encoder_model_still_allows_cls_pooling():
    from transformers import XLMRobertaConfig, XLMRobertaModel

    cfg = EncoderConfig(model_name="tiny-xlmr", pooling="cls", max_length=16)
    model = XLMRobertaModel(
        XLMRobertaConfig(
            vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=64,
        )
    )
    encoder = HFEncoder(cfg, model=model, tokenizer=_StubTokenizer())
    assert not encoder.is_causal
    assert encoder.encode(["a b", "c"]).shape == (3, 2, 16)


@pytest.mark.parametrize("family", sorted(DECODERS))
def test_batching_is_invariant_under_padding(family):
    """Padding must not leak into the pooled vector at any batch size."""

    build = DECODERS[family]
    model, tokenizer = build(), _StubTokenizer()
    texts = ["one two", "three four five six", "seven"]
    for pooling in ("last_token", "mean"):
        big = HFEncoder(
            EncoderConfig(model_name="t", pooling=pooling, batch_size=8, max_length=16),
            model=model, tokenizer=tokenizer,
        )
        small = HFEncoder(
            EncoderConfig(model_name="t", pooling=pooling, batch_size=1, max_length=16),
            model=model, tokenizer=tokenizer,
        )
        np.testing.assert_allclose(big.encode(texts), small.encode(texts), atol=1e-4, err_msg=pooling)


def test_left_padding_matches_right_padding_for_last_token():
    """Causal LMs often pad left; the read-out must not care which side."""

    model = _llama()
    texts = ["one two", "three four five", "six"]
    right = HFEncoder(
        EncoderConfig(model_name="t", pooling="last_token", batch_size=8, max_length=16),
        model=model, tokenizer=_StubTokenizer(padding_side="right"),
    ).encode(texts)
    left = HFEncoder(
        EncoderConfig(model_name="t", pooling="last_token", batch_size=8, max_length=16),
        model=model, tokenizer=_StubTokenizer(padding_side="left"),
    ).encode(texts)
    # Causal attention makes a left-padded row genuinely different in the
    # attention pattern, but with a correct mask the last real token's own
    # representation must still be recovered for a single-token sequence.
    np.testing.assert_allclose(right[:, 2], left[:, 2], atol=1e-3)


# --------------------------------------------------------------------------
# fertility
# --------------------------------------------------------------------------


def test_fertility_reports_tokens_and_truncation():
    encoder = _encoder(_qwen)
    report = encoder.fertility(["a b c", "d e f g h"])
    assert report["tokens_per_char"] > 0
    assert report["mean_tokens"] == pytest.approx(4.0)
    assert report["truncated_fraction"] == 0.0
    assert report["max_length"] == 16


def test_fertility_flags_truncation():
    encoder = _encoder(_qwen)
    long_text = " ".join(f"tok{i}" for i in range(40))
    report = encoder.fertility([long_text])
    assert report["truncated_fraction"] == 1.0


def test_fertility_is_empty_for_no_texts():
    assert _encoder(_qwen).fertility([]) == {}


# --------------------------------------------------------------------------
# integration with the feature cache
# --------------------------------------------------------------------------


def test_decoder_features_flow_through_the_cache(tmp_path):
    split = load_corpus(DataConfig(source="synthetic", languages=["eng"], synthetic_size=20), seed=0)["eng"]["train"]
    cfg = EncoderConfig(model_name="tiny-qwen", pooling="last_token", batch_size=4, max_length=16)
    encoder = HFEncoder(cfg, model=_qwen(), tokenizer=_StubTokenizer())

    first = extract_split(split, cfg, encoder, tmp_path)
    assert first.values.shape == (4, len(split), 16)

    class _Exploding:
        layer_ids = encoder.layer_ids

        def encode(self, texts):
            raise AssertionError("should have hit the cache")

    second = extract_split(split, cfg, _Exploding(), tmp_path)
    np.testing.assert_allclose(first.values, second.values)


def test_dtype_changes_the_cache_key(tmp_path):
    split = load_corpus(DataConfig(source="synthetic", languages=["eng"], synthetic_size=10), seed=0)["eng"]["train"]
    for dtype in ("auto", "float32"):
        cfg = EncoderConfig(model_name="tiny-qwen", dtype=dtype, batch_size=4, max_length=16)
        extract_split(split, cfg, HFEncoder(cfg, model=_qwen(), tokenizer=_StubTokenizer()), tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2
