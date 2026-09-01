# -*- coding: utf-8 -*-
"""Layer extraction, pooling and the feature cache."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from layerprobe.config import DataConfig, EncoderConfig
from layerprobe.data import load_corpus, stable_seed
from layerprobe.encoders import HFEncoder, SyntheticEncoder, _pool, build_encoder
from layerprobe.features import extract_corpus, extract_split, stack_features


class _StubTokenizer:
    """Whitespace tokeniser with the slice of the HF API that HFEncoder uses."""

    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size
        # A masked encoder's tokenizer always ships a pad token, unlike the
        # causal ones exercised in test_decoder_encoders.py.
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __call__(self, batch, padding=True, truncation=True, max_length=16, return_tensors="pt"):
        sequences = [
            [1]
            + [2 + (stable_seed(tok) % (self.vocab_size - 3)) for tok in text.split()][: max_length - 1]
            for text in batch
        ]
        width = max(len(s) for s in sequences)
        input_ids = torch.zeros(len(sequences), width, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), width, dtype=torch.long)
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq)
            attention_mask[i, : len(seq)] = 1
        return _StubBatch({"input_ids": input_ids, "attention_mask": attention_mask})


class _StubBatch(dict):
    def to(self, device):
        return _StubBatch({k: v.to(device) for k, v in self.items()})


def _tiny_xlmr():
    """A randomly initialised XLM-R, built from config so nothing downloads."""

    from transformers import XLMRobertaConfig, XLMRobertaModel

    config = XLMRobertaConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        output_hidden_states=True,
    )
    return XLMRobertaModel(config)


def test_pooling_ignores_padding():
    states = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    np.testing.assert_allclose(_pool(states, mask, "mean").numpy(), [[2.0, 2.0]])
    np.testing.assert_allclose(_pool(states, mask, "max").numpy(), [[3.0, 3.0]])
    np.testing.assert_allclose(_pool(states, mask, "cls").numpy(), [[1.0, 1.0]])


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="unknown pooling"):
        _pool(torch.zeros(1, 2, 2), torch.ones(1, 2), "median")


def test_hf_encoder_returns_every_layer():
    """The real extraction path, on a random-weight model (no download)."""

    cfg = EncoderConfig(model_name="tiny-xlmr", pooling="mean", batch_size=2, max_length=16)
    encoder = HFEncoder(cfg, model=_tiny_xlmr(), tokenizer=_StubTokenizer())
    features = encoder.encode(["hello world", "a b c d", "x"])

    assert encoder.num_layers == 3
    assert encoder.layer_ids == [0, 1, 2, 3]
    # embeddings + one per block, batched correctly
    assert features.shape == (4, 3, 16)
    assert np.isfinite(features).all()
    # different layers must not be identical, or the study has nothing to compare
    assert not np.allclose(features[0], features[3])


def test_hf_encoder_batching_matches_single_pass():
    model, tokenizer = _tiny_xlmr(), _StubTokenizer()
    texts = ["one two", "three four five", "six"]
    big = HFEncoder(EncoderConfig(model_name="t", batch_size=8), model=model, tokenizer=tokenizer)
    small = HFEncoder(EncoderConfig(model_name="t", batch_size=1), model=model, tokenizer=tokenizer)
    # Padding differs between batch sizes, but mask-aware pooling must cancel
    # that out; a mismatch here would mean padding is leaking into features.
    np.testing.assert_allclose(big.encode(texts), small.encode(texts), atol=1e-4)


def test_synthetic_encoder_is_deterministic():
    cfg = EncoderConfig(model_name="synthetic", synthetic_hidden_size=8, synthetic_num_layers=4)
    encoder = build_encoder(cfg)
    assert isinstance(encoder, SyntheticEncoder)
    first = encoder.encode(["e0 w1 w2", "e1 w3"])
    second = build_encoder(cfg).encode(["e0 w1 w2", "e1 w3"])
    np.testing.assert_allclose(first, second)
    assert first.shape == (5, 2, 8)


def test_synthetic_encoder_handles_empty_text():
    cfg = EncoderConfig(model_name="synthetic", synthetic_hidden_size=8, synthetic_num_layers=2)
    features = build_encoder(cfg).encode(["", "e0"])
    assert features.shape == (3, 2, 8)
    assert np.isfinite(features).all()


def test_feature_cache_round_trips(tmp_path):
    data_cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=40)
    split = load_corpus(data_cfg, seed=0)["eng"]["train"]
    enc_cfg = EncoderConfig(model_name="synthetic", synthetic_hidden_size=8, synthetic_num_layers=3)
    encoder = build_encoder(enc_cfg)

    first = extract_split(split, enc_cfg, encoder, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    # A second call must hit the cache -- an encoder that refuses to run
    # proves the values came off disk.
    class _Exploding:
        layer_ids = encoder.layer_ids

        def encode(self, texts):
            raise AssertionError("cache miss: the encoder should not have been called")

    second = extract_split(split, enc_cfg, _Exploding(), tmp_path)
    np.testing.assert_allclose(first.values, second.values)
    assert second.layer_ids == first.layer_ids


def test_cache_key_changes_with_pooling(tmp_path):
    data_cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=20)
    split = load_corpus(data_cfg, seed=0)["eng"]["train"]
    base = EncoderConfig(model_name="synthetic", synthetic_hidden_size=8, synthetic_num_layers=2)
    extract_split(split, base, build_encoder(base), tmp_path)
    other = EncoderConfig(
        model_name="synthetic", pooling="cls", synthetic_hidden_size=8, synthetic_num_layers=2
    )
    extract_split(split, other, build_encoder(other), tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_layer_subset_selection():
    data_cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=20)
    split = load_corpus(data_cfg, seed=0)["eng"]["train"]
    cfg = EncoderConfig(
        model_name="synthetic", layers=[0, 2], synthetic_hidden_size=8, synthetic_num_layers=4
    )
    features = extract_split(split, cfg, build_encoder(cfg))
    assert features.layer_ids == [0, 2]
    assert features.n_layers == 2
    with pytest.raises(KeyError, match="layer 1 was not extracted"):
        features.layer(1)


def test_requesting_a_nonexistent_layer_raises():
    data_cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=10)
    split = load_corpus(data_cfg, seed=0)["eng"]["train"]
    cfg = EncoderConfig(model_name="synthetic", layers=[0, 99], synthetic_num_layers=3)
    with pytest.raises(ValueError, match="requested layers"):
        extract_split(split, cfg, build_encoder(cfg))


def test_extract_corpus_and_stack():
    data_cfg = DataConfig(source="synthetic", languages=["eng", "amh"], synthetic_size=30)
    corpus = load_corpus(data_cfg, seed=0)
    cfg = EncoderConfig(model_name="synthetic", synthetic_hidden_size=8, synthetic_num_layers=3)
    store, encoder = extract_corpus(corpus, cfg, cache_dir=None, verbose=False)

    assert set(store) == {"eng", "amh"}
    pooled = stack_features([store["eng"]["train"], store["amh"]["train"]])
    assert pooled.values.shape[1] == len(corpus["eng"]["train"]) + len(corpus["amh"]["train"])
    assert pooled.layer_ids == encoder.layer_ids


def test_stacking_mismatched_layer_sets_raises():
    data_cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=20)
    split = load_corpus(data_cfg, seed=0)["eng"]["train"]
    a = extract_split(split, EncoderConfig(model_name="synthetic", layers=[0, 1]), build_encoder(EncoderConfig(model_name="synthetic")))
    b = extract_split(split, EncoderConfig(model_name="synthetic", layers=[0, 2]), build_encoder(EncoderConfig(model_name="synthetic")))
    with pytest.raises(ValueError, match="different layer sets"):
        stack_features([a, b])


def test_normalisation_produces_unit_vectors():
    data_cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=20)
    split = load_corpus(data_cfg, seed=0)["eng"]["train"]
    cfg = EncoderConfig(model_name="synthetic", normalize=True, synthetic_num_layers=2)
    features = extract_split(split, cfg, build_encoder(cfg))
    norms = np.linalg.norm(features.values, axis=-1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)
