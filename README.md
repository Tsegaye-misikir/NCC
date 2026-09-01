# The Code

This repository holds two related pieces of work on multilingual
representations.

---

## 1. Inter-lingual document representations

This work proposes several new methods to derive inter-lingual document represen-tations (Marc-lenz ,Tsegaye Misikir and Tomas Horvath). 
these methods aim to enhance the quality content-based Multilingual Document Recommendation Systems. 
The main idea centers around creating inter-lingualrepresentations by using mappings to align monolingual representation spaces.

- `Notebooks/1_..4_` - Jupyter Notebooks which were used for the main evaluations.
- `evaluation_functions.py` - Contains different functions which create, train, and evaluate models. (mainly used inside the Jupyter Notebooks)
- `Preprocessor.py` - Preprocessing Class used to preprocess text of different languages.
- `Utils.py` - containing smaller auxilary functions

---

## 2. Layer-wise emotion representations for low-resource languages

Which transformer layers of a multilingual encoder give the best
representation for emotion recognition — and does choosing better than "the
final layer" reduce negative cross-lingual transfer?

The study freezes a pretrained multilingual encoder (XLM-R by default),
extracts a sentence vector from **every** hidden layer, and trains a
light-weight probe on individual layers, on the final layer, and on
combinations of layers (windowed averages, concatenations, and a learned
weighted mix). It runs three regimes — monolingual, zero-shot cross-lingual,
and joint multilingual — and reports, per language, how much macro-F1 the
default final-layer choice was costing.

```bash
pip install -r requirements.txt

# Offline smoke run: no downloads, ~1 minute, writes every output file.
python run_experiments.py --config configs/smoke.yaml

# The real study.
python run_experiments.py --config configs/brighter.yaml
```

| Path | Contents |
| --- | --- |
| `docs/LAYERWISE_EMOTION.md` | **Start here** — methodology, how to run, how to read the output, limitations |
| `layerprobe/` | The package: data loading, layer extraction, combinations, probes, diagnostics, reporting |
| `configs/` | `smoke.yaml` (offline), `brighter.yaml` (SemEval-2025 Task 11), `local_csv.yaml` (your own files) |
| `run_experiments.py` | CLI entry point |
| `Notebooks/5_Layerwise_Emotion_Probing.ipynb` | Colab-ready walkthrough of a run and its results |
| `tests/` | `python -m pytest tests/ -q` — 81 tests, no network needed |

Alongside the probing tables the run reports two diagnostics that explain the
numbers: a per-layer language-identification probe (how language-specific is
this layer?) and per-layer cross-lingual CKA (how similarly does it organise
two languages?).

The two parts of the repository are independent; `layerprobe/` does not import
`Utils.py` or `evaluation_functions.py`, and has its own dependencies in
`requirements.txt`.
