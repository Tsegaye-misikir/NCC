# Which transformer layers make good emotion representations for low-resource languages?

This document describes the study implemented in [`layerprobe/`](../layerprobe):
what it measures, how to run it, how to read what comes out, and what it
cannot tell you.

---

## 1. The question

A multilingual encoder such as XLM-R does not distribute information evenly
across its layers. Lower layers stay close to the surface form and keep
language identity readily available; middle layers are the most
language-neutral, which is why word-alignment and cross-lingual retrieval
work best there; upper layers specialise back towards the pretraining
objective, which for a masked language model means re-introducing
language-specific, token-level information.

Almost every emotion classifier nonetheless takes the **final** layer, because
that is what `AutoModel(...).last_hidden_state` hands you. For a
high-resource language that is usually fine. For a low-resource language,
where a probe has a few hundred training sentences and often has to lean on
cross-lingual transfer from English, the default may be actively bad: if the
final layer keeps language identity linearly available, a probe trained on
English can key on English-specific directions that mean nothing in Amharic.

So the study asks three things:

1. **Which single layer** gives the best emotion representation, and does the
   answer differ between high- and low-resource languages?
2. **Do simple combinations** — an unweighted average over a window of
   layers, or a learned weighted mixture — beat the best single layer?
3. **Does a better representation reduce negative cross-lingual transfer**,
   i.e. close the gap between a zero-shot probe and an in-language one?

## 2. Design

### Frozen encoder, weak probe

The encoder is never fine-tuned. Fine-tuning would let the model *move* the
information to wherever the classifier needs it, which would answer a
different (also interesting, but different) question. Freezing it means the
comparison is about what the pretrained representation already contains.

For the same reason the probe is deliberately weak. A high-capacity
classifier can extract the emotion signal from almost any layer given enough
parameters, flattening exactly the differences we want to see. The default
probe is one-vs-rest logistic regression; `probe.kind: mlp` runs a
one-hidden-layer network as a robustness check.

### Pooling

Each layer's token states are pooled into one vector per sentence with a
**mask-aware mean** (padding excluded). `cls`, `max` and `last_token` are
available. Mean pooling is the default because `<s>`/CLS is untrained in a
model that was never fine-tuned with a sentence-level objective, which
systematically disadvantages the upper layers and would bias the comparison.

For a **decoder-only** model use `last_token` or `mean`. `cls` is rejected
outright there: under causal attention the first token has attended to
nothing but itself, so it carries no sentence content. `last_token` is the
only sound single-token read-out, because the final real token is the only
position that has seen the whole sentence. It is located from the attention
mask rather than by taking `[:, -1]`, so it is correct whether the tokenizer
pads left (the usual default for causal LMs) or right.

Pooled features are standardised before the probe sees them
(`probe.standardize`). This matters more than it sounds: activation norms
differ by an order of magnitude between layers, and without standardisation
the regularisation strength `C` means something different at each layer, so
the layer comparison would be confounded by scale.

### What is compared

| Family | Example | What it tests |
| --- | --- | --- |
| single layer | `layer0` … `last` | the layer-by-layer profile |
| final layer | `last` | the default everyone uses — the baseline |
| average over all | `avg_all` | is a free, tuning-less average enough? |
| sliding windows | `avg6-9` | is there a contiguous band that carries emotion? |
| thirds | `avg_bottom`, `avg_middle`, `avg_top` | coarse depth regions |
| concatenation | `concat_mid_top` | more expressive, but more probe parameters |
| learned mixture | `scalar_mix` | what the task asks for when allowed to choose |

`scalar_mix` is an ELMo-style scalar mix: `γ · Σ_l softmax(w)_l · LayerNorm(h_l)`,
with `w` and `γ` learned jointly with a linear classifier. Its learned
weights are reported (`layer_weights_mean`, and `scalar_mix_weights.png`) and
are a result in their own right — they locate the useful layers without a
grid search. The per-layer `LayerNorm` is what stops a high-norm layer from
winning the mixture on scale alone.

### The three regimes

- **`monolingual`** — train and test in the same language. Establishes the
  in-language ceiling and shows whether the best layer moves with training
  set size.
- **`zeroshot`** — train on the source language(s), test on a target never
  seen in training. Model selection (`C`, threshold) stays on **source**
  dev data; using target dev would make it few-shot, not zero-shot.
- **`multilingual`** — train once on all languages pooled, test per language.
  Shows whether representation choice also reduces the interference joint
  training introduces.

### Metrics

**Macro-F1** is the headline. Emotion corpora for low-resource languages are
small and skewed; accuracy and micro-F1 both reward a probe for ignoring the
rare emotions, which is precisely the failure mode we care about. The
multi-label decision threshold is tuned on dev (a single global threshold —
per-emotion thresholds overfit a few hundred dev examples).

Two derived columns carry the cross-lingual story:

- `transfer_gap = in_language_macro_f1 − zero_shot_macro_f1`. Positive means
  training out-of-language cost you performance — the negative-transfer
  regime. Negative means zero-shot beat the language's own probe, which does
  happen when the target's training set is tiny.
- `above_majority = macro_f1 − majority_macro_f1`. The sanity check. A
  cross-lingual probe that does not clear the majority-class baseline has
  transferred nothing, however respectable its F1 looks.

### Error bars

Each configuration is run over `seeds`. A logistic probe is deterministic, so
seeds alone would produce zero variance; each seed therefore also draws its
own 90% subsample of the training set (`probe.train_subsample`). The
resulting standard deviation measures the thing worth measuring for a
low-resource corpus: how sensitive a layer's advantage is to *which* few
hundred sentences you happened to train on. All combinations at a given seed
see the same subsample, so the comparison between layers stays paired.

`layerprobe.metrics.paired_bootstrap` turns two configurations' per-seed
scores into a p-value. **Use it before claiming a layer is better** — with
three seeds, a one-point macro-F1 gap is usually noise.

### Diagnostics

Two measurements explain *why* a layer transfers:

- **Language-identification probe** — a linear classifier predicting which
  language a sentence is in, from the same pooled vector. High accuracy means
  the layer keeps language identity linearly available, which is what lets a
  cross-lingual probe latch onto language-specific directions.
  `language_specificity` rescales this to 0 at chance and 1 at perfect
  identifiability (it can dip slightly below 0 as sampling noise, and is
  deliberately not clamped).
- **Linear CKA** between two languages' representations at the same layer:
  how similarly does the layer *organise* the two sets of sentences? CKA is
  invariant to rotation and isotropic scaling, which is what makes it usable
  across languages.
- **Centroid cosine** between the two languages' mean vectors: do the two
  clouds sit in the same *place*?

The last two are not redundant, and the difference is easy to get wrong. CKA
centres each side before comparing, so it is **blind to a constant offset
between two languages** — and a constant offset is precisely what breaks a
linear probe carried from one language to another. Expect a layer to be able
to show high CKA and still transfer badly, and expect centroid cosine to be
the better predictor of zero-shot macro-F1. (The offline smoke run reproduces
exactly this: centroid cosine peaks where zero-shot performance peaks, while
CKA stays flat.) Report both.

`diagnostics_correlation` in `results.json` correlates all three against
zero-shot macro-F1 across layers. With 13 layers these correlations are
**indicative, not confirmatory** — they are a hypothesis generator, and the
`n_layers` field is reported next to them so nobody forgets the sample size.

## 3. Running it

```bash
pip install -r requirements.txt
```

### Offline smoke run (no downloads, about a minute)

```bash
python run_experiments.py --config configs/smoke.yaml
```

This uses a synthetic corpus and a synthetic encoder. It exists to prove the
pipeline works end to end and to show you the shape of every output file.
**Its numbers are not findings** — the synthetic encoder has a
language-neutrality curve written into it by hand.

### The real study

```bash
python run_experiments.py --config configs/brighter.yaml
```

`configs/brighter.yaml` targets the SemEval-2025 Task 11 / BRIGHTER
multi-label emotion corpora with `xlm-roberta-base`. **Check `hf_path` and
`hf_name_template` against the dataset card before the first run** — dataset
ids and per-language config names move, and a few languages use a different
emotion inventory. A mismatch shows up immediately as an all-zero column in
`data_summary.csv`, which is why that file is written first.

For your own data, see `configs/local_csv.yaml`.

Useful overrides:

```bash
python run_experiments.py --config configs/brighter.yaml \
    --model xlm-roberta-large \
    --languages eng amh hau tir \
    --source-languages eng \
    --seeds 1 2 3 4 5
```

### Cost

Features are extracted once per (language, split) and cached in `cache_dir`,
keyed by a fingerprint of the encoder settings *and* the texts, so a stale
cache cannot silently poison a run. Extraction of ~20k sentences with
`xlm-roberta-base` is a few minutes on a single GPU and roughly half an hour
on CPU.

After that the probes dominate, and they are CPU-bound regardless of GPU.
Measured on a 4-core machine (one 6-way one-vs-rest fit at n=2000, d=768
takes ~2 s), `configs/brighter.yaml` costs roughly **2.3 hours** of probe
fitting. Two things keep that manageable:

- The cross-lingual runners fit each (combination, seed) **once** and score
  it against every target language, rather than refitting per target. The
  fit depends only on train and dev, so this is exact — but it is worth
  ~4× on the full config (9.7 h → 2.3 h), almost all of it in the
  multilingual setting.
- `configs/brighter_cpu.yaml` is a ~10-minute first pass (4 languages, 800
  training examples, 2 seeds, one `C`, named windows only) that shares the
  feature cache with the full run.

The knobs that matter most, in order: number of `seeds`, length of
`probe.C`, `combinations.window_sizes` (each size adds ~10 combinations),
and `max_train_per_language`.

Cache size is the thing to watch: all 13 layers of 20k sentences at 768
dimensions is about 800 MB compressed. Restrict `encoder.layers` if that is a
problem.

### Reproducibility

A run is reproducible across processes and machines: same config, same
numbers. Anything seeded is seeded from `layerprobe.data.stable_seed`, never
from Python's `hash()`, which is salted per process — seeding synthetic data
from `hash("amh")` silently produced a different corpus on every run, which
also defeated the feature cache. `tests/test_data.py` guards this in a
separate interpreter, since a same-process check cannot see the salt change.

## 4. Output

Written to `output_dir`:

| File | Contents |
| --- | --- |
| `SUMMARY.md` | the digest: best combination per setting, rankings, transfer gaps |
| `results.csv` | every (experiment, language, combination) with mean/std |
| `best.csv` | winner per setting, with `gain_over_last` |
| `layer_ranking.csv` | mean rank of each combination across languages |
| `language_probe.csv` | per-layer language identifiability |
| `alignment.csv` | per-layer cross-lingual CKA and centroid cosine |
| `data_summary.csv` | split sizes and per-emotion positive rates |
| `results.json` | everything above, plus per-seed scores and layer weights |
| `config.yaml` | the exact configuration that produced these files |
| `*.png` | layer curves, combination bars, mix weights, diagnostics |

### Reading it

Start with `SUMMARY.md`, then:

- **`layer_curve_zeroshot.png`** is the main figure. A curve that peaks well
  before the final layer is the study's central claim, made visible.
- **`gain_over_last` in `best.csv`** is the headline number: how much you were
  losing by taking the default.
- **`layer_ranking.csv`** is what you should actually act on. A combination
  that wins in one language and collapses in another is not a recommendation;
  mean rank across languages is what survives.
- **`scalar_mix_weights.png`** should broadly agree with the single-layer
  curve. If it does not, something is off — most often that a layer's
  activation norm is dominating, so check `scalar_mix.layer_norm`.

## 4b. Comparing several models

`configs/models/` holds one config per encoder, each `extends: _base.yaml`
so that the corpus, the splits, the seeds and the probe are defined **once**.
That is not tidiness for its own sake: if the data block drifted between
models, the differences you measure would be differences in the experiment
rather than in the model.

```bash
python run_experiments.py --config configs/models/xlmr.yaml
python run_experiments.py --config configs/models/qwen.yaml
python compare_models.py results/models/* --experiment zeroshot
```

`compare_models.py` reads each run's `results.json` and prints the best
combination per model and language, the mean gain over the final layer, and
the fertility table. It writes CSVs plus one figure with every model's layer
curve on a shared axis.

### Read layer *depth*, not layer *index*

XLM-R base has 13 layers, Llama-3.2-1B has 17, Gemma-3-1B has 27, Qwen3-0.6B
has 29. "Layer 8 is best" means something completely different in each, so
the comparison reports **`best_depth_fraction`** — the layer as a fraction of
the model's depth — and plots curves against relative depth. Raw indices are
kept in the tables but should not be compared across models.

For the same reason the model configs leave `named_windows` and
`concat_groups` empty, which makes the bottom/middle/top thirds
depth-relative. Hard-coded indices written for a 13-layer model are now
rejected at build time rather than failing deep inside the probe loop.

### The confound to check before believing any cross-model result

Tokenizer **fertility**. XLM-R's SentencePiece vocabulary was built with
Amharic and Hausa in it; an English-centric BPE can spend several times as
many tokens on the same Ge'ez text. At a fixed `max_length` the two models
therefore see *different amounts of each sentence*, and your layer comparison
quietly becomes a comparison of how much got truncated.

Every run writes `fertility` into `results.json` (tokens per character, mean
and p95 token count, and the fraction of examples truncated), and
`compare_models.py` prints a warning naming any model/language pair that
truncates more than 5%. If you see that warning, raise `max_length` for that
model before drawing conclusions — or accept that you are comparing
truncation, not representations.

### Other decoder-only caveats

- **Gating.** Llama and Gemma require accepting a licence on the Hub and
  being logged in (`huggingface-cli login`); loading fails with a 401
  otherwise. Qwen is ungated, which makes it the easiest starting point.
- **Base, not instruct.** The configs point at base checkpoints
  (`Qwen3-0.6B-Base`, `Llama-3.2-1B`, `gemma-3-1b-pt`). Instruction-tuned
  variants have been through post-training that reshapes their
  representations; mixing the two families in one comparison confounds
  architecture with alignment.
- **Cost.** 29 layers at 1024 dimensions is several times XLM-R's cache
  footprint and extraction time. Lower `batch_size`, and restrict
  `encoder.layers` if disk is tight.
- **Multilinguality is not comparable.** These decoders were not trained
  with the balanced multilingual objective XLM-R was. A finding that one
  transfers worse to Amharic may be about pretraining data, not about depth.

## 5. Limitations

- **Frozen features only.** A layer that probes badly may still fine-tune
  well. Nothing here predicts fine-tuning behaviour.
- **Pooling is a confound.** Mean pooling over a layer is not the same as the
  layer. Some of any layer-to-layer difference is a difference in how well
  that layer's geometry survives averaging. Re-running with
  `--pooling cls` is the cheap check.
- **Correlation over 13 points.** The diagnostics-versus-transfer
  correlations are suggestive at best.
- **Cross-model claims are the weakest ones here.** Two encoders differ in
  depth, tokenizer, pretraining mixture, objective and attention direction
  all at once. When Qwen and XLM-R disagree about where the best layer sits,
  the study cannot tell you which of those five things caused it. Treat the
  comparison as descriptive, and check the fertility table before treating
  it as anything more.
- **Corpus artefacts.** BRIGHTER's per-language datasets differ in domain,
  annotation guidelines and size. A per-language difference in best layer may
  be a difference in corpus, not in language. `max_train_per_language`
  equalises size, which is the one confound that is cheap to remove.

## 6. Extending it

- **Another encoder**: `--model` anything with `output_hidden_states` --
  masked (XLM-R, mBERT, LaBSE) or decoder-only (Qwen, Llama, Gemma,
  Mistral). `HFEncoder` reads depth off the model config, fills in a missing
  pad token, and rejects `cls` pooling on a causal model. Copy a file in
  `configs/models/` and change only its encoder block.
- **Another dataset**: add a loader to `layerprobe/data.py` returning
  `EmotionSplit` objects; everything downstream is agnostic.
- **Another combination**: add a `kind` to `layerprobe/combinations.py` and a
  branch in `materialize`.
- **Another diagnostic**: add it to `layerprobe/analysis.py` and reference it
  from `pipeline.run_experiment`.

Tests: `python -m pytest tests/ -q` (123 tests, ~60 s, no network). The
decoder support is covered against randomly initialised Qwen3, Llama and
Gemma3 models built from their configs, so the real transformers code path
is exercised without downloading weights.
