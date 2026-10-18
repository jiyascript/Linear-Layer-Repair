# Linear Layer Repair

Code and experimental pipeline for **"Linear Layer Repair: Residual-Controlled Layer
Profiling for Fault-Tolerant Transformer Inference"** (BlackboxNLP 2026).

When a transformer layer is unavailable, which layers tolerate a drop-in linear
substitute, and at what functional cost?

The short answer the pipeline measures: the conventional way of asking this —
regressing a layer's *output* on its *input* — is uninformative, because the
residual stream makes the identity map a strong baseline (untrained networks
reach full-state R² of 0.95–0.99). Measuring the linear predictability of the
**update** (Δ-R²) instead reveals a U-shaped depth profile that is consistent
across six models and three architecture families, and that profile predicts
which layers a linear patch can actually repair.

📄 [Abstract](https://blackboxnlp.github.io/2026/accepted_papers/) 

## Install

```bash
pip install -r requirements.txt
pip install lm_eval==0.4.5    # only for --stage tasks
```

`transformers` is **pinned to 4.46.3**: layer patching relies on the 4.46
tuple-return convention for decoder/encoder layers. Llama-3.2-1B is gated —
accept the license on Hugging Face and authenticate with
`huggingface_hub.login()` before running it.

Verify the install without downloading large checkpoints:

```bash
python src/test_pipeline.py    # offline end-to-end test, all architecture families
```

## Run

```bash
python src/run.py --model MODEL --stage STAGE [STAGE ...] [--out results] [--force]
```

Models are registered in `src/subst/config.py`: `qwen2-1.5b`, `llama3.2-1b`,
`bert-base`, `roberta-base`, `bart-base`, `t5-base`, `pythia-160m`,
`pythia-410m`, `pythia-1.4b`.

Reproducing the paper's results:

```bash
# Residual-controlled linearity (the Δ-R² measurement)
python src/run.py --model qwen2-1.5b  --stage fit
python src/run.py --model llama3.2-1b --stage fit

# Functional evidence (decoders)
python src/run.py --model qwen2-1.5b  --stage ppl multi rank
python src/run.py --model llama3.2-1b --stage ppl multi rank
python src/run.py --model qwen2-1.5b  --stage tasks      # after ppl
python src/run.py --model llama3.2-1b --stage tasks

# Repair strategies on the hardest (boundary) layers
python src/run.py --model qwen2-1.5b  --stage repair
python src/run.py --model llama3.2-1b --stage repair

# Controls + remaining architecture families
python src/run.py --model qwen2-1.5b   --stage mlp randinit
python src/run.py --model llama3.2-1b  --stage mlp randinit
python src/run.py --model bert-base    --stage fit ppl multi mlp randinit
python src/run.py --model roberta-base --stage fit ppl multi mlp randinit
python src/run.py --model bart-base    --stage fit randinit   # representational only
python src/run.py --model t5-base      --stage fit randinit

# Scale ladder
python src/run.py --model pythia-160m --stage fit
python src/run.py --model pythia-410m --stage fit
python src/run.py --model pythia-1.4b --stage fit
```

Rough single-T4 budget: `fit` 40–80 min per model (6k documents); `ppl` across
four domains 60–120 min; `multi` 60–90 min; `rank`, `mlp`, `randinit` 20–40 min
each; `tasks` 30–60 min at `limit=500`.

### Checkpointing

Stages whose output already exists are skipped; `--force` redoes them. The
`ppl`, `multi`, and `rank` sweeps write after every row and resume mid-stage
from the partial CSV, so an interrupted session costs one evaluation rather
than a whole sweep. Only `fit` is atomic — if your sessions are shorter than
one pass, lower `n_fit_docs` in `src/subst/config.py`. Rerunning the same
command after any interruption is safe.

## Stages and outputs

| Stage | Output | Paper artifact |
|---|---|---|
| `fit` | `r2.csv` (`r2_full`, `r2_identity`, **`r2_delta`**, ±std, per stack) | Residual-controlled linearity |
| `ppl` | `sweep_<domain>.csv` | Substitutability profile; cross-domain and length generalization |
| `multi` | `multi.csv` | Multi-layer failure (k consecutive layers) |
| `rank` | `rank.csv` | Patch size vs. function preserved |
| `mlp` | `mlp.csv` | Nonlinearity ceiling control |
| `randinit` | `r2_randinit.csv` | Architecture-vs-learning control |
| `tasks` | `tasks.csv` | Zero-shot function preservation |
| `repair` | `repair.csv` | Repair strategies on boundary layers |

### The `repair` stage

`repair` targets the layers the Δ-R² profile marks as hardest — the first two
and last two of the stack — and compares four drop-in strategies on each,
writing one row per (layer, strategy):

| Strategy | Row value | What it does |
|---|---|---|
| `skip` | `skip` | Delete the layer; the residual stream passes through |
| `linear_full` | `linear_full` | Full-rank fitted linear map |
| low-rank | `lowrank_r64`, `lowrank_r256`, `lowrank_r512` | Reduced-rank regression, fitted directly at rank *r* rather than by truncating a full fit |
| neighbour reuse | `neighbor_L<n>` | Reuse the adjacent layer's map, testing whether maps transfer across layers |
| mixture | `mixture_k4` | Mixture of linear maps: k-means routing over layer inputs, one ridge per cluster |

The low-rank fit uses the reduced-rank regression closed form — the full ridge
solution projected onto the top-*r* directions of the fitted prediction
covariance — which is the RRR optimum, not naive SVD truncation of `W`.
`repair` is decoder-only and reuses the `fit` statistics pass.

**Reproducibility note.** The `mixture_k4` implementation in
`src/subst/evals.py` (`fit_mixture`) and `src/subst/patching.py`
(`MixturePatch`) is a reconstruction from the method as described; the original
code was lost. The `mixture_k4` values in `results/*/repair.csv` are the
published ones, and rerunning this stage may not reproduce them exactly. All
other strategies are the original code.

## Figures

Once the result CSVs are in place, regenerate every figure in the paper:

```bash
python src/figures.py --results results --figs figs [--tables]
```

It only reads CSVs, so it is safe to rerun. Each figure is written as both PDF
(for LaTeX) and PNG. `--tables` additionally prints the zero-shot task table as
LaTeX.

## Design notes

- **Statistics, not storage.** One extraction pass accumulates Σxxᵀ and Σxyᵀ per
  (stack, layer, fold), where x is the layer input and y the target. The exact centered ridge solution for any λ, and the
  Δ-target fit, are recovered from these sufficient statistics rather than from
  cached activations. Verified against direct solutions to 1e-8.
- **Error bars** come from three disjoint fit folds evaluated on a shared
  validation set. The pooled-fold map is the one deployed for patching.
- **Maps are fit in the model's native representation space** — no centering or
  ℓ2 normalization of inputs — so they can be dropped in directly. The bias
  term absorbs mean structure.
- **Encoder-decoder functional patching is excluded by design.** T5 threads
  `position_bias` between blocks, so a drop-in patch breaks the chain, and BART
  lacks a clean intrinsic metric for this comparison. Both are analyzed
  representationally, including their decoder stacks.
- **MLM masking is seeded**, so every patch condition sees identical masks.
- `subst/tasks_eval.py` is the one module not covered by the offline test suite,
  since it needs network access for task data. If `lm_eval`'s API has moved,
  check the `simple_evaluate` and `HFLM` signatures.

## Layout

```
src/run.py                 orchestrator (stages, checkpointing)
src/figures.py             regenerates every paper figure from the result CSVs
src/test_pipeline.py       offline end-to-end test, all families
src/subst/config.py        model registry + run configuration
src/subst/models.py        loading, stack access, hidden-state extraction
src/subst/ridge.py         streaming statistics, exact ridge, Δ-target, error bars
src/subst/data.py          corpora (OpenWebText fit; OWT/WikiText-103/long/code eval)
src/subst/patching.py      LinearPatch / SkipPatch with restore-on-exit
src/subst/evals.py         CLM perplexity, MLM accuracy, single/multi/rank sweeps
src/subst/controls.py      MLP control
src/subst/tasks_eval.py    lm-eval-harness zero-shot (decoders)
notebooks/colab_driver.ipynb     driver notebook with saved run outputs
notebooks/colab_encoders.ipynb   encoder-family driver
results/                   per-model CSVs backing the paper's tables and figures
figs/                      generated figures (PDF + PNG)
```

## Citation

```bibtex
@inproceedings{rathi2026linear,
  title     = {Linear Layer Repair: Residual-Controlled Layer Profiling for
               Fault-Tolerant Transformer Inference},
  author    = {Rathi, Jiya Sheetal and Bauer, Andr{\'e}},
  booktitle = {Proceedings of the 9th BlackboxNLP Workshop},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
