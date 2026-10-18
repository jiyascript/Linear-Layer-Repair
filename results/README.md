# Results

Per-model CSVs backing the paper's tables and figures, one directory per model,
named as in `src/subst/config.py`. Regenerate the figures from these with
`python src/figures.py --results results --figs figs`.

```
results/
  qwen2-1.5b/      r2, r2_randinit, sweep_{openwebtext,wikitext103,wikitext103_long,code},
                   multi, rank, mlp, tasks, repair
  llama3.2-1b/     r2, r2_randinit, sweep_{openwebtext,wikitext103,code},
                   multi, rank, mlp, tasks, repair
  bert-base/       r2, r2_randinit, sweep_{openwebtext,wikitext103,code}, multi
  roberta-base/    r2, r2_randinit, sweep_{openwebtext,wikitext103,code}, multi
  bart-base/       r2
  t5-base/         r2
  update_magnitude.csv
```

Coverage is uneven by design. The encoder-decoders (`bart-base`, `t5-base`) are
analyzed representationally only — functional patching is excluded for the
reasons given in the top-level README — so they carry `r2.csv` alone. The
encoders have no `rank`, `mlp`, `tasks`, or `repair` runs; those stages are
decoder-only or were not needed for the paper's claims.

## Columns

| File | Columns |
|---|---|
| `r2.csv`, `r2_randinit.csv` | `layer, lam, r2_full, r2_identity, r2_delta, r2_full_std, r2_delta_std, stack` |
| `sweep_<domain>.csv` | `layer, skip, linear` |
| `multi.csv` | `k, start, skip, linear` |
| `rank.csv` | `layer, rank, r2, metric` |
| `mlp.csv` | `layer, r2_mlp` |
| `tasks.csv` | `condition, mode, arc_easy, hellaswag, lambada_openai, piqa` |
| `repair.csv` | `layer, strategy, metric` |
| `update_magnitude.csv` | `model, trained, layer, rel_update` |

`lam` is the ridge penalty selected for that layer. `stack` distinguishes
encoder from decoder stacks in the encoder-decoder models; decoder-only and
encoder-only models use `main`. The `_std` columns are standard deviations
across the three disjoint fit folds.

For decoder models the sweep metric is perplexity (lower is better); for
encoders it is a dict of MLM accuracy and loss (higher accuracy is better).

The per-layer R² values referenced in the paper are in each model's `r2.csv`.

## Note on `update_magnitude.csv`

This file records the relative magnitude of each layer's update
(`rel_update`) for `qwen2-1.5b` and `bert-base`, trained and randomly
initialized. It was produced by an analysis run that is not part of the staged
pipeline in `src/`, so it is included as data without a corresponding `--stage`
entry.
