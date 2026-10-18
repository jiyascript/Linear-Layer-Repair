"""Zero-shot task evaluation (decoders only) via lm-eval-harness.

Pin: pip install lm_eval==0.4.5
Runs the harness on the ALREADY-PATCHED in-memory model, so the same
LinearPatch/SkipPatch machinery applies. `limit` subsamples each task for
Colab-feasible runtime; report it in the paper.

NOTE: this module needs network access to download task data; it is the one
component not covered by the offline test suite. If the lm_eval API changes,
check lm_eval.simple_evaluate / lm_eval.models.huggingface.HFLM signatures.
"""
DEFAULT_TASKS = ("lambada_openai", "hellaswag", "arc_easy", "piqa")


def run_tasks(model, tokenizer, tasks=DEFAULT_TASKS, limit=500, bs=8):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs)
    res = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), limit=limit)
    out = {}
    for t, m in res["results"].items():
        acc = m.get("acc,none", m.get("acc_norm,none"))
        out[t] = acc
    return out


def task_conditions(model, tokenizer, layers, Ws, conditions, device,
                    tasks=DEFAULT_TASKS, limit=500, csv_path=None):
    """conditions: {name: list_of_layer_idxs_or_None}. None = intact.
    Each non-intact condition runs twice: skip and linear.

    lm-eval calls model(inps) WITHOUT use_cache=False, so with caching on,
    patched layers (which return 1-tuples) break the KV-cache indexing.
    Loglikelihood scoring never needs the cache -> disable via config for
    the duration and restore after. Rows are saved incrementally.
    """
    import os
    import pandas as pd
    from .patching import patched, linear_patches, skip_patches

    rows, done = [], set()
    if csv_path and os.path.exists(csv_path):
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(r["condition"], r["mode"]) for r in rows}
        print(f"  [resume] {csv_path}: {len(rows)} rows already done")

    def save(row):
        rows.append(row)
        print(row)
        if csv_path:
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    prev_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    try:
        for name, idxs in conditions.items():
            if idxs is None:
                if (name, "intact") not in done:
                    save(dict(condition=name, mode="intact",
                              **run_tasks(model, tokenizer, tasks, limit)))
                continue
            if (name, "skip") not in done:
                with patched(layers, skip_patches(idxs)):
                    save(dict(condition=name, mode="skip",
                              **run_tasks(model, tokenizer, tasks, limit)))
            if (name, "linear") not in done:
                with patched(layers, linear_patches(Ws, idxs, device)):
                    save(dict(condition=name, mode="linear",
                              **run_tasks(model, tokenizer, tasks, limit)))
    finally:
        model.config.use_cache = prev_cache
    return rows
