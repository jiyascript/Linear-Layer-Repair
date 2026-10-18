"""Orchestrator. Run per model, per stage, with everything checkpointed so a
dead Colab session costs one stage, not the run.

  python run.py --model qwen2-1.5b --stage fit        # stats + ridge + delta + error bars
  python run.py --model qwen2-1.5b --stage ppl        # single-layer sweep, all eval domains
  python run.py --model qwen2-1.5b --stage multi      # k consecutive layers
  python run.py --model qwen2-1.5b --stage rank       # SVD truncation sweep
  python run.py --model qwen2-1.5b --stage mlp        # nonlinearity control
  python run.py --model qwen2-1.5b --stage randinit   # random-init control
  python run.py --model qwen2-1.5b --stage tasks      # lm-eval zero-shot (decoders)
  python run.py --model bert-base  --stage fit ppl    # encoders: ppl stage -> MLM metrics
  python run.py --model t5-base    --stage fit        # encdec: representational only

Stage `fit` must run first for a model; later stages load its artifacts.
"""
import argparse, json, os
import torch
import pandas as pd
from subst.config import REGISTRY, RunConfig
from subst import models as M
from subst.data import load_corpora, batches
from subst.ridge import StatAccumulator, fit_layer
from subst.evals import (clm_perplexity, mlm_accuracy, sweep_single,
                         sweep_multi, rank_sweep)
from subst.controls import mlp_control


def art(cfg, name):
    os.makedirs(f"{cfg.out_dir}/{cfg.model_key}", exist_ok=True)
    return f"{cfg.out_dir}/{cfg.model_key}/{name}"


LOCAL_CACHE = os.environ.get("SUBST_CACHE", "/content/subst_cache"
                             if os.path.isdir("/content") else "subst_cache")


def cache_art(cfg, name):
    """Big rebuildable artifacts (valcache) live on LOCAL disk, not Drive:
    multi-GB files on a quota-limited Drive get silently evicted."""
    d = f"{LOCAL_CACHE}/{cfg.model_key}"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{name}"


def stage_fit(cfg, spec, tok, model, device, fit, val, model_override=None,
              tag=""):
    m = model_override or model
    d = m.config.hidden_size if hasattr(m.config, "hidden_size") else m.config.d_model
    stacks = M.layer_lists(m, spec.family)
    accs = {s: StatAccumulator(len(ls), d, cfg.n_folds)
            for s, ls in stacks.items()}
    # accumulate train stats, batches round-robin across folds
    for bi, (ids, att) in enumerate(batches(fit, cfg.batch_size,
                                            tok.pad_token_id, device)):
        hs = M.hidden_stacks(m, spec.family, ids, att, d)
        for s, h in hs.items():
            accs[s].update(h, bi % cfg.n_folds)
        if bi % 50 == 0:
            print(f"[fit{tag}] batch {bi}, tokens/stack ~ "
                  f"{sum(accs[list(accs)[0]].n[0]):,}")
    # cache val hidden states (fp16 CPU), capped at n_val_tokens
    caches = {s: None for s in stacks}
    got = 0
    for ids, att in batches(val, cfg.batch_size, tok.pad_token_id, device):
        hs = M.hidden_stacks(m, spec.family, ids, att, d)
        for s, h in hs.items():
            h = [t.cpu().half() for t in h]
            caches[s] = h if caches[s] is None else [
                torch.cat([a, b]) for a, b in zip(caches[s], h)]
        got += hs[list(hs)[0]][0].shape[0]
        if got >= cfg.n_val_tokens:
            break
    # solve
    all_rows, Ws = [], {}
    for s, acc in accs.items():
        Ws[s] = {}
        for l in range(acc.n_layers):
            Xv = caches[s][l].float()
            Yv = caches[s][l + 1].float()
            met, wb = fit_layer(acc, l, cfg.lambdas, Xv, Yv)
            met["stack"] = s
            all_rows.append(met)
            Ws[s][l] = wb
            print(f"[ridge{tag}] {s} L{l:2d} full={met['r2_full']:.3f}"
                  f"±{met['r2_full_std']:.3f} ident={met['r2_identity']:.3f}"
                  f" DELTA={met['r2_delta']:.3f}±{met['r2_delta_std']:.3f}")
    pd.DataFrame(all_rows).to_csv(art(cfg, f"r2{tag}.csv"), index=False)
    torch.save({s: {l: (W.cpu(), b.cpu()) for l, (W, b) in d_.items()}
                for s, d_ in Ws.items()}, art(cfg, f"W{tag}.pt"))
    torch.save({s: [t for t in c] for s, c in caches.items()},
               cache_art(cfg, f"valcache{tag}.pt"))
    return Ws, caches


def load_fit(cfg, device):
    Ws = torch.load(art(cfg, "W.pt"), map_location="cpu")
    Ws = {s: {int(l): (W.to(device), b.to(device))
              for l, (W, b) in d_.items()} for s, d_ in Ws.items()}
    caches = None
    for p in (cache_art(cfg, "valcache.pt"), art(cfg, "valcache.pt")):
        if os.path.exists(p):
            caches = torch.load(p, map_location="cpu")
            break
    if caches is None:
        print("[load_fit] valcache.pt missing; will rebuild if a stage needs it")
    return Ws, caches


def build_valcache(cfg, spec, tok, model, device, val, d):
    """Rebuild the held-out hidden-state cache (only rank/mlp need it)."""
    print("[valcache] rebuilding from val split (~2-3 min)...")
    caches, got = {}, 0
    for ids, att in batches(val, cfg.batch_size, tok.pad_token_id, device):
        hs = M.hidden_stacks(model, spec.family, ids, att, d)
        for s, h in hs.items():
            h = [t.cpu().half() for t in h]
            caches[s] = h if caches.get(s) is None else [
                torch.cat([a, b]) for a, b in zip(caches[s], h)]
        got += hs[list(hs)[0]][0].shape[0]
        if got >= cfg.n_val_tokens:
            break
    torch.save(caches, cache_art(cfg, "valcache.pt"))
    return caches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY))
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--out", default="results",
                    help="results dir; point at Drive, e.g. "
                         "/content/drive/MyDrive/blackboxnlp/results")
    ap.add_argument("--force", action="store_true",
                    help="redo stages even if their output CSV exists")
    args = ap.parse_args()
    if args.out.startswith("/content/drive") and not os.path.ismount(
            "/content/drive"):
        raise SystemExit("Google Drive is not mounted -- run drive.mount() "
                         "first, or results will silently go to a fake "
                         "local folder.")

    spec = REGISTRY[args.model]
    cfg = RunConfig(model_key=args.model, out_dir=args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = M.load(spec, device)
    fit, val, eval_corpora = load_corpora(tok, cfg)
    print(f"{args.model}: {len(fit)} fit / {len(val)} val docs; "
          f"eval domains: {list(eval_corpora)}")

    import os as _os

    def skip(name):
        p = art(cfg, name)
        if _os.path.exists(p) and not args.force:
            print(f"[skip] {p} exists (use --force to redo)")
            return True
        return False

    if "fit" in args.stage and not skip("W.pt"):
        Ws, caches = stage_fit(cfg, spec, tok, model, device, fit, val)
    else:
        Ws, caches = load_fit(cfg, device)

    if spec.family == "encdec":
        # functional patching excluded by design (see config.py docstring)
        return
    stack = "main"
    layers = M.layer_lists(model, spec.family)[stack]
    L = len(layers)
    Wmain = Ws[stack]

    if spec.family == "decoder":
        def eval_fn_for(blocks):
            return lambda m: round(clm_perplexity(
                m, blocks, tok.pad_token_id, cfg.batch_size, device), 3)
    else:  # encoder
        mask_id = tok.mask_token_id
        def eval_fn_for(blocks):
            return lambda m: {k: round(v, 4) for k, v in mlm_accuracy(
                m, blocks, tok.pad_token_id, mask_id, cfg.batch_size,
                device).items()}

    if "ppl" in args.stage:
        for dom, blocks in eval_corpora.items():
            if not blocks:
                continue
            ef = eval_fn_for(blocks)
            print(f"[{dom}] intact: {ef(model)}")
            sweep_single(model, layers, Wmain, ef, L, device,
                         csv_path=art(cfg, f"sweep_{dom}.csv"))

    if "multi" in args.stage:
        ef = eval_fn_for(eval_corpora["wikitext103"])
        sweep_multi(model, layers, Wmain, ef, L, cfg.multi_k, device,
                    csv_path=art(cfg, "multi.csv"))

    def ensure_caches():
        nonlocal caches
        d_ = model.config.hidden_size
        if caches is None:
            caches = build_valcache(cfg, spec, tok, model, device, val, d_)
        return caches

    if "rank" in args.stage:
        caches = ensure_caches()
        sel = [int(L * .25), int(L * .5), int(L * .75)]
        ef = eval_fn_for(eval_corpora["wikitext103"])
        rank_sweep(model, layers, Wmain, ef, sel, cfg.ranks, device,
                   Xv=caches[stack][sel[0]].float(),
                   Yv=caches[stack][sel[0] + 1].float(),
                   csv_path=art(cfg, "rank.csv"))

    if "mlp" in args.stage and not skip("mlp.csv"):
        caches = ensure_caches()
        d = model.config.hidden_size
        sel = [int(L * .25), int(L * .5), int(L * .75)]
        cached, got = {l: [[], []] for l in sel}, 0
        for ids, att in batches(fit, cfg.batch_size, tok.pad_token_id,
                                device):
            hs = M.hidden_stacks(model, spec.family, ids, att, d)[stack]
            for l in sel:
                cached[l][0].append(hs[l].cpu().half())
                cached[l][1].append(hs[l + 1].cpu().half())
            got += hs[0].shape[0]
            if got >= 100_000:
                break
        cxy = {l: (torch.cat(a), torch.cat(b)) for l, (a, b) in
               cached.items()}
        vxy = {l: (caches[stack][l], caches[stack][l + 1]) for l in sel}
        rows = mlp_control(cxy, vxy, d, device)
        pd.DataFrame(rows).to_csv(art(cfg, "mlp.csv"), index=False)

    if "randinit" in args.stage and not skip("r2_randinit.csv"):
        rnd = M.from_config_random(spec, spec.hf_name, device)
        small = RunConfig(model_key=args.model, out_dir=args.out,
                          n_fit_docs=cfg.n_fit_docs // 4, n_folds=1)
        stage_fit(small, spec, tok, model, device, fit[:small.n_fit_docs],
                  val, model_override=rnd, tag="_randinit")
        del rnd
        torch.cuda.empty_cache()

    if "repair" in args.stage and spec.family == "decoder":
        caches = ensure_caches()
        # rebuild the accumulator is expensive; instead reload W and use cached
        # sufficient stats via a fresh fit-lite is overkill -> we need `acc`.
        # Simplest: recompute acc quickly over fit set (same as stage_fit prefix).
        from subst.ridge import StatAccumulator
        d = model.config.hidden_size
        acc = StatAccumulator(L, d, cfg.n_folds)
        for bi, (ids, att) in enumerate(batches(fit, cfg.batch_size,
                                                tok.pad_token_id, device)):
            hs = M.hidden_stacks(model, spec.family, ids, att, d)[stack]
            acc.update(hs, bi % cfg.n_folds)
        ef = eval_fn_for(eval_corpora["wikitext103"])
        # hard layers = boundaries: first two + last two
        hard = [0, 1, L-2, L-1]
        from subst.evals import repair_hard_layers
        repair_hard_layers(model, layers, Wmain, acc, caches, stack, ef, L,
                           device, cfg.lambdas, hard, [64, 256, 512],
                           csv_path=art(cfg, "repair.csv"))

    if "tasks" in args.stage and spec.family == "decoder":
        from subst.tasks_eval import task_conditions
        # pick best/worst mid layers from the wikitext sweep
        sw = pd.read_csv(art(cfg, "sweep_wikitext103.csv"))
        mid = sw[(sw.layer > 1) & (sw.layer < L - 1)]
        best = int(mid.sort_values("linear").iloc[0].layer)
        worst = int(mid.sort_values("linear").iloc[-1].layer)
        conds = {"intact": None, f"best_L{best}": [best],
                 f"worst_L{worst}": [worst],
                 "mid_pair": [int(L * .5), int(L * .5) + 1]}
        task_conditions(model, tok, layers, Wmain, conds, device,
                        csv_path=art(cfg, "tasks.csv"))


if __name__ == "__main__":
    main()
