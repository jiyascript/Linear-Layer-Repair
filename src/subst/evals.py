"""Functional evaluations.

Decoders : causal-LM perplexity (any corpus), single/multi-layer sweeps.
Encoders : masked-token top-1 accuracy + MLM loss under patching.
"""
import math
import torch
from .data import batches
from .patching import patched, linear_patches, skip_patches

import os
import pandas as pd


def _resume(csv_path, key_cols):
    """Return (done_keys, rows) from an existing partial CSV."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        rows = df.to_dict("records")
        done = {tuple(r[k] for k in key_cols) for r in rows}
        print(f"  [resume] {csv_path}: {len(rows)} rows already done")
        return done, rows
    return set(), []


def _append(csv_path, rows):
    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, index=False)




@torch.no_grad()
def clm_perplexity(model, blocks, pad_id, bs, device):
    nll, ntok = 0.0, 0
    for ids, att in batches(blocks, bs, pad_id, device):
        labels = ids.masked_fill(att == 0, -100)
        out = model(input_ids=ids, attention_mask=att, labels=labels,
                    use_cache=False)
        valid = (labels[:, 1:] != -100).sum().item()
        nll += out.loss.item() * valid
        ntok += valid
    return math.exp(nll / ntok)


@torch.no_grad()
def mlm_accuracy(model, blocks, pad_id, mask_id, bs, device,
                 frac=0.15, seed=0):
    """Deterministic masking so all patch conditions see identical masks."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    correct, total, loss_sum = 0, 0, 0.0
    for ids, att in batches(blocks, bs, pad_id, device):
        maskable = att.bool().cpu()
        maskable[:, 0] = False                       # keep [CLS]/<s>
        m = (torch.rand(ids.shape, generator=g) < frac) & maskable
        m = m.to(device)
        if m.sum() == 0:
            continue
        corrupted = ids.masked_fill(m, mask_id)
        labels = torch.full_like(ids, -100).masked_scatter_(m, ids[m])
        out = model(input_ids=corrupted, attention_mask=att, labels=labels)
        pred = out.logits[m].argmax(-1)
        correct += (pred == ids[m]).sum().item()
        total += m.sum().item()
        loss_sum += out.loss.item() * m.sum().item()
    return dict(mlm_acc=correct / total, mlm_loss=loss_sum / total)


def sweep_single(model, layers, Ws, eval_fn, L, device, csv_path=None):
    """Per-layer: skip vs linear patch. Incrementally saved + resumable."""
    done, rows = _resume(csv_path, ["layer"])
    for l in range(L):
        if (l,) in done:
            continue
        with patched(layers, skip_patches([l])):
            s = eval_fn(model)
        with patched(layers, linear_patches(Ws, [l], device)):
            p = eval_fn(model)
        rows.append(dict(layer=l, skip=s, linear=p))
        _append(csv_path, rows)
        print(f"  L{l:2d} skip={s}  linear={p}")
    return rows


def sweep_multi(model, layers, Ws, eval_fn, L, ks, device, csv_path=None):
    """k consecutive layers replaced together. Incrementally saved + resumable."""
    done, rows = _resume(csv_path, ["k", "start"])
    for k in ks:
        for start in range(0, L - k + 1, max(1, (L - k) // 6 or 1)):
            if (k, start) in done:
                continue
            idxs = list(range(start, start + k))
            with patched(layers, skip_patches(idxs)):
                s = eval_fn(model)
            with patched(layers, linear_patches(Ws, idxs, device)):
                p = eval_fn(model)
            rows.append(dict(k=k, start=start, skip=s, linear=p))
            _append(csv_path, rows)
            print(f"  k={k} start={start:2d} skip={s}  linear={p}")
    return rows


def rank_sweep(model, layers, Ws, eval_fn, sel_layers, ranks, device,
               Xv=None, Yv=None, csv_path=None):
    from .ridge import r2
    done, rows = _resume(csv_path, ["layer", "rank"])
    if Xv is not None:
        dev = Ws[sel_layers[0]][0].device
        Xv, Yv = Xv.to(dev), Yv.to(dev)
    for l in sel_layers:
        W, b = Ws[l]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        for r in list(ranks) + [W.shape[0]]:
            if (l, r) in done:
                continue
            Wr = (U[:, :r] * S[:r]) @ Vh[:r]
            row = dict(layer=l, rank=r)
            if Xv is not None:
                row["r2"] = r2(Yv, Xv @ Wr + b)
            from .patching import LinearPatch
            with patched(layers, {l: LinearPatch(Wr, b, device)}):
                row["metric"] = eval_fn(model)
            rows.append(row)
            _append(csv_path, rows)
            print(f"  L{l} r={r:4d} -> {row}")
    return rows


def direct_lowrank_fit(acc, l, lam, Xv, Yv, ranks):
    """Fit W constrained to rank r DIRECTLY (not SVD-truncated after full fit).
    Uses the reduced-rank regression closed form: full ridge W*, then project
    its column space onto the top-r directions of the *fitted prediction*
    covariance. Returns {r: (W_r, b, r2)}."""
    import torch
    from .ridge import _solve, r2
    Cxx, Cxy, xbar, ybar = acc.moments(l, list(range(acc.n_folds)))
    evals, Q = torch.linalg.eigh(Cxx)
    W, b, _, _ = _solve(Cxx, Cxy, xbar, ybar, lam, evals, Q)  # full ridge
    # reduced-rank regression: SVD of the *whitened* prediction, keep top-r
    # (this is the RRR optimum, unlike naive truncation of W)
    P = (Xv @ W)                      # fitted predictions on val
    Pc = P - P.mean(0)
    U, S, Vh = torch.linalg.svd(Pc, full_matrices=False)
    out = {}
    for r in ranks:
        Vr = Vh[:r].T @ Vh[:r]        # project onto top-r prediction directions
        Wr = W @ Vr
        br = ybar - xbar @ Wr.double() if hasattr(xbar,'double') else ybar - xbar @ Wr
        out[r] = (Wr.float(), br.float(), r2(Yv, Xv @ Wr.float() + br.float()))
    return out


def fit_mixture(Xv, Yv, k, lam, iters=25, seed=0, chunk=65536):
    """Fit a mixture of k linear maps with hard k-means routing.

    Clusters the layer inputs with k-means, then fits an independent centered
    ridge within each cluster. Returns (centroids [k,d], W [k,d,d], b [k,d]).

    NOTE: reconstruction of the `mixture_k4` strategy from the method as
    described; see MixturePatch in patching.py.
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, d = Xv.shape
    cent = Xv[torch.randperm(n, generator=g)[:k].to(Xv.device)].clone()

    def assign_all(C):
        out = torch.empty(n, dtype=torch.long, device=Xv.device)
        for s in range(0, n, chunk):
            out[s:s + chunk] = torch.cdist(Xv[s:s + chunk], C).argmin(1)
        return out

    assign = assign_all(cent)
    for _ in range(iters):
        for j in range(k):
            m = assign == j
            if m.any():
                cent[j] = Xv[m].mean(0)
        new = assign_all(cent)
        if torch.equal(new, assign):
            break
        assign = new

    I = torch.eye(d, dtype=torch.float64, device=Xv.device)
    Wl, bl = [], []
    for j in range(k):
        m = assign == j
        if int(m.sum()) < d:          # too few points to fit: use all of them
            m = torch.ones(n, dtype=torch.bool, device=Xv.device)
        x, y = Xv[m].double(), Yv[m].double()
        xb, yb = x.mean(0), y.mean(0)
        xc, yc = x - xb, y - yb
        W = torch.linalg.solve(xc.T @ xc + lam * I, xc.T @ yc)
        Wl.append(W.float())
        bl.append((yb - xb @ W).float())
    return cent, torch.stack(Wl), torch.stack(bl)


def repair_hard_layers(model, layers, Ws, acc, caches, stack, eval_fn, L,
                       device, lambdas, hard_layers, ranks, csv_path=None,
                       mixture_k=4):
    """For each hard layer: compare skip, full linear, direct-low-rank,
    neighbor-map-reuse, and a mixture of linear maps with k-means routing.
    Writes one row per (layer, strategy)."""
    import os, pandas as pd, torch
    from .patching import patched, LinearPatch, SkipPatch, MixturePatch
    from .ridge import r2
    rows, done = [], set()
    if csv_path and os.path.exists(csv_path):
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(r["layer"], r["strategy"]) for r in rows}
    def rec(layer, strat, mod):
        if (layer, strat) in done: return
        with patched(layers, {layer: mod}):
            m = eval_fn(model)
        rows.append(dict(layer=layer, strategy=strat, metric=m))
        if csv_path: pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  L{layer} {strat}: {m}")
    for l in hard_layers:
        if l < 0 or l >= L: continue
        W, b = Ws[l]
        rec(l, "skip", SkipPatch())
        rec(l, "linear_full", LinearPatch(W, b, device))
        # B1: direct low-rank
        Xv = caches[stack][l].float().to(device)
        Yv = caches[stack][l+1].float().to(device)
        lr = direct_lowrank_fit(acc, l, lambdas[len(lambdas)//2], Xv, Yv, ranks)
        for r,(Wr,br,_) in lr.items():
            rec(l, f"lowrank_r{r}", LinearPatch(Wr, br, device))
        # B2: reuse neighbor's map (L-1 if exists else L+1)
        nb = l-1 if (l-1) in Ws else l+1
        if nb in Ws:
            Wn, bn = Ws[nb]
            rec(l, f"neighbor_L{nb}", LinearPatch(Wn, bn, device))
        # B3: mixture of linear maps, hard k-means routing
        if mixture_k and mixture_k > 1:
            cent, Wm, bm = fit_mixture(Xv, Yv, mixture_k,
                                       lambdas[len(lambdas)//2])
            rec(l, f"mixture_k{mixture_k}", MixturePatch(cent, Wm, bm, device))
    return rows
