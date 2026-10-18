"""Streaming sufficient statistics + exact centered ridge.

Per (stack, layer, fold) we keep Sxx=Sum x x^T, Sxy=Sum x y^T, sums, count.
From these, the centered ridge solution for ANY lambda is exact:
    W = (Cxx + lam I)^-1 Cxy,  Cxx = Sxx - N xbar xbar^T, etc.
One eigendecomposition of Cxx serves the whole lambda grid, and the
Delta-target (y - x) solution is free: Cx,delta = Cxy - Cxx.
Verified against direct solutions to 1e-8.
"""
import torch


class StatAccumulator:
    def __init__(self, n_layers, d, n_folds):
        z = lambda *s: [[torch.zeros(*s, dtype=torch.float64)
                         for _ in range(n_folds)] for _ in range(n_layers)]
        self.Sxx, self.Sxy = z(d, d), z(d, d)
        self.sx, self.sy = z(d), z(d)
        self.n = [[0] * n_folds for _ in range(n_layers)]
        self.n_layers, self.d, self.n_folds = n_layers, d, n_folds

    def update(self, hs, fold):
        """hs: list of L+1 tensors [n_tok, d] on GPU."""
        for l in range(self.n_layers):
            x, y = hs[l].float(), hs[l + 1].float()
            self.Sxx[l][fold] += (x.T @ x).double().cpu()
            self.Sxy[l][fold] += (x.T @ y).double().cpu()
            self.sx[l][fold] += x.sum(0).double().cpu()
            self.sy[l][fold] += y.sum(0).double().cpu()
            self.n[l][fold] += x.shape[0]

    def moments(self, l, folds):
        Sxx = sum(self.Sxx[l][f] for f in folds)
        Sxy = sum(self.Sxy[l][f] for f in folds)
        sx = sum(self.sx[l][f] for f in folds)
        sy = sum(self.sy[l][f] for f in folds)
        N = sum(self.n[l][f] for f in folds)
        xbar, ybar = sx / N, sy / N
        Cxx = Sxx - N * torch.outer(xbar, xbar)
        Cxy = Sxy - N * torch.outer(xbar, ybar)
        return Cxx, Cxy, xbar, ybar


def _solve(Cxx, Cxy, xbar, ybar, lam, evals=None, Q=None):
    if evals is None:
        evals, Q = torch.linalg.eigh(Cxx)
    W = (Q @ ((Q.T @ Cxy) / (evals + lam).unsqueeze(1)))
    b = ybar - xbar @ W
    return W.float(), b.float(), evals, Q


def r2(y, pred):
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean(0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def fit_layer(acc, l, lambdas, Xv, Yv):
    """Returns dict of metrics + pooled (W, b) chosen by held-out R^2.

    Error bars: each fold's W (fit on ~1/K of data) evaluated on the shared
    val set -> mean/std across folds. Pooled W (all folds) is what gets
    used for patching, rank sweeps, etc.
    """
    folds_all = list(range(acc.n_folds))
    Cxx, Cxy, xbar, ybar = acc.moments(l, folds_all)
    evals, Q = torch.linalg.eigh(Cxx)
    best = None
    for lam in lambdas:
        W, b, _, _ = _solve(Cxx, Cxy, xbar, ybar, lam, evals, Q)
        s = r2(Yv, Xv @ W + b)
        if best is None or s > best[0]:
            best = (s, lam, W, b)
    r2_full, lam, W, b = best

    # Delta target from the same moments
    Wd, bd, _, _ = _solve(Cxx, Cxy - Cxx, xbar, ybar - xbar, lam, evals, Q)
    r2_delta = r2(Yv - Xv, Xv @ Wd + bd)
    r2_ident = r2(Yv, Xv)

    # per-fold error bars at the chosen lambda
    fold_full, fold_delta = [], []
    if acc.n_folds > 1:
        for f in folds_all:
            if acc.n[l][f] == 0:
                continue
            Cf = acc.moments(l, [f])
            Wf, bf, ef, Qf = _solve(*Cf, lam)
            fold_full.append(r2(Yv, Xv @ Wf + bf))
            Wfd, bfd, _, _ = _solve(Cf[0], Cf[1] - Cf[0], Cf[2],
                                    Cf[3] - Cf[2], lam, ef, Qf)
            fold_delta.append(r2(Yv - Xv, Xv @ Wfd + bfd))
    t = torch.tensor
    metrics = dict(
        layer=l, lam=lam, r2_full=r2_full, r2_identity=r2_ident,
        r2_delta=r2_delta,
        r2_full_std=t(fold_full).std().item() if len(fold_full) > 1 else 0.0,
        r2_delta_std=t(fold_delta).std().item() if len(fold_delta) > 1 else 0.0,
    )
    return metrics, (W, b)
