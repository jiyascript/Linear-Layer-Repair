"""Drop-in layer replacements.

transformers==4.46 convention (pinned in requirements): decoder layers
(Llama/Qwen/GPTNeoX) and encoder layers (BERT/RoBERTa) return TUPLES whose
first element is hidden_states. We always run evals with use_cache=False and
output_attentions=False, so a one-tuple return is sufficient.
"""
import torch
import torch.nn as nn
from contextlib import contextmanager


class LinearPatch(nn.Module):
    """h -> h W + b, fp32 matmul for numerical stability."""
    def __init__(self, W, b, device):
        super().__init__()
        self.register_buffer("W", W.to(device))
        self.register_buffer("b", b.to(device))

    def forward(self, hidden_states, *args, **kwargs):
        h = hidden_states.float() @ self.W + self.b
        return (h.to(hidden_states.dtype),)


class SkipPatch(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


class MixturePatch(nn.Module):
    """Mixture of linear maps with hard k-means routing.

    Each token is routed to its nearest centroid and transformed by that
    cluster's map: h -> h W_j + b_j for j = argmin_j ||h - c_j||.

    NOTE: this is a reconstruction of the `mixture_k4` strategy from the
    method as described. The code that produced the published mixture_k4
    numbers in results/*/repair.csv was lost, so values from this
    implementation may differ from those in the paper.
    """
    def __init__(self, centroids, Ws, bs, device):
        super().__init__()
        self.register_buffer("cent", centroids.to(device).float())
        self.register_buffer("W", Ws.to(device).float())
        self.register_buffer("b", bs.to(device).float())

    def forward(self, hidden_states, *args, **kwargs):
        h = hidden_states.float()
        shape = h.shape
        flat = h.reshape(-1, shape[-1])
        assign = torch.cdist(flat, self.cent).argmin(1)
        out = torch.empty_like(flat)
        for j in range(self.cent.shape[0]):
            m = assign == j
            if m.any():
                out[m] = flat[m] @ self.W[j] + self.b[j]
        return (out.reshape(shape).to(hidden_states.dtype),)


@contextmanager
def patched(layers, replacements):
    """replacements: {index: module}. Restores originals on exit."""
    originals = {i: layers[i] for i in replacements}
    for i, m in replacements.items():
        layers[i] = m
    try:
        yield
    finally:
        for i, m in originals.items():
            layers[i] = m


def linear_patches(Ws, idxs, device):
    return {i: LinearPatch(*Ws[i], device) for i in idxs}


def skip_patches(idxs):
    return {i: SkipPatch() for i in idxs}
