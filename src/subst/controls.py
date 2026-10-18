"""Controls: (1) matched MLP -- is linear actually enough?
            (2) random-init model -- does high R^2 need learned weights?"""
import torch
import torch.nn as nn
from .ridge import r2


def mlp_control(cached_xy, val_xy, d, device, epochs=5, lr=1e-3, width=None):
    """cached_xy/val_xy: {layer: (X, Y)} fp16 CPU tensors."""
    rows = []
    for l, (X, Y) in cached_xy.items():
        mlp = nn.Sequential(nn.Linear(d, width or d), nn.GELU(),
                            nn.Linear(width or d, d)).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=4096,
            shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.float().to(device), yb.float().to(device)
                loss = ((mlp(xb) - yb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        Xv, Yv = (t.float().to(device) for t in val_xy[l])
        with torch.no_grad():
            s = r2(Yv, mlp(Xv))
        rows.append(dict(layer=l, r2_mlp=s))
        print(f"  MLP L{l}: R2={s:.3f}")
        del mlp
        torch.cuda.empty_cache() if device == "cuda" else None
    return rows
