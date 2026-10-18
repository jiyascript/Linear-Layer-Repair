"""Paper figures.

Reads the per-model result CSVs and writes PDF (for LaTeX) plus PNG previews.
Only reads CSVs, so it is safe to rerun at any time.

    python src/figures.py --results results --figs figs [--tables]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = {"qwen2-1.5b": "Qwen2-1.5B", "llama3.2-1b": "LLaMA-3.2-1B"}
MARKS = {"qwen2-1.5b": "o", "llama3.2-1b": "^"}
C = {"full": "#2a78d6", "ident": "#898781", "delta": "#eb6834",
     "skip": "#898781", "linear": "#2a78d6", "rand": "#b3b0a9"}

# Intact (unpatched) WikiText-103 perplexity, drawn as a reference line where known.
INTACT = {"qwen2-1.5b": 15.172}

# ACL-friendly styling: serif, small, crisp.
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", help="directory of per-model result CSVs")
    ap.add_argument("--figs", default="figs", help="output directory for figures")
    ap.add_argument("--tables", action="store_true", help="also print the zero-shot task table as LaTeX")
    args = ap.parse_args()

    out, fig_dir = args.results, args.figs
    os.makedirs(fig_dir, exist_ok=True)

    def load(key, name):
        p = os.path.join(out, key, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    def save(fig, name):
        fig.savefig(os.path.join(fig_dir, f"{name}.pdf"))
        fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=300)
        plt.close(fig)
        print("saved", name)

    def depth(df):  # normalized depth, for cross-model comparison
        return df.layer / df.layer.max()

    def main_stack(df):
        return df[df["stack"] == "main"] if "stack" in df else df

    # ---- Fig 1 (central): Delta-R^2 U-shape, trained vs. random init ----
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for key, label in MODELS.items():
        r2, rnd = load(key, "r2.csv"), load(key, "r2_randinit.csv")
        if r2 is None:
            continue
        r2 = main_stack(r2)
        ax.errorbar(depth(r2), r2.r2_delta, yerr=r2.get("r2_delta_std"),
                    marker=MARKS[key], ms=3, lw=1.3, label=f"{label} (trained)",
                    color=C["delta"] if "llama" in key else C["full"])
        if rnd is not None:
            rnd = main_stack(rnd)
            rnd = rnd[rnd.layer < rnd.layer.max()]  # drop last-layer norm artifact
            ax.plot(depth(rnd), rnd.r2_delta, MARKS[key], ls="--", ms=2, lw=1,
                    color=C["rand"], label=f"{label} (random init)")
    ax.set(xlabel="normalized depth", ylabel="update ($\\Delta$) held-out $R^2$",
           ylim=(0, 1.05))
    ax.legend(frameon=False, loc="lower center", ncol=2, handlelength=1.6,
              fontsize=7)
    save(fig, "fig1_delta_ushape")

    # ---- Fig 1 (all families): Delta-R^2 across three architecture families ----
    # One panel per family. Trained models in color, random-init controls as grey
    # dashed lines. Encoder-decoders show both stacks and have no random-init run.
    PANELS = [
        ("decoder-only", [("qwen2-1.5b", "main", "Qwen2-1.5B", "#2a78d6", "o"),
                          ("llama3.2-1b", "main", "LLaMA-3.2-1B", "#eb6834", "^")]),
        ("encoder-only", [("bert-base", "main", "BERT-base", "#2e9e5b", "s"),
                          ("roberta-base", "main", "RoBERTa-base", "#8d56c4", "D")]),
        ("encoder--decoder", [("bart-base", "enc", "BART-base (enc)", "#d4a017", "v"),
                              ("bart-base", "dec", "BART-base (dec)", "#d4a017", "v"),
                              ("t5-base", "enc", "T5-base (enc)", "#c0392b", "P"),
                              ("t5-base", "dec", "T5-base (dec)", "#c0392b", "P")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), sharey=True)
    for ax, (fam, entries) in zip(axes, PANELS):
        drew_rand = False
        for key, stack, label, color, mark in entries:
            r2 = load(key, "r2.csv")
            if r2 is None:
                continue
            d = r2[r2["stack"] == stack] if "stack" in r2 else r2
            if d.empty:
                continue
            ax.plot(depth(d), d.r2_delta, marker=mark, ms=3.5, lw=1.3,
                    color=color, label=label)
            rnd = load(key, "r2_randinit.csv")
            if rnd is not None:
                rd = rnd[rnd["stack"] == stack] if "stack" in rnd else rnd
                rd = rd[rd.layer < rd.layer.max()]  # drop last-layer norm artifact
                if not rd.empty:
                    ax.plot(depth(rd), rd.r2_delta, ls="--", lw=1.1,
                            marker=mark, ms=2, color=C["rand"],
                            label="random init" if not drew_rand else None)
                    drew_rand = True
        ax.set(title=fam, xlabel="normalized depth", ylim=(0, 1.05))
        ax.legend(frameon=False, fontsize=7, handlelength=1.8)
    axes[0].set_ylabel("update $\\Delta$-$R^2$")
    save(fig, "fig1_delta_ushape_all")

    # ---- Fig 2: three R^2 curves per model (full / identity / Delta) ----
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    for ax, (key, label) in zip(axes, MODELS.items()):
        r2 = load(key, "r2.csv")
        if r2 is None:
            continue
        r2 = main_stack(r2)
        ax.plot(r2.layer, r2.r2_full, "o-", ms=2.5, lw=1.2, color=C["full"],
                label="linear (full state)")
        ax.plot(r2.layer, r2.r2_identity, "s--", ms=2.5, lw=1.2, color=C["ident"],
                label="identity (residual)")
        ax.plot(r2.layer, r2.r2_delta, "^-", ms=2.5, lw=1.2, color=C["delta"],
                label="linear ($\\Delta$ target)")
        ax.set(title=label, xlabel="layer", ylim=(-0.5, 1.05))
    axes[0].set_ylabel("held-out $R^2$")
    axes[0].legend(frameon=False, loc="lower center")
    save(fig, "fig2_r2_three_curves")

    # ---- Fig 3: substitutability profile (WikiText-103 perplexity, log scale) ----
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for ax, (key, label) in zip(axes, MODELS.items()):
        sw = load(key, "sweep_wikitext103.csv")
        if sw is None:
            continue
        ax.plot(sw.layer, sw.skip, "s--", ms=2.5, lw=1.2, color=C["skip"], label="skip")
        ax.plot(sw.layer, sw.linear, "o-", ms=2.5, lw=1.2, color=C["linear"],
                label="linear patch")
        if key in INTACT:
            ax.axhline(INTACT[key], color="k", lw=0.8, ls=":", label="intact")
        ax.set(title=label, xlabel="replaced layer", yscale="log")
    axes[0].set_ylabel("WikiText-103 perplexity")
    axes[0].legend(frameon=False)
    save(fig, "fig3_substitutability_profile")

    # ---- Fig 4: multi-layer failure, best mid-depth window per k ----
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for key, label in MODELS.items():
        mu = load(key, "multi.csv")
        if mu is None:
            continue
        mid = mu[(mu.start >= mu.start.max() * .25) & (mu.start <= mu.start.max() * .75)]
        best = mid.groupby("k").agg(skip=("skip", "min"), linear=("linear", "min"))
        m = MARKS[key]
        ax.plot(best.index, best.skip, m + "--", ms=4, lw=1.2, color=C["skip"])
        ax.plot(best.index, best.linear, m + "-", ms=4, lw=1.2, color=C["linear"])
        ax.plot([], [], m, color="k", label=label)
    ax.plot([], [], "--", color=C["skip"], label="skip")
    ax.plot([], [], "-", color=C["linear"], label="linear")
    ax.set(xlabel="k consecutive layers replaced (best mid window)",
           ylabel="WikiText-103 ppl", yscale="log", xticks=[1, 2, 3, 4])
    ax.legend(frameon=False)
    save(fig, "fig4_multilayer")

    # ---- Fig 5: rank sweep ----
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for key, label in MODELS.items():
        rk = load(key, "rank.csv")
        if rk is None:
            continue
        mid = rk[rk.layer == sorted(rk.layer.unique())[1]]   # 50%-depth layer
        ax.plot(mid["rank"], mid.metric, MARKS[key] + "-", ms=4, lw=1.2,
                color=C["delta"], label=label)
    ax.set(xlabel="rank of $W$ (50% depth layer)", ylabel="WikiText-103 ppl",
           xscale="log", yscale="log")
    ax.legend(frameon=False)
    save(fig, "fig5_rank")

    # ---- Fig 6: cross-domain patch cost (linear vs. skip) ----
    doms = ["openwebtext", "wikitext103", "wikitext103_long", "code"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=False)
    for ax, (key, label) in zip(axes, MODELS.items()):
        for d in doms:
            sw = load(key, f"sweep_{d}.csv")
            if sw is None:
                continue
            rel = (sw.linear - sw.skip) / sw.skip * 100   # % change vs. skip
            ax.plot(sw.layer[2:], rel[2:], "-", lw=1.1, ms=2, marker="o",
                    label=d.replace("wikitext103_long", "wikitext (1024)"))
        ax.axhline(0, color="k", lw=0.6)
        ax.set(title=label, xlabel="replaced layer (0–1 omitted: off-scale)")
    axes[0].set_ylabel("ppl change vs. skip (%)\n(negative = patch helps)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig6_crossdomain")

    if args.tables:
        for key, label in MODELS.items():
            t = load(key, "tasks.csv")
            if t is None:
                continue
            print(f"\n% ---- {label} tasks table ----")
            print(t.to_latex(index=False, float_format="%.3f"))

    print("\nAll figures in:", fig_dir)


if __name__ == "__main__":
    main()
