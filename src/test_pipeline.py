"""Offline end-to-end test on tiny randomly initialized models.
Exercises: extraction, stats/folds, ridge+delta+error bars, patching,
CLM ppl sweeps (single+multi), MLM accuracy sweep, rank sweep, MLP control.
No network needed.
"""
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config,
                          Qwen2ForCausalLM, BertConfig, BertForMaskedLM,
                          BartConfig, BartForConditionalGeneration,
                          T5Config, T5ForConditionalGeneration,
                          GPTNeoXConfig, GPTNeoXForCausalLM)
from subst.config import ModelSpec, RunConfig
from subst import models as M
from subst.ridge import StatAccumulator, fit_layer
from subst.evals import (clm_perplexity, mlm_accuracy, sweep_single,
                         sweep_multi, rank_sweep)
from subst.controls import mlp_control
from subst.data import batches

torch.manual_seed(0)
device = "cpu"
D, L, V = 32, 4, 100
PAD, MASK = 0, 99

TINY = {
    "llama":  (ModelSpec("x", "decoder"), LlamaForCausalLM(LlamaConfig(
        hidden_size=D, intermediate_size=64, num_hidden_layers=L,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=V))),
    "qwen2":  (ModelSpec("x", "decoder"), Qwen2ForCausalLM(Qwen2Config(
        hidden_size=D, intermediate_size=64, num_hidden_layers=L,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=V))),
    "pythia": (ModelSpec("x", "decoder"), GPTNeoXForCausalLM(GPTNeoXConfig(
        hidden_size=D, intermediate_size=64, num_hidden_layers=L,
        num_attention_heads=4, vocab_size=V))),
    "bert":   (ModelSpec("x", "encoder"), BertForMaskedLM(BertConfig(
        hidden_size=D, intermediate_size=64, num_hidden_layers=L,
        num_attention_heads=4, vocab_size=V, pad_token_id=PAD))),
    "bart":   (ModelSpec("x", "encdec", patchable=False),
               BartForConditionalGeneration(BartConfig(
        d_model=D, encoder_ffn_dim=64, decoder_ffn_dim=64,
        encoder_layers=L, decoder_layers=L, encoder_attention_heads=4,
        decoder_attention_heads=4, vocab_size=V, pad_token_id=PAD,
        bos_token_id=1, eos_token_id=2, decoder_start_token_id=2,
        forced_eos_token_id=None, max_position_embeddings=128))),
    "t5":     (ModelSpec("x", "encdec", patchable=False),
               T5ForConditionalGeneration(T5Config(
        d_model=D, d_ff=64, d_kv=8, num_layers=L, num_heads=4,
        vocab_size=V, pad_token_id=PAD, decoder_start_token_id=0))),
}

blocks = [torch.randint(3, V - 1, (40,)).tolist() for _ in range(24)]
val_blocks = blocks[:6]
cfg = RunConfig(n_folds=3, batch_size=4,
                lambdas=(0.1, 1.0, 10.0), ranks=(4, 16))

for name, (spec, model) in TINY.items():
    model = model.eval()
    d = getattr(model.config, "hidden_size", None) or model.config.d_model
    stacks = M.layer_lists(model, spec.family)
    print(f"\n=== {name} ({spec.family}) stacks={list(stacks)} "
          f"L={[len(v) for v in stacks.values()]}")

    # ---- extraction + stats + ridge ----
    accs = {s: StatAccumulator(len(ls), d, cfg.n_folds)
            for s, ls in stacks.items()}
    for bi, (ids, att) in enumerate(batches(blocks, cfg.batch_size, PAD,
                                            device)):
        hs = M.hidden_stacks(model, spec.family, ids, att, d)
        for s, h in hs.items():
            assert len(h) == len(stacks[s]) + 1, \
                f"{name}/{s}: {len(h)} hidden vs {len(stacks[s])} layers"
            accs[s].update(h, bi % cfg.n_folds)
    caches = {}
    for s in stacks:
        parts = []
        for ids, att in batches(val_blocks, cfg.batch_size, PAD, device):
            parts.append([t.half() for t in
                          M.hidden_stacks(model, spec.family, ids, att,
                                          d)[s]])
        caches[s] = [torch.cat([p[l] for p in parts])
                     for l in range(len(stacks[s]) + 1)]
    Ws = {}
    for s, acc in accs.items():
        Ws[s] = {}
        for l in range(acc.n_layers):
            met, wb = fit_layer(acc, l, cfg.lambdas,
                                caches[s][l].float(),
                                caches[s][l + 1].float())
            Ws[s][l] = wb
            assert met["r2_full"] >= met["r2_identity"] - 0.05, met
        print(f"  ridge ok: {s} r2_full[0]={met['r2_full']:.3f} "
              f"delta={met['r2_delta']:.3f} std={met['r2_full_std']:.3f}")

    if spec.family == "encdec":
        continue

    layers = stacks["main"]
    Ll = len(layers)
    if spec.family == "decoder":
        ef = lambda m: round(clm_perplexity(m, val_blocks, PAD, 4, device), 2)
    else:
        ef = lambda m: mlm_accuracy(m, val_blocks, PAD, MASK, 4, device)

    base = ef(model)
    rows = sweep_single(model, layers, Ws["main"], ef, Ll, device)
    assert ef(model) == base, f"{name}: model not restored after patching!"
    rows_m = sweep_multi(model, layers, Ws["main"], ef, Ll, (1, 2), device)
    assert ef(model) == base, f"{name}: not restored after multi-patch!"
    rk = rank_sweep(model, layers, Ws["main"], ef, [1, 2], cfg.ranks,
                    device, Xv=caches["main"][1].float(),
                    Yv=caches["main"][2].float())
    print(f"  sweeps ok: base={base} single={len(rows)} multi={len(rows_m)} "
          f"rank={len(rk)}")

# ---- MLP control on one model ----
spec, model = TINY["llama"]
d = model.config.hidden_size
hs_all = [[], []]
for ids, att in batches(blocks, 4, PAD, device):
    h = M.hidden_stacks(model, "decoder", ids, att, d)["main"]
    hs_all[0].append(h[1].half()); hs_all[1].append(h[2].half())
cxy = {1: (torch.cat(hs_all[0]), torch.cat(hs_all[1]))}
rows = mlp_control(cxy, cxy, d, device, epochs=2)
print("\nMLP control ok:", rows)

print("\nALL PIPELINE TESTS PASSED")
