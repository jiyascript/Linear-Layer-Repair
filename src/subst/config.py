"""Model registry + run configuration.

Family determines: how layers are accessed, which hidden-state stacks exist,
and which functional evaluations apply.
  decoder : causal LM perplexity + zero-shot tasks + layer patching
  encoder : MLM masked-token accuracy + layer patching
  encdec  : representational analysis on BOTH stacks (patching excluded --
            T5 threads position_bias between blocks, so a drop-in patch
            breaks the chain; BART enc patching is possible but has no
            clean functional metric. Documented in README.)
"""
from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    hf_name: str
    family: str            # 'decoder' | 'encoder' | 'encdec'
    patchable: bool = True # supports functional layer replacement


REGISTRY = {
    "qwen2-1.5b":   ModelSpec("Qwen/Qwen2-1.5B", "decoder"),
    "llama3.2-1b":  ModelSpec("meta-llama/Llama-3.2-1B", "decoder"),
    "bert-base":    ModelSpec("bert-base-uncased", "encoder"),
    "roberta-base": ModelSpec("roberta-base", "encoder"),
    "bart-base":    ModelSpec("facebook/bart-base", "encdec", patchable=False),
    "t5-base":      ModelSpec("t5-base", "encdec", patchable=False),
    # optional scale ladder (identical architecture/data across sizes):
    "pythia-160m":  ModelSpec("EleutherAI/pythia-160m", "decoder"),
    "pythia-410m":  ModelSpec("EleutherAI/pythia-410m", "decoder"),
    "pythia-1.4b":  ModelSpec("EleutherAI/pythia-1.4b", "decoder"),
}


@dataclass
class RunConfig:
    model_key: str = "qwen2-1.5b"
    seq_len: int = 512
    long_seq_len: int = 1024      # length-generalization eval
    batch_size: int = 8
    n_fit_docs: int = 6000        # bumped from the paper's 1500
    n_val_tokens: int = 30000     # held-out R^2 / lambda selection
    n_ppl_docs: int = 300         # per eval corpus
    n_folds: int = 3              # disjoint fit subsamples -> error bars
    lambdas: tuple = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    ranks: tuple = (8, 32, 128, 512)
    multi_k: tuple = (1, 2, 3, 4) # consecutive layers replaced together
    mlm_mask_frac: float = 0.15
    out_dir: str = "results"
    seed: int = 0
