"""Corpora + batching.

Fit corpus: OpenWebText via the Hub's auto-converted parquet branch
(script-free, works with datasets 3.x). Falls back to FineWeb sample-10BT
if that branch is unavailable. The chosen source is printed -- record it,
it goes in the paper's experimental setup.

Eval corpora: held-out web text, WikiText-103 test (parquet-native),
WikiText at longer sequence length, optional code.
"""
import torch


def tokenize_blocks(texts, tokenizer, seq_len, min_len=64):
    blocks = []
    for t in texts:
        ids = tokenizer(t, truncation=True, max_length=seq_len,
                        return_tensors=None)["input_ids"]
        if len(ids) >= min_len:
            blocks.append(ids[:seq_len])
    return blocks


def _load_webtext(n_docs, seed):
    """Streaming, script-free web text. Returns (source_name, list[str])."""
    from datasets import load_dataset
    sources = [
        ("OpenWebText (parquet branch)", lambda: load_dataset(
            "parquet",
            data_files="hf://datasets/Skylion007/openwebtext@refs/convert/"
                       "parquet/plain_text/**/*.parquet",
            split="train", streaming=True)),
        ("FineWeb sample-10BT", lambda: load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT",
            split="train", streaming=True)),
    ]
    last_err = None
    for name, loader in sources:
        try:
            ds = loader().shuffle(seed=seed, buffer_size=10_000)
            texts = [ex["text"] for _, ex in zip(range(n_docs), iter(ds))]
            if len(texts) < n_docs // 2:
                raise RuntimeError(f"only {len(texts)} docs")
            print(f"[data] fit/eval web corpus: {name} ({len(texts)} docs)")
            return name, texts
        except Exception as e:
            print(f"[data] {name} failed ({type(e).__name__}: {e}); "
                  f"trying next source")
            last_err = e
    raise RuntimeError(f"no web-text source available: {last_err}")


def load_corpora(tokenizer, cfg):
    """Returns fit blocks, val blocks, {domain: eval blocks}."""
    from datasets import load_dataset
    n = cfg.n_fit_docs + 200 + cfg.n_ppl_docs
    _, texts = _load_webtext(n, cfg.seed)
    fit = tokenize_blocks(texts[:cfg.n_fit_docs], tokenizer, cfg.seq_len)
    val = tokenize_blocks(texts[cfg.n_fit_docs:cfg.n_fit_docs + 200],
                          tokenizer, cfg.seq_len)
    owt_eval = tokenize_blocks(texts[cfg.n_fit_docs + 200:], tokenizer,
                               cfg.seq_len)[:cfg.n_ppl_docs]

    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="test")
    wt_texts = [t for t in wt["text"] if len(t) > 200][:cfg.n_ppl_docs]
    wt_eval = tokenize_blocks(wt_texts, tokenizer, cfg.seq_len)
    wt_long = tokenize_blocks(wt_texts, tokenizer, cfg.long_seq_len,
                              min_len=cfg.seq_len)

    evals = {"openwebtext": owt_eval, "wikitext103": wt_eval,
             "wikitext103_long": wt_long}
    try:  # optional code domain; skip silently if unavailable
        code = load_dataset("codeparrot/codeparrot-clean-valid",
                            split="train", streaming=True)
        code_texts = [ex["content"] for _, ex in
                      zip(range(cfg.n_ppl_docs), iter(code))]
        evals["code"] = tokenize_blocks(code_texts, tokenizer, cfg.seq_len)
    except Exception as e:
        print(f"[data] code domain skipped: {type(e).__name__}: {e}")
    return fit, val, evals


def batches(blocks, bs, pad_id, device):
    for i in range(0, len(blocks), bs):
        chunk = blocks[i:i + bs]
        ml = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), ml), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), ml), dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.as_tensor(x)
            att[j, :len(x)] = 1
        yield ids.to(device), att.to(device)
