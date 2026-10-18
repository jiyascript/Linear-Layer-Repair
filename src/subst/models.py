"""Model loading, layer-stack access, and hidden-state extraction.

A "stack" is a named list of transformer layers plus a function that maps a
batch to the (L+1) hidden states of that stack. Decoders/encoders have one
stack ("main"); encoder-decoder models have two ("enc", "dec").
"""
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForMaskedLM, AutoModelForSeq2SeqLM)
from .config import ModelSpec


def load(spec: ModelSpec, device, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(spec.hf_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cls = {"decoder": AutoModelForCausalLM,
           "encoder": AutoModelForMaskedLM,
           "encdec": AutoModelForSeq2SeqLM}[spec.family]
    model = cls.from_pretrained(spec.hf_name, torch_dtype=dtype).to(device).eval()
    if spec.family == "encdec":
        # teacher-forcing via labels= requires this; real ckpts set it, be safe
        if getattr(model.config, "decoder_start_token_id", None) is None:
            model.config.decoder_start_token_id = tok.pad_token_id
    # encoders in fp16 can overflow on some checkpoints; fp32 is cheap at <300M
    return tok, model


def from_config_random(spec: ModelSpec, hf_name, device, dtype=torch.float16):
    """Randomly initialized copy with identical architecture (control)."""
    from transformers import AutoConfig
    cls = {"decoder": AutoModelForCausalLM,
           "encoder": AutoModelForMaskedLM,
           "encdec": AutoModelForSeq2SeqLM}[spec.family]
    cfg = AutoConfig.from_pretrained(hf_name)
    m = cls.from_config(cfg).to(dtype).to(device).eval()
    if spec.family == "encdec" and getattr(
            m.config, "decoder_start_token_id", None) is None:
        m.config.decoder_start_token_id = getattr(m.config, "pad_token_id", 0)
    return m


def layer_lists(model, family):
    """name -> nn.ModuleList of transformer layers."""
    if family == "decoder":
        base = model.model if hasattr(model, "model") else model
        if hasattr(base, "layers"):                       # llama/qwen
            return {"main": base.layers}
        if hasattr(base, "gpt_neox"):                     # pythia via top-level
            return {"main": base.gpt_neox.layers}
        if hasattr(model, "gpt_neox"):
            return {"main": model.gpt_neox.layers}
        raise ValueError("unknown decoder layout")
    if family == "encoder":
        base = getattr(model, model.base_model_prefix)    # .bert / .roberta
        return {"main": base.encoder.layer}
    if family == "encdec":
        if hasattr(model, "model"):                       # bart
            return {"enc": model.model.encoder.layers,
                    "dec": model.model.decoder.layers}
        return {"enc": model.encoder.block,               # t5
                "dec": model.decoder.block}
    raise ValueError(family)


@torch.no_grad()
def hidden_stacks(model, family, ids, att, d):
    """Return {stack_name: [h_0..h_L] each [n_valid_tokens, d]} for a batch."""
    mask = att.bool().reshape(-1)
    flat = lambda hs: [h.reshape(-1, d)[mask] for h in hs]
    if family in ("decoder",):
        out = model(input_ids=ids, attention_mask=att,
                    output_hidden_states=True, use_cache=False)
        return {"main": flat(out.hidden_states)}
    if family == "encoder":
        out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        return {"main": flat(out.hidden_states)}
    # encdec: teacher-force the decoder on the same text so decoder hidden
    # states exist; decoder tokens are all "valid" (labels != pad handled ok).
    out = model(input_ids=ids, attention_mask=att, labels=ids,
                output_hidden_states=True, use_cache=False)
    dec = [h.reshape(-1, d)[mask] for h in out.decoder_hidden_states]
    return {"enc": flat(out.encoder_hidden_states), "dec": dec}
