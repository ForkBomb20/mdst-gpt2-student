#!/usr/bin/env python3
"""
Minimal GPT-2 CLI for your fine-tuned checkpoint.

Usage:
  python3 cli.py

Notes:
  - Expects ./model.pt in the same folder as this script (and tokenizer files if you saved them).
  - Falls back to the base GPT-2 tokenizer if local tokenizer files aren't present.
  - Works on CUDA, Apple Silicon (MPS), or CPU automatically.
  - Commands: /reset clears history, /quit exits.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
# Use CONFIG_MAPPING to rebuild the right Config class from a dict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Quiet chatter + allow MPS CPU fallback for missing kernels
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

HERE = Path(__file__).resolve().parent
CKPT_PATH = HERE / "model.pt"
BASE_ID = "openai-community/gpt2"  # used for tokenizer (and model fallback)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path: Path):
    if not path.exists():
        sys.exit(f"[error] checkpoint not found: {path}")
    # Prefer weights_only=True (PyTorch ≥ 2.4), fall back if older
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        sys.exit("[error] bad checkpoint format (expected dict with key 'model')")
    return ckpt["model"], ckpt.get("config")


def build_model_from_config_dict(config_dict: dict | None):
    """Return an uninitialized model given a saved config dict (or a BASE_ID fallback)."""
    if not config_dict:
        return AutoModelForCausalLM.from_pretrained(BASE_ID)
    # Determine correct Config subclass from 'model_type'
    model_type = config_dict.get("model_type", "gpt2")
    if model_type not in CONFIG_MAPPING:
        # Unknown type; safest fallback is base GPT-2 arch
        return AutoModelForCausalLM.from_pretrained(BASE_ID)
    config_cls = CONFIG_MAPPING[model_type]
    config = config_cls.from_dict(config_dict)
    return AutoModelForCausalLM.from_config(config)


def load_model_and_tokenizer(device: str):
    state_dict, config_dict = load_checkpoint(CKPT_PATH)

    # Tokenizer: prefer local saved files, else base GPT-2
    if (HERE / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(HERE))
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_ID)

    # Ensure pad token exists for generation
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model, then load weights
    model = build_model_from_config_dict(config_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 5) {missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 5) {unexpected[:5]}", file=sys.stderr)

    model.to(device)
    if device == "mps":
        model = model.float()  # MPS prefers fp32

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


def truncate_context(tokenizer, text: str, max_tokens: int = 900) -> str:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.numel() <= max_tokens:
        return text
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)


def generate_reply(model, tokenizer, device: str, prompt: str,
                   max_new_tokens=160, temperature=0.8, top_p=0.95,
                   top_k=0, repetition_penalty=1.0, stop_at_eos=True):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id if stop_at_eos and tokenizer.eos_token_id is not None else None,
    )
    if top_k and top_k > 0:
        gen_kwargs["top_k"] = top_k

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):]


def main():
    device = pick_device()
    print(f"[info] device: {device}", file=sys.stderr)
    model, tokenizer = load_model_and_tokenizer(device)

    history = ""
    sys.stderr.write("[info] Chat ready. Type /reset to clear or /quit to exit.\n")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"/quit", "/exit"}:
            break
        if user.lower() == "/reset":
            history = ""
            print("[history cleared]")
            continue

        history = truncate_context(tokenizer, history, max_tokens=900)
        prompt = f"{history}\nUser: {user}\nAssistant:"

        reply = generate_reply(model, tokenizer, device, prompt)
        print(reply.strip())

        history = f"{prompt}{reply}"


if __name__ == "__main__":
    main()
