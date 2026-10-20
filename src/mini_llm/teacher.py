import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from mini_llm.checkpoints import save_checkpoint, sha256_file
from mini_llm.config import get_preset, model_dir
from mini_llm.model import LLM

REPO = "GautierP/MiniLLM-150M"
REVISION = "6c218a122ce1a410bbe3cf5afe4b8504ae6e8684"
WEIGHTS_SHA256 = "4dd1bf4a84a52272456a89d2bf7555f595bc1638bbaf21cd33a518fe0d3755ac"
SIZE = "150M"
STEPS = 34332
VAL_LOSS = 2.9408972710371017


def to_native_state_dict(sd):
    sd = dict(sd)
    sd.pop("cos", None)
    sd.pop("sin", None)
    sd["head.weight"] = sd["emb.weight"]
    return sd


def verify(native, local, cfg):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        str(local / "base"), trust_remote_code=True, dtype=torch.float32
    ).eval()
    ids = torch.randint(0, cfg.vocab, (2, 64))
    with torch.no_grad():
        ref = native(ids)[0].float()
        got = hf(input_ids=ids).logits.float()
    diff = (ref - got).abs().max().item()
    print(f"logits max abs diff: {diff:.3e}")
    if diff > 1e-5:
        raise SystemExit("teacher verification failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    cfg = get_preset(SIZE)
    local = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["base/*"]))
    weights = local / "base" / "model.safetensors"
    actual = sha256_file(weights)
    if actual != WEIGHTS_SHA256:
        raise SystemExit(f"unexpected weights sha256: {actual}")

    model = LLM(cfg)
    model.load_state_dict(to_native_state_dict(load_file(weights)), strict=True)
    model.eval()
    if not args.no_verify:
        verify(model, local, cfg)

    out = Path(args.out) if args.out else model_dir(cfg.vocab, SIZE) / "teacher-150M-base.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out, model, cfg,
        kind="teacher", step=STEPS, total_steps=STEPS, val_loss=VAL_LOSS,
        teacher_repo=REPO, teacher_revision=REVISION,
    )
    print(f"teacher -> {out}")


if __name__ == "__main__":
    main()
