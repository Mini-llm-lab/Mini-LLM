import argparse
import shutil
from pathlib import Path

import torch

from mini_llm.prompt import DEFAULT_SYSTEM_PROMPT, format_prompt
from mini_llm.config import MODELS_DIR
from mini_llm.configuration_mini_llm import MiniLlmConfig
from mini_llm.modeling_mini_llm import MiniLlmForCausalLM
from mini_llm.prompt import format_prompt

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ 'User: ' + m['content'].strip() + '\\n' if m['role'] == 'user' "
    "else 'Assistant: ' + m['content'].strip() + '\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
)


def v2_chat_template():
    """The digit32k (v2) format: System: <default+date>\\nUser: \\\\nAssistant:"""
    header = (
        "System: Today is {{ strftime_now('%b %d, %Y') }}.\n"
        + DEFAULT_SYSTEM_PROMPT.strip()
        + "\n"
    )
    loop = (
        "{% for m in messages %}"
        "{{ ('System: ' + m['content'].strip() + '\\n') if m['role'] == 'system' "
        "else ('User: ' + m['content'].strip() + '\\n') if m['role'] == 'user' "
        "else ('Assistant: ' + m['content'].strip() + '\\n') }}"
        "{% endfor %}"
    )
    gen = (
        "{% if add_generation_prompt %}"
        "{% set has_system = messages and messages[0]['role'] == 'system' %}"
        "{% if not has_system %}" + header + "{% endif %}"
        "Assistant:{% endif %}"
    )
    return loop + gen


def build_tokenizer(tok_path, chat_template=True, v2=False):
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_path),
        eos_token="<eos>",
        pad_token="<eos>",
        unk_token=None,
        bos_token=None,
    )
    if chat_template:
        tok.chat_template = v2_chat_template() if v2 else CHAT_TEMPLATE
    return tok


def export(checkpoint=None, out=None, verify=True, base=False):
    ckpt = resolve_checkpoint(checkpoint)
    print(f"checkpoint: {ckpt.name}")
    dev = "cpu"
    native, cfg = load_model(str(ckpt), dev)
    native.eval()
    tok_path = tokenizer_file(ckpt)
    raw_tok = load_tokenizer(str(ckpt))
    v2 = getattr(cfg, "lineage", "") == "digit32k"
    verify_prompt = (
        format_prompt("Hello world", DEFAULT_SYSTEM_PROMPT) if v2 else format_prompt("Hello world")
    )
    eos = raw_tok.token_to_id("<eos>")

    hf_config = MiniLlmConfig.from_cfg(cfg)
    hf_config.auto_map = {
        "AutoConfig": "configuration_mini_llm.MiniLlmConfig",
        "AutoModelForCausalLM": "modeling_mini_llm.MiniLlmForCausalLM",
    }
    hf = MiniLlmForCausalLM(hf_config)
    missing, unexpected = hf.load_state_dict(native.state_dict(), strict=False)
    hf.tie_weights()
    if missing or unexpected:
        print(f"load_state_dict: missing={missing} unexpected={unexpected}")
        allowed_missing = {"head.weight", "cos", "sin"}
        leftover_missing = [k for k in missing if k not in allowed_missing]
        if leftover_missing or unexpected:
            raise SystemExit(f"weight mismatch: missing={missing} unexpected={unexpected}")

    if out is None:
        out = MODELS_DIR.parent / "exports" / f"{ckpt.stem}-hf"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if verify:
        with torch.no_grad():
            sample = torch.tensor(
                [raw_tok.encode(verify_prompt).ids[:32] or [eos]]
            )
            ref, _ = native(sample)
            got = hf(input_ids=sample).logits
            max_diff = (ref.float() - got.float()).abs().max().item()
            print(f"logits max abs diff: {max_diff:.3e}")
            if max_diff > 1e-5:
                raise SystemExit(f"export verification failed (diff {max_diff})")

    hf.save_pretrained(str(out), safe_serialization=True)

    tok = build_tokenizer(tok_path, chat_template=not base, v2=v2)
    tok.save_pretrained(str(out))

    if not base:
        from transformers import GenerationConfig

        rep_penalty, max_new = (1.1, 512) if v2 else (1.15, 256)
        gen = GenerationConfig(
            do_sample=True,
            temperature=0.5,
            top_k=40,
            repetition_penalty=rep_penalty,
            max_new_tokens=max_new,
            eos_token_id=eos,
            pad_token_id=eos,
        )
        gen.save_pretrained(str(out))

    src_dir = Path(__file__).resolve().parent
    for name in ("configuration_mini_llm.py", "modeling_mini_llm.py"):
        shutil.copy(src_dir / name, out / name)

    if verify:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        rt_tok = AutoTokenizer.from_pretrained(str(out), trust_remote_code=True)
        rt = AutoModelForCausalLM.from_pretrained(
            str(out), trust_remote_code=True, dtype=torch.float32
        ).eval()
        with torch.no_grad():
            sample = torch.tensor([rt_tok.encode(verify_prompt)[:32] or [eos]])
            a = hf(input_ids=sample).logits.float()
            b = rt(input_ids=sample).logits.float()
            diff = (a - b).abs().max().item()
            print(f"reload max abs diff: {diff:.3e}")
            if diff > 1e-5:
                raise SystemExit("reload verification failed")

    files = sorted(p.name for p in out.iterdir())
    print(f"done -> {out}\n  {', '.join(files)}")
    print("upload: huggingface-cli upload <org>/<repo> " + str(out) + " .")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument(
        "--base",
        action="store_true",
        help="base model: no chat template and no sampling defaults (an assistant is not attached)",
    )
    args = ap.parse_args()
    export(args.checkpoint, args.out, verify=not args.no_verify, base=args.base)


if __name__ == "__main__":
    main()
