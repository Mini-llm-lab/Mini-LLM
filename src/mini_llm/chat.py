import sys
from datetime import date

import torch

from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.config import get_device
from mini_llm.prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt, format_history

TEMPERATURE = 0.5
TOP_K = 40
REPETITION_PENALTY = 1.1
MAX_NEW_TOKENS = 512
HISTORY_TURNS = 2


def default_system(lineage):
    if lineage != "digit32k":
        return None
    return build_system_prompt(DEFAULT_SYSTEM_PROMPT, date=date.today().isoformat())


def reply(model, tok, eos, dev, question, history=(), ctx=None, system=None):
    """Answer one question, seeing up to HISTORY_TURNS answered turns before it."""
    if ctx is None:
        ctx = getattr(getattr(model, "c", None), "ctx", 512)
    history = list(history)[-HISTORY_TURNS:]
    question = question.strip()
    budget = ctx - 1

    def prefix_len(h, q):
        return len(tok.encode(format_history(system, h + [(q, None)])).ids)

    while prefix_len(history, question) > budget:
        if history:
            history = history[1:]
        elif len(question) > 400:
            question = question[len(question) // 10 :]
        else:
            break

    ids = tok.encode(format_history(system, history + [(question, None)])).ids[:ctx]
    n_new = max(1, min(MAX_NEW_TOKENS, ctx - len(ids)))
    x = torch.tensor([ids], device=dev)
    out = model.generate(
        x, n=n_new, temp=TEMPERATURE, top_k=TOP_K, eos=eos, rep_penalty=REPETITION_PENALTY
    )
    return tok.decode(out[0, x.size(1):].tolist()).strip()


def chat(path=None):
    dev = get_device()
    path = resolve_checkpoint(path, pattern="mini-llm-assistant-*.pt")
    print(f"[{path.name}] Ctrl+C to quit, /quit to exit (keeps the last {HISTORY_TURNS} answers)")
    model, cfg = load_model(path, dev)
    model.eval()
    ctx = cfg.ctx
    tok = load_tokenizer(path)
    eos = tok.token_to_id("<eos>")
    system = default_system(getattr(cfg, "lineage", ""))
    history = []
    while True:
        try:
            question = input("\nYou > ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        cmd = question.strip().lower()
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/help":
            print(f"Commands: /quit exit, Ctrl+C quit. Keeps the last {HISTORY_TURNS} answers in the prompt.")
            continue
        if not question.strip():
            continue
        answer = reply(model, tok, eos, dev, question, history, ctx, system)
        history.append((question.strip(), answer))
        print("AI  >", answer)


def main():
    chat(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
