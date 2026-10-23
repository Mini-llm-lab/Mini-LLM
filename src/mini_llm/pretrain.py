import argparse
import math
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from mini_llm.checkpoints import (
    cfg_from,
    load_model,
    rank,
    read_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
    sha256_file,
)
from mini_llm.config import (
    DATA_DIR,
    get_preset,
    model_dir,
    resolve_device,
    size_millions,
    tokenizer_path,
)
from mini_llm.model import LLM

DATA_SEED = 1234
EVAL_WINDOWS = 1024

HPARAMS = {
    "34M": {"lr_max": 5e-4, "lr_cont": 3e-4, "lr_min": 5e-5, "warmup": 200},
    "40M": {"lr_max": 5e-4, "lr_cont": 3e-4, "lr_min": 5e-5, "warmup": 1000},
    "70M": {"lr_max": 6e-4, "lr_cont": 3e-4, "lr_min": 6e-5, "warmup": 500},
    "150M": {"lr_max": 5e-4, "lr_cont": 3e-4, "lr_min": 5e-5, "warmup": 1000},
}

BIG_BATCH_SIZES = ("40M", "150M")
KD_CHUNK = 4096


def kd_loss(logits_s, logits_t, temp, chunk=KD_CHUNK):
    ls = logits_s.reshape(-1, logits_s.size(-1))
    lt = logits_t.reshape(-1, logits_t.size(-1))
    total = ls.new_zeros((), dtype=torch.float32)
    for i in range(0, ls.size(0), chunk):
        s = ls[i : i + chunk].float() / temp
        t = lt[i : i + chunk].float() / temp
        total = total + F.kl_div(
            F.log_softmax(s, dim=-1),
            F.log_softmax(t, dim=-1),
            reduction="sum",
            log_target=True,
        )
    return total * temp**2 / ls.size(0)


def validate_teacher(tcfg, cfg):
    if tcfg.vocab != cfg.vocab:
        sys.exit(
            f"The teacher predicts {tcfg.vocab} tokens but the {cfg.vocab} lineage needs {cfg.vocab}."
        )
    if getattr(tcfg, "lineage", "") != getattr(cfg, "lineage", ""):
        sys.exit(
            f"The teacher was trained on the {getattr(tcfg, 'lineage', '')!r} tokenizer lineage "
            f"but the student needs {getattr(cfg, 'lineage', '')!r}."
        )
    if tcfg.ctx < cfg.ctx:
        sys.exit(f"The teacher context is {tcfg.ctx} but the student needs {cfg.ctx}.")


def default_batch(dev, size="34M", ctx=512):
    if dev == "cuda":
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            batch = 64 if total >= 40 * 1024**3 else 32
        except Exception:
            batch = 32
    else:
        batch = 32
    batch = min(batch, max(1, 65536 // ctx))
    target = 131072 if size in BIG_BATCH_SIZES else 65536
    return batch, max(1, target // (batch * ctx))


def check_data_matches(meta_path, cfg, tok_path=None, train_len=None, val_len=None):
    import json

    if tok_path is None:
        tok_path = tokenizer_path(cfg.vocab, getattr(cfg, "lineage", ""))
    if not meta_path.exists():
        print("warning: no data/meta.json (data predates vocab tracking)")
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("vocab") != cfg.vocab:
        raise SystemExit(
            f"data/meta.json has vocab={meta.get('vocab')} but the {cfg.vocab} preset needs vocab={cfg.vocab}. "
            "Regenerate data/ with prep, or pick the matching --size."
        )
    stored_lineage = meta.get("lineage", "")
    if stored_lineage != getattr(cfg, "lineage", ""):
        raise SystemExit(
            f"data belongs to the lineage {stored_lineage!r} but the preset needs {getattr(cfg, 'lineage', '')!r}. "
            "Run prep with --lineage to regenerate data/ for this tokenizer lineage."
        )
    if not tok_path.exists():
        raise SystemExit(f"Tokenizer missing: {tok_path}")
    actual = sha256_file(tok_path)
    if meta.get("tokenizer_sha256") != actual:
        raise SystemExit(
            f"data/meta.json tokenizer {meta.get('tokenizer_sha256')} does not match {tok_path} ({actual}). "
            "Regenerate data/ with prep, or restore the matching tok.json."
        )
    for name, actual_len in (("train", train_len), ("val", val_len)):
        expected = meta.get(f"{name}_tokens")
        if expected is not None and actual_len is not None and expected != actual_len:
            raise SystemExit(
                f"data/{name}.bin has {actual_len} tokens but data/meta.json records {expected}. "
                "The .bin files do not belong to this meta.json: regenerate data/ with prep."
            )


def check_ids_fit(train, val, cfg):
    if not len(train) or not len(val):
        return
    top = max(int(train.max()), int(val.max()))
    if top >= cfg.vocab:
        raise SystemExit(
            f"data contains token id {top} but the preset needs vocab={cfg.vocab}. "
            "Regenerate data/ with prep, or pick the matching --size."
        )


def pretrain(steps=5000, resume=False, init=None, size="34M", batch=None, accum=None, lr=None,
             save_every=None, milestone_every=None, device="auto", teacher=None,
             kd_weight=0.5, kd_temp=2.0):
    cfg = get_preset(size)
    hp = HPARAMS[size]
    lr_max, lr_cont, lr_min, warmup = hp["lr_max"], hp["lr_cont"], hp["lr_min"], hp["warmup"]
    dev = resolve_device(device)
    if dev == "cuda":
        torch.set_float32_matmul_precision("high")
    if teacher is not None:
        if not 0.0 < kd_weight <= 1.0:
            sys.exit("--kd-weight must be in (0, 1]")
        if kd_temp <= 0:
            sys.exit("--kd-temp must be > 0")
    if save_every is None:
        save_every = 1000
    if milestone_every is None:
        milestone_every = 10000 if size == "150M" else 5000
    db, da = default_batch(dev, size, cfg.ctx)
    batch = db if batch is None else batch
    accum = da if accum is None else accum

    data = {
        s: np.memmap(str(DATA_DIR / f"{s}.bin"), dtype=np.uint16, mode="r")
        for s in ("train", "val")
    }
    train, val = data["train"], data["val"]
    check_data_matches(DATA_DIR / "meta.json", cfg, train_len=len(train), val_len=len(val))
    check_ids_fit(train, val, cfg)
    n_win = (len(train) - 1) // cfg.ctx
    if n_win <= 0:
        sys.exit(f"train.bin too small ({len(train)} tokens) for ctx {cfg.ctx}")

    model = LLM(cfg).to(dev)
    teacher_model, kd = None, None
    if teacher is not None:
        teacher_path = resolve_checkpoint(teacher)
        if not teacher_path.exists():
            sys.exit(f"Teacher checkpoint not found: {teacher_path}")
        teacher_model, teacher_cfg = load_model(teacher_path, dev)
        validate_teacher(teacher_cfg, cfg)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        kd = {"teacher": teacher_path.name, "weight": kd_weight, "temp": kd_temp}
        print(f"teacher: {teacher_path.name} | kd weight {kd_weight:g} | temp {kd_temp:g}")
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"{size_millions(n_params)}M"
    out_dir = model_dir(cfg.vocab, tag, getattr(cfg, "lineage", ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mini-llm-base-{tag}-{steps}steps"
    weights_path = out_dir / f"{stem}.pt"
    state_path = out_dir / f"{stem}.state.pt"
    print(f"device: {dev}")
    print(f"{n_params / 1e6:.1f}M parameters -> {weights_path.name}")

    decay, nodecay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=lr_max, betas=(0.9, 0.95), fused=(dev == "cuda"),
    )

    start, phase_start = 0, 0
    if resume and init:
        sys.exit("Use either --resume or --init, not both")
    if resume:
        if not state_path.exists():
            sys.exit(f"Nothing to resume: {state_path.name} not found")
        state = read_checkpoint(state_path, dev)
        if state["total_steps"] != steps:
            sys.exit(f"The saved run targets {state['total_steps']} steps, not {steps}")
        if state.get("kd") != kd:
            sys.exit(
                "The saved run used different distillation settings (--teacher/--kd-weight/--kd-temp)"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start = state["step"]
        phase_start = state.get("phase_start", 0)
        print(f"resuming from step {start}")
    else:
        if weights_path.exists() or state_path.exists():
            sys.exit(f"{stem} already exists. Use --resume to continue it, or delete its files to start over.")
        if init:
            init_path = resolve_checkpoint(init)
            ck = read_checkpoint(init_path, dev)
            if cfg_from(ck["cfg"]) != cfg:
                sys.exit(f"{init_path.name} was trained with a different configuration")
            phase_start = rank(init_path)[0]
            if phase_start >= steps:
                sys.exit(f"{init_path.name} already has {phase_start} steps, ask for more than that")
            model.load_state_dict(ck["model"])
            start = phase_start
            print(f"continuing from {init_path.name} (step {phase_start})")

    lr_peak = lr_max if phase_start == 0 else lr_cont
    if lr is not None and phase_start == 0:
        lr_peak = lr
    warmup = min(warmup, max(1, (steps - phase_start) // 20))

    toks_step = batch * accum * cfg.ctx
    total_toks = toks_step * steps
    print(f"batch {batch}x{accum} | {toks_step} tokens/step")
    print(f"steps {steps} | total {total_toks / 1e9:.2f}B tokens | {total_toks / n_params:.1f} tokens/param")
    print(f"train {len(train) / 1e6:.1f}M tokens | {total_toks / max(1, len(train)):.1f} passes")
    if total_toks > 1.1 * len(train):
        print(f"warning: total tokens exceed 1.1x train.bin size ({len(train) / 1e6:.1f}M)")
    print(f"lr peak {lr_peak:g} | warmup {warmup}")

    n_seen = start * batch * accum
    epoch, pos = divmod(n_seen, n_win)
    perm = np.random.default_rng(DATA_SEED + epoch).permutation(n_win)

    def get_train_batch():
        nonlocal epoch, pos, perm
        starts = []
        for _ in range(batch):
            if pos >= n_win:
                epoch += 1
                perm = np.random.default_rng(DATA_SEED + epoch).permutation(n_win)
                pos = 0
            starts.append(int(perm[pos]) * cfg.ctx)
            pos += 1
        x = np.stack([train[s : s + cfg.ctx] for s in starts]).astype(np.int64)
        y = np.stack([train[s + 1 : s + 1 + cfg.ctx] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    n_vwin = (len(val) - 1) // cfg.ctx
    if n_vwin <= 0:
        eval_starts = np.array([0], dtype=np.int64)
    elif n_vwin <= EVAL_WINDOWS:
        eval_starts = np.arange(n_vwin, dtype=np.int64) * cfg.ctx
    else:
        eval_starts = np.sort(np.random.default_rng(0).choice(n_vwin, size=EVAL_WINDOWS, replace=False)).astype(np.int64) * cfg.ctx

    cmodel = torch.compile(model) if dev == "cuda" else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else nullcontext()

    def lr_at(s):
        i = s - phase_start
        if i < warmup:
            return lr_peak * (i + 1) / warmup
        p = (i - warmup) / max(1, steps - phase_start - warmup)
        return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * p))

    @torch.no_grad()
    def evaluate():
        model.eval()
        ls = []
        for k in range(0, len(eval_starts), batch):
            ss = eval_starts[k : k + batch]
            x = np.stack([val[s : s + cfg.ctx] for s in ss]).astype(np.int64)
            y = np.stack([val[s + 1 : s + 1 + cfg.ctx] for s in ss]).astype(np.int64)
            x, y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
            with amp:
                ls.append(cmodel(x, y)[1].item())
        model.train()
        return sum(ls) / len(ls)

    val_loss = None
    t0 = time.time()
    t_last, done_last = t0, start
    for step in range(start, steps):
        cur_lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        step_loss = torch.zeros((), device=dev)
        step_ce = torch.zeros((), device=dev)
        step_kd = torch.zeros((), device=dev)
        for _ in range(accum):
            x, y = get_train_batch()
            with amp:
                logits, ce = cmodel(x, y)
            if teacher_model is not None:
                with torch.no_grad(), amp:
                    tlogits = teacher_model(x)[0]
                batch_kd = kd_loss(logits, tlogits, kd_temp)
                loss = ((1 - kd_weight) * ce + kd_weight * batch_kd) / accum
                step_kd += batch_kd.detach() / accum
            else:
                loss = ce / accum
            loss.backward()
            step_loss += loss.detach()
            step_ce += ce.detach() / accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss_val = step_loss.item()
        gnorm_val = gnorm.item()
        if not math.isfinite(loss_val) or not math.isfinite(gnorm_val):
            print(f"!! non-finite value at step {step} (loss={loss_val}, gnorm={gnorm_val}). Stopping.")
            if state_path.exists():
                print(f"Last saved state: {state_path.name} (continue with --resume).")
            return
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step == start:
            t_last, done_last = time.time(), step + 1
        if step % 50 == 0:
            now = time.time()
            head = f"step {step:5d}/{steps} | loss {loss_val:.3f} | gnorm {gnorm_val:.2f} | lr {cur_lr:.2e}"
            if teacher_model is not None:
                head = (
                    f"step {step:5d}/{steps} | loss {loss_val:.3f} | ce {step_ce.item():.3f} "
                    f"| kd {step_kd.item():.3f} | gnorm {gnorm_val:.2f} | lr {cur_lr:.2e}"
                )
            if step == start:
                print(f"{head} | warm-up step (compile), throughput from the next log | {now - t0:.0f}s", flush=True)
            else:
                rate = (step + 1 - done_last) * toks_step / max(1e-6, now - t_last)
                eta_h = (steps - step - 1) * toks_step / max(1.0, rate) / 3600
                print(f"{head} | {rate / 1000:.1f}k tok/s | {now - t0:.0f}s | ETA {eta_h:.1f}h", flush=True)
                t_last, done_last = now, step + 1
        done = step + 1
        if done % save_every == 0 or done == steps:
            val_loss = evaluate()
            print(f"  >> val loss {val_loss:.3f}", flush=True)
            save_checkpoint(
                state_path, model, cfg,
                kind="base", step=done, total_steps=steps, val_loss=val_loss,
                phase_start=phase_start, optimizer=opt.state_dict(), kd=kd,
            )
        if done % milestone_every == 0 and done != steps:
            mile_dir = out_dir / "milestones"
            mile_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                mile_dir / f"{stem}.at{done}.pt", model, cfg,
                kind="base", step=done, total_steps=steps, val_loss=val_loss,
                phase_start=phase_start, kd=kd,
            )

    save_checkpoint(
        weights_path, model, cfg,
        kind="base", step=steps, total_steps=steps, val_loss=val_loss, phase_start=phase_start,
        kd=kd,
    )
    state_path.unlink(missing_ok=True)
    print(f"done -> {weights_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", type=int, nargs="?", default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init")
    ap.add_argument("--size", default="34M")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--milestone-every", type=int, default=None)
    ap.add_argument("--teacher")
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    args = ap.parse_args()
    pretrain(args.steps, args.resume, args.init, args.size, args.batch, args.accum, args.lr,
             args.save_every, args.milestone_every, args.device, args.teacher,
             args.kd_weight, args.kd_temp)


if __name__ == "__main__":
    main()
