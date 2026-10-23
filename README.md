# Mini-LLM

Small assistants (~40M and ~150M parameters on the 32k tokenizer lineage, ~34M and
~70M on the earlier 16k lineage) trained **end-to-end** with PyTorch — custom BPE
tokenizer, transformer decoder architecture, pretraining and SFT, no pretrained backbone.

Training runs on NVIDIA GPUs (CUDA).

> Training scale translates directly into output quality: base val loss drops from
> 3.618 (34M, ~0.2B tokens) to 3.030 (70M, ~1.4B tokens) and to 2.941 (150M, ~4.5B
> tokens) — larger models are markedly more fluent and better formatted. Conformance
> and factual accuracy are tracked continuously with
> [Mini-Eval](https://github.com/Mini-llm-lab/Mini-Eval).

**Full documentation: [../Documentation/Mini-LLM](../Documentation/Mini-LLM/README.md)**

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```
git clone https://github.com/Mini-llm-lab/Mini-LLM
cd Mini-LLM
uv sync

uv run prep          # tokenizer + data
uv run pretrain 300  # short smoke run
uv run sft 1
uv run chat
```

To speed up Hugging Face downloads, create a free token and export it before running
`prep` or `sft`: `export HF_TOKEN=...`.

Full training runs go through the cloud GPU workflow — see
[Cloud GPU](../Documentation/Mini-LLM/cloud-gpu.md). Everything else — architecture,
checkpoints, data sources, Hugging Face export, design notes — lives in the
[documentation](../Documentation/Mini-LLM/README.md).
