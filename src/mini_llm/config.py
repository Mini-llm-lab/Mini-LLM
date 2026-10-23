from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


LINEAGE_DIRS = {"": "{vocab//1024}k", "digit32k": "digit32k"}


def lineage_dir(lineage):
    key = lineage or ""
    if key not in LINEAGE_DIRS:
        return None
    return LINEAGE_DIRS[key]


def tokenizer_dir(vocab, lineage=""):
    spec = LINEAGE_DIRS[lineage or ""]
    return MODELS_DIR / spec.replace("{vocab//1024}", str(vocab // 1024))


def tokenizer_path(vocab, lineage=""):
    return tokenizer_dir(vocab, lineage) / "tok.json"


def model_dir(vocab, tag, lineage=""):
    return tokenizer_dir(vocab, lineage) / tag


@dataclass
class Cfg:
    vocab: int = 16384
    dim: int = 512
    layers: int = 8
    heads: int = 8
    ffn: int = 1408
    ctx: int = 512
    lineage: str = ""


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(name):
    if name is None or name == "auto":
        return get_device()
    if name not in ("cuda", "cpu"):
        raise SystemExit(f"Unknown device {name!r}, expected one of: auto, cuda, cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested (--device cuda) but no GPU is available")
    return name


def size_millions(n_params):
    return round(n_params / 1e7) * 10


PRESETS = {
    "34M": Cfg(),
    "40M": Cfg(vocab=32768, ctx=1024),
    "70M": Cfg(dim=640, layers=12, heads=10, ffn=1728),
    "150M": Cfg(dim=768, layers=18, heads=12, ffn=2048, vocab=32768, ctx=2048, lineage="digit32k"),
}


def get_preset(name):
    try:
        return PRESETS[name]
    except KeyError:
        raise SystemExit(f"Unknown size {name!r}, expected one of: {', '.join(sorted(PRESETS))}")