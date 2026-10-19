#!/usr/bin/env bash
# Preflight check before `cloud/run.sh prep` (40M default).
# Verifies the exact guards that would make prep or pretrain refuse to run:
#   1. data/meta.json vocab vs SIZE (run.sh)
#   2. models/<lineage>/tok.json vocab vs --vocab (prep.py)
#   3. data/*.bin length vs the token counts in meta.json (pretrain.py)
# Read-only: never modifies data/ or models/.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${SIZE:-40M}"
case "$SIZE" in
  40M|150M) EXPECT_VOCAB=32768 ;;
  *) EXPECT_VOCAB=16384 ;;
esac
TOK="models/$((EXPECT_VOCAB / 1024))k/tok.json"

fail=0
say() { echo "$1"; }
warn() { echo "!! $1" >&2; fail=1; }

say "SIZE=$SIZE EXPECT_VOCAB=$EXPECT_VOCAB"

# 1. data/meta.json (guard in run.sh prep)
if [ -f data/meta.json ]; then
  have=$(uv run python -c "import json; print(json.load(open('data/meta.json')).get('vocab'))")
  say "data/meta.json vocab=$have"
  if [ "$have" != "$EXPECT_VOCAB" ]; then
    if [ "${PREP_FORCE:-0}" = "1" ]; then
      say "PREP_FORCE=1: prep will overwrite data/ (vocab $have -> $EXPECT_VOCAB)."
    else
      warn "data/ holds vocab=$have but SIZE=$SIZE needs vocab=$EXPECT_VOCAB."
      warn "Move data/ aside, or run with PREP_FORCE=1."
    fi
  fi
else
  say "data/meta.json: absent (fresh volume, prep will create vocab=$EXPECT_VOCAB)"
fi

# 2. data/*.bin token counts (guard in pretrain.py: a 16k-tokenized train.bin
#    sitting next to a 32k meta.json passes every other check)
if [ -f data/meta.json ]; then
  if ! uv run python - <<'PY'
import json
from pathlib import Path

meta = json.loads(Path("data/meta.json").read_text())
bad = False
for name in ("train", "val"):
    path = Path(f"data/{name}.bin")
    if not path.exists():
        print(f"data/{name}.bin: absent")
        continue
    actual = path.stat().st_size // 2  # uint16 token ids
    expected = meta.get(f"{name}_tokens")
    if expected is None:
        print(f"data/{name}.bin: {actual} tokens (meta.json records no count)")
    elif actual != expected:
        print(f"data/{name}.bin has {actual} tokens but meta.json records {expected}.")
        bad = True
    else:
        print(f"data/{name}.bin: {actual} tokens (matches meta.json)")
raise SystemExit(1 if bad else 0)
PY
  then
    warn "The .bin files do not belong to this meta.json: regenerate data/ with prep."
  fi
fi

# 3. Lineage tokenizer (guard in prep.py)
if [ -f "$TOK" ]; then
  existing=$(uv run python -c "from tokenizers import Tokenizer; print(Tokenizer.from_file('$TOK').get_vocab_size())")
  say "$TOK vocab=$existing"
  if [ "$existing" != "$EXPECT_VOCAB" ]; then
    warn "$TOK has vocab=$existing but prep needs vocab=$EXPECT_VOCAB."
    warn "Move it aside or drop --vocab."
  fi
else
  say "$TOK: absent (prep will train vocab=$EXPECT_VOCAB)"
fi

# 4. Disk (info, never fails the check on its own unless /workspace is full)
if [ -d /workspace ]; then
  df -h /workspace
else
  df -h .
fi
ls -lh data/ 2>/dev/null || say "data/: absent"
ls -lh models/ models/*/ 2>/dev/null || true

if [ "$fail" -ne 0 ]; then
  echo "preflight: FAIL - prep or pretrain would refuse. Fix above, then re-run $0." >&2
  exit 1
fi
echo "preflight: OK - cloud/run.sh prep can run (SIZE=$SIZE)."
