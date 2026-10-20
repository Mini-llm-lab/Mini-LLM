#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=40M
STEPS=35096
PREP_MILLIONS=4600
TEACHER=${TEACHER:-}
KD_WEIGHT=${KD_WEIGHT:-0.5}
KD_TEMP=${KD_TEMP:-2.0}

case "$SIZE" in
  40M|150M) EXPECT_VOCAB=32768 ;;
  *) EXPECT_VOCAB=16384 ;;
esac

BASE="models/$((EXPECT_VOCAB / 1024))k/${SIZE}/mini-llm-base-${SIZE}-${STEPS}steps.pt"
ASSISTANT="models/$((EXPECT_VOCAB / 1024))k/${SIZE}/mini-llm-assistant-${SIZE}.pt"

export UV_CACHE_DIR=/workspace/.uv-cache
export UV_LINK_MODE=copy
export HF_HOME=/workspace/.hf
export PYTHONUNBUFFERED=1

LOGDIR=/workspace/logs
mkdir -p "$LOGDIR" "$UV_CACHE_DIR" "$HF_HOME"

cmd="${1:-help}"
case "$cmd" in
setup)
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv sync
  nvidia-smi
  uv run python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "CUDA not available" >&2; exit 1; }
  SIZE="$SIZE" cloud/preflight.sh
  ;;
teacher)
  uv run teacher
  ;;
test)
  uv run pytest
  EVAL_DIR="${MINI_EVAL_DIR:-../Mini-Eval}"
  if [ -d "$EVAL_DIR" ] && [ "$(uname -s)" = "Linux" ]; then
    (cd "$EVAL_DIR" && uv run pytest)
  fi
  ;;
prep)
  if [ -f data/meta.json ]; then
    have=$(uv run python -c "import json; print(json.load(open('data/meta.json')).get('vocab'))")
    if [ "$have" != "$EXPECT_VOCAB" ] && [ "${PREP_FORCE:-0}" != "1" ]; then
      echo "data/ holds vocab=$have, but SIZE=$SIZE needs vocab=$EXPECT_VOCAB." >&2
      echo "Move data/ aside, or set PREP_FORCE=1 to overwrite it." >&2
      exit 1
    fi
  fi
  if [ "$EXPECT_VOCAB" = "32768" ]; then
    vocab_args=(--vocab 32768 --tokenizer-docs 300000)
  else
    vocab_args=()
  fi
  setsid nohup uv run prep "$PREP_MILLIONS" "${vocab_args[@]}" >"$LOGDIR/prep.log" 2>&1 < /dev/null &
  echo "prep started -> $LOGDIR/prep.log"
  ;;
smoke)
  args=(uv run pretrain 100 --size "$SIZE" --device cuda --save-every 50 --milestone-every 1000000)
  if [ -n "$TEACHER" ]; then
    args+=(--teacher "$TEACHER" --kd-weight "$KD_WEIGHT" --kd-temp "$KD_TEMP")
  fi
  "${args[@]}"
  rm -f models/*/*/mini-llm-base-*-100steps*.pt
  echo "smoke done, note the tok/s above"
  ;;
train)
  args=(uv run pretrain "$STEPS" --size "$SIZE" --device cuda)
  if [ -n "$TEACHER" ]; then
    args+=(--teacher "$TEACHER" --kd-weight "$KD_WEIGHT" --kd-temp "$KD_TEMP")
  fi
  if compgen -G "models/*/*/mini-llm-base-*-${STEPS}steps.state.pt" > /dev/null; then
    args+=(--resume)
  fi
  setsid nohup "${args[@]}" >"$LOGDIR/train.log" 2>&1 < /dev/null &
  echo "train started (${args[*]}) -> $LOGDIR/train.log"
  ;;
sft)
  setsid nohup uv run sft 2 "$BASE" --device cuda >"$LOGDIR/sft.log" 2>&1 < /dev/null &
  echo "sft started (2 epochs on $BASE) -> $LOGDIR/sft.log"
  ;;
eval)
  EVAL_DIR="${MINI_EVAL_DIR:-../Mini-Eval}"
  MODELS=(mini-llm-40M-35096steps mini-llm-150M-34332steps mini-llm-70M-22000steps
          SmolLM2-135M pythia-160m-step143000 pythia-70m-step143000)
  setsid nohup bash -c "cd '$EVAL_DIR' && uv run mini-eval check mini-llm-40M-35096steps --device cuda && uv run mini-eval run small_lm ${MODELS[*]} --device cuda --require-clean" \
    >"$LOGDIR/eval.log" 2>&1 < /dev/null &
  echo "eval started (${#MODELS[@]} models, protocol v2) -> $LOGDIR/eval.log"
  ;;
status)
  nvidia-smi || true
  for f in "$LOGDIR"/*.log; do
    [ -e "$f" ] || continue
    echo "== $f =="
    tail -5 "$f"
  done
  echo "== models/ =="; ls -lh models/ models/*/ models/*/*/ || true
  echo "== milestones =="; ls -lh models/*/*/milestones/ || true
  echo "== data/ =="; ls -lh data/ || true
  df -h /workspace
  ;;
pack)
  TOK="models/$((EXPECT_VOCAB / 1024))k/tok.json"
  files=("$TOK" data/meta.json)
  while IFS= read -r f; do files+=("$f"); done < <(find models -maxdepth 3 -name "mini-llm-base-${SIZE}-*.pt" ! -name '*.state.pt' ! -name '*.at*.pt' | sort)
  while IFS= read -r f; do files+=("$f"); done < <(find models -maxdepth 3 -name "mini-llm-assistant-${SIZE}.pt" | sort)
  tar -cf /workspace/export.tar "${files[@]}"
  ls -lh /workspace/export.tar
  echo "runpodctl send /workspace/export.tar"
  ;;
export-hf)
  cloud/run.sh test
  uv run export "$BASE" --base
  uv run export "$ASSISTANT"
  ;;
*)
  echo "usage: $0 {setup|prep|teacher|test|smoke|train|sft|eval|status|pack|export-hf}"
  exit 1
  ;;
esac
