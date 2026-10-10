#!/usr/bin/env bash
# 学習環境の確認: GPU が使えるか、テスト、学習速度の計測(Linux / Docker / WSL)。
#   bash scripts/check_env.sh [--skip-tests] [--skip-bench]
set -euo pipefail
cd "$(dirname "$0")/.."
SKIP_TESTS=0
SKIP_BENCH=0
for a in "$@"; do
    case "$a" in
        --skip-tests) SKIP_TESTS=1 ;;
        --skip-bench) SKIP_BENCH=1 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

echo "==> PyTorch / GPU"
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    print("CUDA available: True |", torch.cuda.get_device_name(0))
else:
    print("CUDA available: False (training will run on CPU)")
PY
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

if [ "$SKIP_TESTS" = 0 ]; then
    echo "==> Rust tests"
    cargo test --workspace --release -q
    echo "==> Python tests"
    python -m pytest -q
fi

if [ "$SKIP_BENCH" = 0 ]; then
    echo "==> Training speed (env steps/s); pick the fastest for start_training.sh"
    for n in 8 16; do
        python scripts/bench_train.py --devices cpu auto --steps 20480 --set "env.n_envs=$n"
    done
fi
echo "==> OK"
