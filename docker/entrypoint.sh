#!/usr/bin/env bash
# マウントされたソース(/work)から Rust 拡張をビルドしてから、指定のコマンドを実行する。
# ビルド結果は /cache(docker のボリューム)に残るので、2回目以降は数秒で終わる。
set -euo pipefail
cd /work
if [ ! -f pyproject.toml ]; then
    echo "error: mount the puyotrain repository at /work" >&2
    exit 1
fi
if [ "${PUYO_SKIP_BUILD:-0}" != "1" ]; then
    echo "[puyotrain] building the Rust extension..."
    maturin develop --release --skip-install
fi
exec "$@"
