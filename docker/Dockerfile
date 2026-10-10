# puyotrain の学習用イメージ(Linux + Rust + PyTorch)。
# ソースはイメージに入れず、実行時にリポジトリを /work にマウントしてビルドする
# (git pull するだけで最新のコードになる)。使い方は README の「Docker で学習する」。

# フル版のイメージには gcc / curl / git が入っているので apt は使わない
FROM python:3.12-bookworm

# PyTorch の配布元。GPU ドライバが新しければ cu130、古めなら cu126 などに変えられる。
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    VIRTUAL_ENV=/opt/venv \
    PYTHONPATH=/work/python \
    CARGO_TARGET_DIR=/cache/target
ENV PATH=/opt/venv/bin:/opt/cargo/bin:$PATH

# Rust(rustup の公式インストーラを直接取得)
RUN arch="$(uname -m)" \
    && curl -sSfL -o /tmp/rustup-init \
        "https://static.rust-lang.org/rustup/dist/${arch}-unknown-linux-gnu/rustup-init" \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --profile minimal --default-toolchain stable --no-modify-path \
    && rm /tmp/rustup-init \
    && chmod -R a+rwX /opt/rustup /opt/cargo

RUN python -m venv /opt/venv && pip install --upgrade pip
RUN pip install torch --index-url "${TORCH_INDEX_URL}"
RUN pip install maturin pytest numpy gymnasium stable-baselines3 sb3-contrib \
        tensorboard pyyaml pillow imageio imageio-ffmpeg

COPY docker/entrypoint.sh /usr/local/bin/puyo-entrypoint
# Windows で clone すると改行が CRLF になることがあるので LF に直す
RUN sed -i 's/\r$//' /usr/local/bin/puyo-entrypoint && chmod +x /usr/local/bin/puyo-entrypoint

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/puyo-entrypoint"]
CMD ["bash", "scripts/start_training.sh"]
