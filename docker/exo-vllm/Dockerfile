# exo-vllm: exo on top of eugr's vllm-node (NO vLLM compilation needed)
#
# eugr's image already has vLLM 0.18 + FlashInfer + PyTorch + CUDA for GB10.
# We just add exo (Rust bindings + Python package) on top.
# Python 3.12 (from eugr's image) works fine — exo has no 3.13-only features.

ARG VLLM_BASE=192.168.0.181:5000/pnivek/vllm-node:latest
FROM ${VLLM_BASE}

ARG DEBIAN_FRONTEND=noninteractive

# ── System deps for Rust build ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# ── Rust toolchain ─────────────────────────────────────────────────────────────
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Build exo Rust bindings (libp2p networking) ───────────────────────────────
WORKDIR /workspace/exo
COPY rust/ rust/
COPY Cargo.toml Cargo.lock ./
RUN pip install maturin && \
    sed -i 's/requires-python = ">=3.13"/requires-python = ">=3.12"/' rust/exo_pyo3_bindings/pyproject.toml && \
    cd rust/exo_pyo3_bindings && \
    maturin build --release -i python3 && \
    pip install /workspace/exo/target/wheels/*.whl

# ── Install exo Python package ────────────────────────────────────────────────
COPY src/ src/
COPY pyproject.toml .
# Override requires-python so pip doesn't reject on 3.12
RUN sed -i 's/requires-python = ">=3.13"/requires-python = ">=3.12"/' pyproject.toml && \
    pip install --no-deps -e . && \
    pip install \
      "anyio==4.11.0" \
      "aiofiles>=24.1.0" \
      "aiohttp>=3.12" \
      "types-aiofiles>=24.1.0" \
      "rustworkx>=0.17.1" \
      "hypercorn>=0.18.0" \
      "tomlkit>=0.14.0" \
      "zstandard>=0.23.0" \
      "mlx-lm>=0.30" \
      2>/dev/null || true

# ── Dashboard ──────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && \
    cd /workspace/exo && npm --prefix dashboard install && npm --prefix dashboard run build && \
    apt-get remove -y nodejs npm && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# ── Runtime env ────────────────────────────────────────────────────────────────
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_KV_CACHE_LAYOUT=NHD
ENV FASTSAFETENSORS_NOGDS=1
ENV MLX_CUDA_GRAPH_CACHE_SIZE=3000

EXPOSE 52415 52416 55000 55001 55002
WORKDIR /workspace/exo
ENTRYPOINT ["python3", "-m", "exo"]
