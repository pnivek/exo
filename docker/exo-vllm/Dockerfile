# exo-vllm: exo disaggregated inference on top of eugr's vLLM image
# Adds exo (Python + Rust libp2p bindings) to the existing vLLM container
# that already has vLLM, FlashInfer, PyTorch, NCCL, etc. for DGX Spark GB10.
#
# Build:
#   docker build -t exo-vllm:latest -f docker/exo-vllm/Dockerfile .
#
# The build context should be the exo repo root.

ARG VLLM_BASE=192.168.0.181:5000/pnivek/vllm-node:latest
FROM ${VLLM_BASE}

# Build deps for Rust + exo
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (needed for exo_pyo3_bindings / maturin)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin for building PyO3 bindings
RUN pip install maturin

# Copy exo source
WORKDIR /workspace/exo
COPY . .

# Build the Rust PyO3 bindings
RUN cd rust/exo_pyo3_bindings && maturin build --release && \
    pip install target/wheels/*.whl

# Install exo Python deps (skip mlx on Linux — only needed for lightweight cache ops)
# The vLLM container already has torch, so we skip heavy deps
RUN pip install --no-deps -e . 2>/dev/null || true && \
    pip install \
      anyio \
      sniffio \
      fastapi \
      uvicorn \
      starlette \
      pydantic \
      loguru \
      httpx \
      aiofiles \
      orjson \
      huggingface_hub \
      safetensors \
      "mlx-lm>=0.30" \
      2>/dev/null || true

# Build the dashboard (needs node)
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && \
    cd dashboard && npm install && npm run build && cd .. && \
    apt-get remove -y nodejs npm && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# vLLM env defaults
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_KV_CACHE_LAYOUT=NHD
ENV FASTSAFETENSORS_NOGDS=1

# exo ports: API=52415, KV transfer=52416, libp2p=55000-55002
EXPOSE 52415 52416 55000 55001 55002

# Default: run exo
ENTRYPOINT ["python3", "-m", "exo"]
