# exo on top of eugr's vllm-node-tf5 (vLLM 0.19.1, FlashInfer 0.6.7, CUDA 13.2, Python 3.12)
#
# Build: docker build -t exo-vllm-tf5 -f docker/exo-vllm-tf5/Dockerfile .
# Run:   docker run --gpus all --network host exo-vllm-tf5 python3 -m exo --listen-port 55001

ARG BASE_IMAGE=192.168.0.181:5000/pnivek/vllm-node-tf5:latest
FROM ${BASE_IMAGE}

# ── Install Rust toolchain for building exo_pyo3_bindings ────────────────
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Install build deps ───────────────────────────────────────────────────
RUN pip install --no-cache-dir maturin rustworkx

# ── Copy exo source ─────────────────────────────────────────────────────
WORKDIR /app
COPY rust/ rust/
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY pyproject.toml README.md ./
COPY resources/ resources/
# Dashboard build artifact is optional — this node is a vLLM prefill worker.
# Create empty placeholder so exo's dashboard path finder doesn't crash.
RUN mkdir -p /app/dashboard/build

# ── Build Rust bindings for Python 3.12 ──────────────────────────────────
ARG BUILD_JOBS=16
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
RUN cd rust/exo_pyo3_bindings && \
    maturin build --release --interpreter python3 2>&1 | tail -5 && \
    pip install /app/target/wheels/exo_pyo3_bindings*.whl

# ── Install exo (no-deps since most deps are already in the base image) ──
RUN pip install --no-deps -e . 2>&1 | tail -3

# ── Install any missing exo deps ────────────────────────────────────────
# Install all exo deps. Pin anyio to 4.11.0 (base image has newer version
# that renamed MemoryObjectStreamState → _MemoryObjectStreamState).
# Use pip install -e .[cuda] but first override anyio.
RUN pip install --no-cache-dir "anyio==4.11.0" && \
    pip install -e ".[cuda]" 2>&1 | tail -5 || \
    pip install -e . 2>&1 | tail -5
# Verify all exo imports work
RUN python3 -c "from exo.main import main; print('All exo imports OK')"

# ── Tiktoken encodings ──────────────────────────────────────────────────
RUN python3 -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')" 2>/dev/null || true

# ── Environment ─────────────────────────────────────────────────────────
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_KV_CACHE_LAYOUT=NHD
ENV FASTSAFETENSORS_NOGDS=1
ENV MLX_CUDA_GRAPH_CACHE_SIZE=3000

EXPOSE 52415 55000 55001 55002
ENTRYPOINT ["python3", "-m", "exo"]
