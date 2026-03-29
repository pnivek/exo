# exo-vllm: exo + vLLM disaggregated inference for DGX Spark GB10
#
# Strategy: Start from exo-spark (Python 3.13 + exo + Rust bindings + MLX + PyTorch)
# and add vLLM + FlashInfer on top. This avoids the Python 3.12 vs 3.13 problem.
#
# Pre-built FlashInfer wheels (abi3/py3) are compatible with Python 3.13.
# vLLM must be built from source since eugr's wheel is cp312 only.
#
# Build on Sparkly (aarch64 + GPU required for some vLLM JIT):
#   docker build -t 192.168.0.181:5000/pnivek/exo-vllm:latest \
#     -f docker/exo-vllm/Dockerfile .
#
# Build context: exo repo root

ARG EXO_BASE=192.168.0.181:5000/pnivek/exo-spark:latest
FROM ${EXO_BASE}

ARG BUILD_JOBS=16
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV DEBIAN_FRONTEND=noninteractive

# ── Torch CUDA arch for vLLM compilation ───────────────────────────────────────
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# ── Install FlashInfer from eugr's pre-built wheels ───────────────────────────
# These are py3-none (pure Python) and cp39-abi3 (stable ABI) — work on 3.13.
# eugr's wheels are at /etc/komodo/repos/eugr/spark-vllm-docker/wheels/ on Sparks.
# The build script (or Komodo pre_build) copies them into the build context.
# Fallback: install from PyPI if wheels dir doesn't exist.
COPY docker/exo-vllm/wheels/ /tmp/wheels/
RUN if ls /tmp/wheels/flashinfer_*.whl 1>/dev/null 2>&1; then \
        echo "Installing FlashInfer from local wheels" && \
        pip install /tmp/wheels/flashinfer_*.whl; \
    else \
        echo "No local FlashInfer wheels, installing from PyPI..." && \
        pip install flashinfer; \
    fi && rm -rf /tmp/wheels

# ── Build vLLM from source for Python 3.13 ────────────────────────────────────
# We clone upstream vLLM and build. eugr's pre-built wheel is cp312 only.
WORKDIR /build/vllm
ARG VLLM_REF=main
RUN git clone --depth=1 --branch ${VLLM_REF} https://github.com/vllm-project/vllm.git . && \
    git submodule update --init --recursive

# vLLM build requirements
RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/cuda.txt && \
    pip install -r requirements/build.txt

# Build and install vLLM
RUN --mount=type=cache,id=vllm-ccache,target=/root/.ccache \
    pip install --no-build-isolation -e . 2>&1 | tail -20 || \
    pip install --no-build-isolation . 2>&1 | tail -20

# ── Additional runtime deps ───────────────────────────────────────────────────
RUN pip install \
    "ray[default]" \
    "fastsafetensors>=0.1.10" \
    "apache-tvm-ffi<0.2"

# ── Update exo source to disaggregated-inference-leo branch ───────────────────
# The base exo-spark image has exo installed at /app with dashboard already built.
# We overlay our updated source files on top.
WORKDIR /app
COPY src/ src/
COPY pyproject.toml .
# Re-install exo (in case deps changed) — skip Rust rebuild (already installed)
RUN pip install --no-deps -e . 2>/dev/null || true

# ── vLLM environment ──────────────────────────────────────────────────────────
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_KV_CACHE_LAYOUT=NHD
ENV FASTSAFETENSORS_NOGDS=1
ENV MLX_CUDA_GRAPH_CACHE_SIZE=3000

# Download tiktoken encodings (vLLM needs these)
RUN mkdir -p /workspace/tiktoken_encodings && \
    python3 -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')" 2>/dev/null || true

EXPOSE 52415 52416 55000 55001 55002
ENTRYPOINT ["python3", "-m", "exo"]
