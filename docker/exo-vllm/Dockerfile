# exo-vllm: Start from exo-spark (Python 3.13, MLX, exo, Rust bindings all working)
# and add vLLM + FlashInfer on top.
#
# exo-spark already has: Python 3.13, custom MLX CUDA (SM121a), exo + Rust bindings,
# PyTorch 2.11+cu130, NCCL, dashboard — all proven working.
# We just need to add: vLLM (built from source for 3.13) + FlashInfer wheels.

ARG EXO_BASE=192.168.0.181:5000/pnivek/exo-spark:latest
FROM ${EXO_BASE}

ARG BUILD_JOBS=16
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# ── Install FlashInfer from eugr's pre-built wheels ───────────────────────────
# py3-none and cp39-abi3 wheels work on Python 3.13
COPY docker/exo-vllm/wheels/ /tmp/wheels/
RUN pip install /tmp/wheels/flashinfer_python-*.whl /tmp/wheels/flashinfer_jit_cache-*.whl 2>/dev/null || \
    pip install flashinfer && \
    rm -rf /tmp/wheels

# ── Build vLLM from source for Python 3.13 ────────────────────────────────────
WORKDIR /build/vllm
ARG VLLM_REF=v0.18.1
RUN echo "Building vLLM at ref: ${VLLM_REF}" && \
    git clone https://github.com/vllm-project/vllm.git . && \
    git checkout ${VLLM_REF} && \
    git submodule update --init --recursive

RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/cuda.txt && \
    pip install -r requirements/build.txt

RUN pip install --no-build-isolation -e . 2>&1 | tail -5

# ── Additional runtime deps ───────────────────────────────────────────────────
RUN pip install \
    "ray[default]" \
    "fastsafetensors>=0.1.10" \
    "apache-tvm-ffi<0.2" \
    2>/dev/null || true

# ── Update exo source to disaggregated-inference-leo branch ───────────────────
WORKDIR /app
COPY src/ src/
COPY pyproject.toml README.md ./
RUN pip install --no-deps -e . 2>/dev/null || true

# ── vLLM environment ──────────────────────────────────────────────────────────
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_KV_CACHE_LAYOUT=NHD
ENV FASTSAFETENSORS_NOGDS=1
ENV MLX_CUDA_GRAPH_CACHE_SIZE=3000

# Tiktoken encodings
RUN python3 -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')" 2>/dev/null || true

EXPOSE 52415 52416 55000 55001 55002
ENTRYPOINT ["python3", "-m", "exo"]
