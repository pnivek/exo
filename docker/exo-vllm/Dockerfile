# exo-vllm: exo on top of eugr's vllm-node (NO vLLM compilation needed)
#
# eugr's image already has vLLM 0.18 + FlashInfer + PyTorch + CUDA for GB10.
# We add: exo (Rust bindings + Python), MLX (built from source for CUDA),
# and the dashboard.

ARG VLLM_BASE=192.168.0.181:5000/pnivek/vllm-node:latest
FROM ${VLLM_BASE}

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=16
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev protobuf-compiler \
    cmake ninja-build python3-dev \
    libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Rust toolchain ─────────────────────────────────────────────────────────────
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Build MLX from source (PyPI mlx-cuda packages lack core.so) ───────────────
WORKDIR /build/mlx
RUN pip install nanobind scikit-build-core && \
    git clone --depth=1 https://github.com/ml-explore/mlx.git .

RUN CUDNN_INCLUDE=$(python3 -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'include'))") && \
    CUDNN_LIB_DIR=$(python3 -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))") && \
    cmake . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_CUDA=ON \
    -DMLX_CUDA_ARCHITECTURES=121a \
    -DNCCL_INCLUDE_DIRS=/usr/include \
    -DNCCL_LIBRARIES=/usr/lib/aarch64-linux-gnu/libnccl.so \
    -DCUDNN_INCLUDE_PATH=${CUDNN_INCLUDE} \
    -DCUDNN_LIBRARY_PATH=${CUDNN_LIB_DIR} \
    -DMLX_BUILD_PYTHON_BINDINGS=ON \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLAPACK_INCLUDE_DIRS=/usr/include/aarch64-linux-gnu \
    && cmake --build build --parallel ${BUILD_JOBS}

RUN cp $(find build -name "core.cpython*.so" | head -1) python/mlx/ && \
    pip install --no-build-isolation --no-deps . && \
    python3 -c "import mlx.core as mx; print('MLX:', mx.__version__)"

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
COPY pyproject.toml README.md ./
COPY dashboard/ dashboard/
COPY resources/ resources/
RUN sed -i 's/requires-python = ">=3.13"/requires-python = ">=3.12"/' pyproject.toml && \
    touch README.md

RUN pip install uv_build && pip install --no-deps -e .

# Install remaining deps
RUN pip install \
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
