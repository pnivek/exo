# exo-spark: DGX Spark (GB10 / SM121) container for disaggregated inference
# Packages: custom MLX (sm121 branch) + exo (feat/disaggregated-inference)
#
# Build on Sparkly (aarch64 required for CUDA kernel compilation):
#   docker build -t 192.168.0.181:5000/pnivek/exo-spark:latest .
# Run: see docker-compose.yml

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ARG BUILD_JOBS=16
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ── System deps ────────────────────────────────────────────────────────────────
# python3.13 is not in Ubuntu 24.04 default repos — requires deadsnakes PPA.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build build-essential git curl wget \
    python3.13 python3.13-venv python3.13-dev python3-pip \
    libopenblas-dev liblapack-dev liblapacke-dev libgfortran5 \
    libnccl2 libnccl-dev \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── Python venv (3.13 required by exo) ────────────────────────────────────────
RUN python3.13 -m venv /opt/exo-venv
ENV PATH="/opt/exo-venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/exo-venv"
RUN pip install --upgrade pip wheel setuptools

# ── Rust toolchain ─────────────────────────────────────────────────────────────
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Build custom MLX (pnivek/mlx@sm121) ───────────────────────────────────────
# MLX_CUDA_ARCHITECTURES must be set explicitly: __nvcc_device_query can't query
# the GPU during docker build (no device access). GB10 = SM121 = arch 121a.
# BUILD_SHARED_LIBS=OFF avoids the duplicate-symbol linker issue with shared libs.
WORKDIR /build/mlx
RUN git clone --depth=1 --branch sm121 https://github.com/pnivek/mlx.git .
RUN pip install nanobind scikit-build-core numpy

RUN cmake /build/mlx -B /build/mlx/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_CUDA=ON \
    -DMLX_CUDA_ARCHITECTURES=121a \
    -DNCCL_INCLUDE_DIRS=/usr/include \
    -DNCCL_LIBRARIES=/usr/lib/aarch64-linux-gnu/libnccl.so \
    -DMLX_BUILD_PYTHON_BINDINGS=ON \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLAPACK_INCLUDE_DIRS=/usr/include/aarch64-linux-gnu \
    -DPython3_EXECUTABLE=/opt/exo-venv/bin/python3 \
    -DMLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY=/build/mlx/build/mlx

RUN cmake --build /build/mlx/build --parallel ${BUILD_JOBS}

# Install into venv: copy core.so into the source tree, then pip install (no-deps)
RUN cp $(find /build/mlx/build -name "core.cpython*.so" | head -1) /build/mlx/python/mlx/
RUN CMAKE_ARGS="-DMLX_BUILD_CUDA=ON -DMLX_CUDA_ARCHITECTURES=121a -DNCCL_INCLUDE_DIRS=/usr/include -DNCCL_LIBRARIES=/usr/lib/aarch64-linux-gnu/libnccl.so -DBUILD_SHARED_LIBS=OFF" \
    pip install --no-build-isolation --no-deps /build/mlx
# CCCL headers: MLX JIT looks for <cuda/std/tuple> via /usr/local/cuda/include
# but on sbsa-linux the CCCL headers are under targets/sbsa-linux/include/cccl/.
# Symlink so /usr/local/cuda/include/cuda/std/tuple resolves correctly.
RUN ln -sf /usr/local/cuda/targets/sbsa-linux/include/cccl/cuda /usr/local/cuda/include/cuda

RUN python3 -c "import mlx.core as mx; print('MLX installed:', mx.__version__)" || echo "MLX import test skipped (no GPU during build)"

# ── Clone exo (feat/disaggregated-inference) ──────────────────────────────────
WORKDIR /app
RUN git clone --depth=1 --branch feat/disaggregated-inference \
    https://github.com/pnivek/exo.git .

# ── Rust bindings — must come BEFORE pip install exo ──────────────────────────
# exo-pyo3-bindings is a local workspace package, not on PyPI.
# Build it first so pip can resolve it when installing exo's deps.
RUN pip install maturin && \
    maturin develop -m rust/exo_pyo3_bindings/Cargo.toml

# ── Install exo Python deps ────────────────────────────────────────────────────
# pip install pulls mlx==0.30.6 as a placeholder (from exo's lockfile).
# We force-reinstall our custom sm121 build immediately after.
RUN pip install -e ".[cuda]"
RUN CMAKE_ARGS="-DMLX_BUILD_CUDA=ON -DMLX_CUDA_ARCHITECTURES=121a -DNCCL_INCLUDE_DIRS=/usr/include -DNCCL_LIBRARIES=/usr/lib/aarch64-linux-gnu/libnccl.so -DBUILD_SHARED_LIBS=OFF" \
    pip install --no-build-isolation --force-reinstall /build/mlx

# ── Dashboard ──────────────────────────────────────────────────────────────────
RUN cd /app/dashboard && npm ci && npm run build

# ── Runtime ────────────────────────────────────────────────────────────────────
ENV MLX_CUDA_GRAPH_CACHE_SIZE=3000
ENV EXO_MODELS_PATH=/models
EXPOSE 52415 52416 55000 55001 55002

ENTRYPOINT ["python3", "-m", "exo"]
