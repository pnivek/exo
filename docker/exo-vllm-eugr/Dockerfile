# exo (port/testing-vllm-on-main) layered on eugr's spark-vllm GB10-optimized
# image — the preferred vLLM runtime for DGX Spark (precompiled vLLM +
# FlashInfer tuned for sm_121a). See https://github.com/eugr/spark-vllm-docker
#
# The base image ships Python 3.12 with cp312 vLLM/FlashInfer wheels, while exo
# main pins requires-python ==3.13.*. We install exo with
# --ignore-requires-python — exo's source is 3.12-compatible (it shipped on
# >=3.12 until May) and rebuilding the optimized cp312 wheels for 3.13 would
# defeat the point of using eugr's builds.
#
# Build (repo root): docker build -t exo-vllm-eugr -f docker/exo-vllm-eugr/Dockerfile .
# Run:               docker run --gpus all --network host exo-vllm-eugr

ARG BASE_IMAGE=192.168.0.181:5000/pnivek/vllm-node-tf5:latest
FROM ${BASE_IMAGE}

# ── Rust toolchain for building the exo_rs bindings ─────────────────────
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

# ── exo source ──────────────────────────────────────────────────────────
WORKDIR /app
COPY rust/ rust/
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY pyproject.toml README.md ./
COPY resources/ resources/

# Prefill nodes don't serve the UI; placeholder keeps find_dashboard() happy.
RUN mkdir -p /app/dashboard/build/_app && \
    echo '<!DOCTYPE html><html><body>exo vllm prefill node</body></html>' > /app/dashboard/build/index.html

# ── Build exo_rs for the image's Python 3.12 ────────────────────────────
ARG BUILD_JOBS=16
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
RUN cd rust/exo_rs && \
    maturin build --release --interpreter python3 && \
    pip install --ignore-requires-python /app/target/wheels/exo_rs*.whl

# ── Install exo itself (deps come next; base image owns vllm/torch) ─────
RUN pip install --no-deps --ignore-requires-python -e . 2>&1 | tail -3

# exo's pure-python deps. anyio pinned: newer anyio renamed internals exo's
# channel wrappers reach into. Never touch vllm/torch/flashinfer here.
RUN pip install --no-cache-dir --ignore-requires-python \
    "anyio==4.11.0" aiofiles aiohttp "pydantic>=2.11.7" "fastapi>=0.116.1" \
    filelock "rustworkx>=0.17.1" "huggingface-hub>=1.8.0" psutil loguru \
    "tiktoken>=0.12.0" hypercorn "openai-harmony>=0.0.8" httpx tomlkit \
    python-multipart msgspec zstandard "transformers>=5.6.2" python-daemon \
    types-aiofiles nvidia-ml-py

# mlx (CPU backend) + Leo's mlx-lm fork: the vLLM engine uses mlx_lm's
# TokenizerWrapper for chat templating, and the shared runner import graph
# reaches mlx + mlx_lm.models.deepseek_v4 (fork-only). GPU MLX compute on
# this node is NOT expected — decode belongs on the Mac.
RUN pip install --no-cache-dir --ignore-requires-python \
    "mlx==0.32.0" "mlx-cpu==0.32.0" \
    "git+https://github.com/rltakashige/mlx-lm@leo/deepseek-v4"

RUN python3 -c "from exo.main import main; print('exo imports OK')" && \
    python3 -c "from exo.worker.engines.vllm.builder import VllmBuilder; print('vllm builder imports OK')"

# Pre-cache tiktoken encodings (offline nodes)
RUN python3 -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')" 2>/dev/null || true

# ── Runtime env (mirrors worker/runner/bootstrap.py for manual runs) ────
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_KV_CACHE_LAYOUT=NHD
ENV VLLM_BATCH_INVARIANT=1
ENV FASTSAFETENSORS_NOGDS=1

EXPOSE 52415 52413/udp 52414
ENTRYPOINT ["python3", "-m", "exo"]
