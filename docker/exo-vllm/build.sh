#!/usr/bin/env bash
# Build the exo-vllm Docker image on a DGX Spark.
#
# Run from the exo repo root:
#   bash docker/exo-vllm/build.sh
#
# Options (env vars):
#   REGISTRY      - Docker registry (default: 192.168.0.181:5000)
#   TAG           - Image tag (default: latest)
#   EXO_BASE      - Base exo-spark image (default: $REGISTRY/pnivek/exo-spark:latest)
#   VLLM_REF      - vLLM git ref to build (default: main)
#   NO_PUSH       - Set to 1 to skip pushing to registry

set -euo pipefail

REGISTRY="${REGISTRY:-192.168.0.181:5000}"
TAG="${TAG:-latest}"
EXO_BASE="${EXO_BASE:-${REGISTRY}/pnivek/exo-spark:latest}"
VLLM_REF="${VLLM_REF:-main}"
NO_PUSH="${NO_PUSH:-0}"

FULL_TAG="${REGISTRY}/pnivek/exo-vllm:${TAG}"
WHEELS_DIR="docker/exo-vllm/wheels"
EUGR_WHEELS="/etc/komodo/repos/eugr/spark-vllm-docker/wheels"

# Copy FlashInfer wheels from eugr if not already present
mkdir -p "$WHEELS_DIR"
if [ ! -f "$WHEELS_DIR/.copied" ]; then
    if [ -d "$EUGR_WHEELS" ] && ls "$EUGR_WHEELS"/flashinfer_*.whl 1>/dev/null 2>&1; then
        echo "Copying FlashInfer wheels from eugr..."
        cp "$EUGR_WHEELS"/flashinfer_*.whl "$WHEELS_DIR/"
        touch "$WHEELS_DIR/.copied"
    else
        echo "WARNING: eugr FlashInfer wheels not found at $EUGR_WHEELS"
        echo "FlashInfer will be installed from PyPI (may need to compile)."
    fi
fi

echo "============================================"
echo "Building exo-vllm image"
echo "  Base:     ${EXO_BASE}"
echo "  vLLM ref: ${VLLM_REF}"
echo "  Tag:      ${FULL_TAG}"
echo "============================================"
echo ""

docker build \
  --build-arg "EXO_BASE=${EXO_BASE}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "BUILD_JOBS=16" \
  -t "${FULL_TAG}" \
  -f docker/exo-vllm/Dockerfile \
  .

if [ "$NO_PUSH" != "1" ]; then
    echo ""
    echo "Pushing to registry..."
    docker push "${FULL_TAG}"
    echo "Pushed: ${FULL_TAG}"
fi

echo ""
echo "Done! Image: ${FULL_TAG}"
