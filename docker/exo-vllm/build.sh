#!/usr/bin/env bash
# Build the exo-vllm Docker image on a DGX Spark.
#
# Prerequisites:
#   1. Copy FlashInfer wheels from eugr into docker/exo-vllm/wheels/:
#      cp /etc/komodo/repos/eugr/spark-vllm-docker/wheels/flashinfer_*.whl docker/exo-vllm/wheels/
#
#   2. Build the dashboard on a machine with Node.js:
#      cd dashboard && npm ci && npm run build && cd ..
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

# Check FlashInfer wheels exist
WHEELS_DIR="docker/exo-vllm/wheels"
if [ ! -d "$WHEELS_DIR" ] || [ -z "$(ls $WHEELS_DIR/flashinfer_*.whl 2>/dev/null)" ]; then
    echo "FlashInfer wheels not found in $WHEELS_DIR/"
    echo ""
    echo "Copying from eugr's spark-vllm-docker..."
    mkdir -p "$WHEELS_DIR"
    cp /etc/komodo/repos/eugr/spark-vllm-docker/wheels/flashinfer_*.whl "$WHEELS_DIR/" 2>/dev/null || {
        echo "ERROR: Could not find FlashInfer wheels."
        echo "Copy them manually: cp /etc/komodo/repos/eugr/spark-vllm-docker/wheels/flashinfer_*.whl $WHEELS_DIR/"
        exit 1
    }
    echo "Copied FlashInfer wheels."
fi

# Check dashboard is built
if [ ! -d "dashboard/build" ]; then
    echo "WARNING: dashboard/build not found. The image will not have the web UI."
    echo "Build it first: cd dashboard && npm ci && npm run build && cd .."
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
