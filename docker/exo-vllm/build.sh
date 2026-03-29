#!/usr/bin/env bash
# Build the exo-vllm Docker image on a DGX Spark and push to local registry.
#
# Run from the exo repo root:
#   bash docker/exo-vllm/build.sh
#
# Or on a Spark via SSH:
#   ssh pnivek@192.168.0.172 "cd /path/to/exo && bash docker/exo-vllm/build.sh"

set -euo pipefail

REGISTRY="${REGISTRY:-192.168.0.181:5000}"
IMAGE_NAME="${IMAGE_NAME:-pnivek/exo-vllm}"
TAG="${TAG:-latest}"
VLLM_BASE="${VLLM_BASE:-${REGISTRY}/pnivek/vllm-node:latest}"

FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Building exo-vllm image..."
echo "  Base: ${VLLM_BASE}"
echo "  Tag:  ${FULL_TAG}"
echo ""

docker build \
  --build-arg "VLLM_BASE=${VLLM_BASE}" \
  -t "${FULL_TAG}" \
  -f docker/exo-vllm/Dockerfile \
  .

echo ""
echo "Pushing to registry..."
docker push "${FULL_TAG}"

echo ""
echo "Done! Image available at: ${FULL_TAG}"
echo ""
echo "To deploy via Komodo:"
echo "  sparkly: use compose.sparkly-head.yaml"
echo "  sparky:  use compose.sparky-worker.yaml"
