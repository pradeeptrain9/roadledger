#!/usr/bin/env bash
# Fetch the YuNet face detector used for privacy scrubbing.
# Pinned to a specific commit so the download is reproducible.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODEL_DIR"

# Pinned to a tag rather than main so the download is reproducible, as the
# submission requires pinned dependencies. Note the media.githubusercontent.com
# host: opencv_zoo keeps models in git-LFS, and the ordinary raw.githubusercontent
# URL returns a 131-byte LFS pointer that fails to parse as ONNX. The checksum
# check below is what caught that.
YUNET_URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/4.10.0/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
TARGET="$MODEL_DIR/face_detection_yunet.onnx"
EXPECTED_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

if [ -f "$TARGET" ]; then
  echo "already present: $TARGET"
  exit 0
fi

echo "downloading YuNet face detector..."
curl -fsSL "$YUNET_URL" -o "$TARGET"

# Verify the download, since this model gates whether faces get blurred.
if command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "$TARGET" | awk '{print $1}')"
  if [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
    echo "checksum mismatch for $TARGET" >&2
    echo "  expected $EXPECTED_SHA256" >&2
    echo "  actual   $ACTUAL" >&2
    echo "Refusing to install an unverified privacy model." >&2
    rm -f "$TARGET"
    exit 1
  fi
  echo "checksum verified"
fi
echo "wrote $TARGET ($(wc -c < "$TARGET") bytes)"
echo
echo "Note: privacy.enabled defaults to True and face blurring fails closed."
echo "Without this model the pipeline refuses to process frames rather than"
echo "passing them through unscrubbed."
