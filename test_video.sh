#!/usr/bin/env bash
set -euo pipefail

cargo run --release -- \
  --model yolo26m-pose.fp16.onnx \
  --source "ACRM-LBP-CNP-2026-05-100000018-1-SQ-v3.mp4" \
  --output annotated.mp4 \
  --fp16 \
  --cache ./trt_cache \
  --opensim ./pose/video
