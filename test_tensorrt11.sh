#!/usr/bin/env bash
set -euo pipefail

model="yolo26m-pose.fp16.onnx"
engine="trt_cache/yolo26m-pose.fp16.trt11.engine"

mkdir -p trt_cache

if [[ -f "$engine" ]]; then
  echo "Benchmarking cached TensorRT 11 engine: $engine"
  trtexec --loadEngine="$engine" --warmUp=500 --duration=10
else
  echo "Building and benchmarking TensorRT 11 engine: $engine"
  # The ONNX model is already FP16 and has a fixed 1x3x640x640 input shape.
  trtexec --onnx="$model" --saveEngine="$engine" --warmUp=500 --duration=10
fi
