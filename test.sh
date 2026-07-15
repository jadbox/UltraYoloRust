cargo run --release -- \
  --model yolo26m-pose.fp16.onnx \
  --source image.jpg \
  --output annotated.jpg \
  --fp16 \
  --cache ./trt_cache \
  --opensim ./pose/img
