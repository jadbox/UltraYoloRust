# TensorRT 11 Runtime

The primary inference path uses the installed TensorRT 11 runtime directly via
the `tensorrt-infer` Rust crate. ONNX Runtime is not used for inference.

## Engine Build

The first image or video run invokes `trtexec` to convert the FP16 ONNX model
into a cached engine under `trt_cache/`. Engines are specific to TensorRT,
CUDA, GPU architecture, and model contents; delete the cache after upgrading
any of those components.

## Test

Run the full Rust video pipeline with:

```bash
./test_video.sh
```

It builds or loads the TensorRT engine, then executes TensorRT inference, GPU
pose rendering, and NVENC video encoding.
