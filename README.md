# UltraYoloRust

UltraYoloRust is a Rust implementation of the Ultralytics YOLO26 pose
inference pipeline. It runs a fixed-shape FP16 ONNX pose model through
TensorRT 11, decodes COCO-17 keypoints, renders the pose rig on the GPU, and
writes annotated images or MP4 videos.

The primary backend is direct TensorRT rather than ONNX Runtime. On first use,
the bundled `trtexec` utility converts the ONNX model into a cached TensorRT
engine. Subsequent runs deserialize that engine through the Rust
`tensorrt-infer` runtime.

## Features

- YOLO26 pose inference with COCO-17 keypoint decoding.
- Ultralytics-compatible centered letterbox preprocessing.
- TensorRT 11 engine creation and per-GPU engine cache.
- GPU pose-rig rendering with boxes, limbs, and keypoint joints.
- JPEG, PNG, and MP4 input support.
- MP4 output through FFmpeg with H.264 NVENC video encoding and source audio
  stream copy.
- Per-frame video timing for inference, rendering, and end-to-end throughput.

## Requirements

- NVIDIA GPU supported by TensorRT 11: compute capability 7.5 or newer.
- NVIDIA driver, CUDA, cuDNN, and TensorRT 11.
- `trtexec`, provided by the TensorRT installation.
- FFmpeg with the `h264_nvenc` encoder for MP4 output.
- Rust toolchain with a C++ compiler, required by `tensorrt-infer`.

The project has been tested on CachyOS/Arch with an RTX 4090 Laptop GPU,
CUDA 13, TensorRT 11.1, and FFmpeg NVENC.

### Arch / CachyOS Setup

Install the system dependencies:

```bash
sudo pacman -S --needed base-devel cuda cudnn ffmpeg
paru -S tensorrt
```

Install Rust if it is not already available:

```bash
sudo pacman -S --needed rustup
rustup default stable
```

Verify TensorRT and NVENC before building the project:

```bash
trtexec --version
ffmpeg -hide_banner -encoders | rg h264_nvenc
```

TensorRT engines are not portable across TensorRT major versions, CUDA
versions, GPU architectures, or model changes. Remove `trt_cache/` after any
of those changes.

## Build

```bash
cargo build --release
```

The project requires an FP16 YOLO26 pose ONNX model. The included test scripts
use `yolo26m-pose.fp16.onnx`.

## Usage

### Annotate An Image

```bash
cargo run --release -- \
  --model yolo26m-pose.fp16.onnx \
  --source image.jpg \
  --output annotated.jpg \
  --cache ./trt_cache
```

The first run builds `trt_cache/yolo26m-pose.fp16.trt11.engine`. Later runs
reuse the engine.

`--image` is accepted as an alias for `--source` for image workflows.

### Annotate A Video

```bash
cargo run --release -- \
  --model yolo26m-pose.fp16.onnx \
  --source input.mp4 \
  --output annotated.mp4 \
  --cache ./trt_cache
```

The video path decodes source frames with FFmpeg, runs direct TensorRT pose
inference, renders the rig through `wgpu` on the GPU, and encodes H.264 using
NVENC. Audio streams are copied from the source MP4 when present.

Example progress output:

```text
Processed 300 video frames in 8.6s (34.71 fps): infer 18.8 ms/frame, render 3.4 ms/frame
```

### Included Tests

```bash
./test.sh       # Builds/loads the engine and annotates image.jpg as annotated.jpg
./test_video.sh # Builds/loads the engine and annotates the configured MP4 fixture
```

Generated media, engines, and build artifacts are ignored by Git.

## CLI Reference

| Option | Default | Description |
| --- | --- | --- |
| `--model PATH` | Required | FP16 YOLO26 pose ONNX model. |
| `--source PATH` | Required | Input `.jpg`, `.jpeg`, `.png`, or `.mp4` file. `--image` is an alias. |
| `--output PATH` | Required | Annotated output. Image inputs require `.jpg`, `.jpeg`, or `.png`; MP4 inputs require `.mp4`. |
| `--device N` | `0` | CUDA device selected while building the TensorRT engine. |
| `--cache PATH` | `./trt_cache` | TensorRT engine cache directory. |
| `--conf FLOAT` | `0.25` | Minimum detection confidence. |
| `--kpt-conf FLOAT` | `0.5` | Minimum keypoint visibility for console output and rig rendering. |
| `--imgsz N` | `640` | Square model input size. This must match the static TensorRT engine input shape. |
| `--fp16` | `false` | Compatibility flag. Direct TensorRT uses the precision encoded in the ONNX model; use an FP16 ONNX model for FP16 inference. |

## Architecture

```text
JPEG / PNG / MP4
       |
       v
Ultralytics-style letterbox preprocessing
       |
       v
TensorRT 11 cached engine
       |
       v
YOLO26 pose decode and coordinate restoration
       |
       +--> console detections
       |
       v
wgpu pose overlay
       |
       v
JPEG / PNG or FFmpeg + NVENC MP4
```

## Notes

- Direct TensorRT requires a supported NVIDIA GPU. For older GPUs such as the
  GTX 1080 Ti (compute capability 6.1), use a separate CUDA-ONNX Runtime
  fallback implementation instead.
- The engine cache is intentionally excluded from version control.
- `test_video.sh` reports end-to-end, inference, and GPU rendering timing.
