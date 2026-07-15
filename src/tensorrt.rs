use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{bail, Context, Result};
use tensorrt_infer::{CudaBuffer, CudaStream, TrtContext, TrtDataType, TrtEngine};

pub struct TensorRtEngine {
    // Field order matters: buffers and context must be destroyed before engine.
    buffers: Vec<CudaBuffer>,
    context: TrtContext,
    stream: CudaStream,
    _engine: TrtEngine,
    binding_ptrs: Vec<*mut c_void>,
    input_index: usize,
    output_index: usize,
    input_count: usize,
    output_count: usize,
}

impl TensorRtEngine {
    pub fn new(model: &Path, cache: &Path, device: i32) -> Result<Self> {
        let engine_path = cache.join(engine_name(model));
        if !engine_path.exists() {
            println!("Building TensorRT 11 engine: {}", engine_path.display());
            let status = Command::new("trtexec")
                .arg(format!("--onnx={}", model.display()))
                .arg(format!("--saveEngine={}", engine_path.display()))
                .arg(format!("--device={device}"))
                .arg("--builderOptimizationLevel=5")
                .status()
                .context("starting trtexec to build the TensorRT engine")?;
            if !status.success() {
                bail!("trtexec could not build the TensorRT engine");
            }
        }

        let engine =
            TrtEngine::from_file(engine_path.to_str().context("engine path is not UTF-8")?)
                .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        let bindings = engine.bindings();
        let input_index = bindings
            .iter()
            .position(|binding| binding.is_input)
            .context("TensorRT engine has no input binding")?;
        let output_index = bindings
            .iter()
            .position(|binding| !binding.is_input)
            .context("TensorRT engine has no output binding")?;
        if bindings.len() != 2
            || bindings[input_index].data_type != TrtDataType::Float
            || bindings[output_index].data_type != TrtDataType::Float
        {
            bail!("expected one FP32 input and one FP32 output in the TensorRT engine");
        }
        let buffers = bindings
            .iter()
            .map(|binding| {
                CudaBuffer::new(binding.byte_size).map_err(|err| anyhow::anyhow!(err.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let binding_ptrs = buffers.iter().map(CudaBuffer::as_ptr).collect();
        let context = engine
            .create_context()
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        let stream = CudaStream::new().map_err(|err| anyhow::anyhow!(err.to_string()))?;
        println!(
            "Inference backend: direct TensorRT 11 ({})",
            engine_path.display()
        );
        Ok(Self {
            buffers,
            context,
            stream,
            _engine: engine,
            binding_ptrs,
            input_index,
            output_index,
            input_count: bindings[input_index].byte_size / std::mem::size_of::<f32>(),
            output_count: bindings[output_index].byte_size / std::mem::size_of::<f32>(),
        })
    }

    pub fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_count {
            bail!(
                "TensorRT engine expects {} input values, got {}",
                self.input_count,
                input.len()
            );
        }
        self.buffers[self.input_index]
            .copy_from_host(bytemuck::cast_slice(input), &self.stream)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        self.context
            .enqueue(&mut self.binding_ptrs, &self.stream)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        let mut output = vec![0.0; self.output_count];
        self.buffers[self.output_index]
            .copy_to_host(bytemuck::cast_slice_mut(&mut output), &self.stream)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        Ok(output)
    }
}

fn engine_name(model: &Path) -> PathBuf {
    let stem = model
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    PathBuf::from(format!("{stem}.trt11.engine"))
}
