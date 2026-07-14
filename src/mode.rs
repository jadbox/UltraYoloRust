//! ORT session initialisation with the TensorRT execution provider.

use anyhow::Result;
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};
use std::path::Path;

/// Build an ORT [`Session`] targeting TensorRT (falls back to CUDA, then CPU).
///
/// The `engine_cache_path` should point to a directory where ORT-TRT may
/// store its compiled engine cache. On first run this will build the engine;
/// subsequent runs load it from cache and start in <1 s.
pub fn build_session(
    model_path: &Path,
    device_id: i32,
    fp16: bool,
    engine_cache_path: &Path,
) -> Result<Session> {
    // TensorRT EP — with engine caching so the first expensive build is amortised.
    let trt_ep = TensorRTExecutionProvider::default()
        .with_device_id(device_id)
        .with_fp16(fp16)
        .with_engine_cache(true)
        .with_engine_cache_path(engine_cache_path.to_str().unwrap());

    // CUDA EP as a fallback (in case TRT is unavailable on this machine).
    let cuda_ep = CUDAExecutionProvider::default().with_device_id(device_id);

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?
        .with_execution_providers([trt_ep.build(), cuda_ep.build()])
        .map_err(|err| anyhow::anyhow!(err.to_string()))?
        .commit_from_file(model_path)?;

    Ok(session)
}
