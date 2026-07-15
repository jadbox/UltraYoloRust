use anyhow::Result;
use image::DynamicImage;

use crate::{postprocess, preprocess, tensorrt::TensorRtEngine, types::PoseDetection};

pub struct PoseInferencer {
    engine: TensorRtEngine,
    imgsz: u32,
    conf: f32,
    kpt_conf: f32,
    pub preprocess_accum: std::time::Duration,
    pub engine_accum: std::time::Duration,
    pub postprocess_accum: std::time::Duration,
}

impl PoseInferencer {
    pub fn new(
        model: &std::path::Path,
        device: i32,
        _fp16: bool,
        cache: &std::path::Path,
        imgsz: u32,
        conf: f32,
        kpt_conf: f32,
    ) -> Result<Self> {
        Ok(Self {
            engine: TensorRtEngine::new(model, cache, device)?,
            imgsz,
            conf,
            kpt_conf,
            preprocess_accum: std::time::Duration::ZERO,
            engine_accum: std::time::Duration::ZERO,
            postprocess_accum: std::time::Duration::ZERO,
        })
    }

    pub fn infer(&mut self, image: &DynamicImage) -> Result<Vec<PoseDetection>> {
        let t0 = std::time::Instant::now();
        let (blob, letterbox) = preprocess::letterbox_to_tensor(image, self.imgsz);
        self.preprocess_accum += t0.elapsed();

        let t1 = std::time::Instant::now();
        let data = self.engine.infer(&blob)?;
        self.engine_accum += t1.elapsed();

        let t2 = std::time::Instant::now();
        let detections = postprocess::decode_pose(
            &data,
            &[1, 300, 57],
            &letterbox,
            self.conf,
            self.kpt_conf,
        );
        self.postprocess_accum += t2.elapsed();

        Ok(detections)
    }

    pub fn infer_rgba(
        &mut self,
        rgba: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<PoseDetection>> {
        let t0 = std::time::Instant::now();
        let (blob, letterbox) =
            preprocess::letterbox_rgba_to_tensor(rgba, width, height, self.imgsz);
        self.preprocess_accum += t0.elapsed();

        let t1 = std::time::Instant::now();
        let data = self.engine.infer(&blob)?;
        self.engine_accum += t1.elapsed();

        let t2 = std::time::Instant::now();
        let detections = postprocess::decode_pose(
            &data,
            &[1, 300, 57],
            &letterbox,
            self.conf,
            self.kpt_conf,
        );
        self.postprocess_accum += t2.elapsed();

        Ok(detections)
    }
}
