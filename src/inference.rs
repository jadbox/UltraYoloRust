use anyhow::Result;
use image::DynamicImage;

use crate::{postprocess, preprocess, tensorrt::TensorRtEngine, types::PoseDetection};

pub struct PoseInferencer {
    engine: TensorRtEngine,
    imgsz: u32,
    conf: f32,
    kpt_conf: f32,
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
        })
    }

    pub fn infer(&mut self, image: &DynamicImage) -> Result<Vec<PoseDetection>> {
        let (blob, letterbox) = preprocess::letterbox_to_tensor(image, self.imgsz);
        self.run(blob, letterbox)
    }

    pub fn infer_rgba(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<PoseDetection>> {
        let (blob, letterbox) =
            preprocess::letterbox_rgba_to_tensor(rgba, width, height, self.imgsz);
        self.run(blob, letterbox)
    }

    fn run(
        &mut self,
        blob: Vec<f32>,
        letterbox: preprocess::LetterboxInfo,
    ) -> Result<Vec<PoseDetection>> {
        let data = self.engine.infer(&blob)?;
        let shape = [1, 300, 57];

        Ok(postprocess::decode_pose(
            &data,
            &shape,
            &letterbox,
            self.conf,
            self.kpt_conf,
        ))
    }
}
