use anyhow::Result;
use image::DynamicImage;
use ort::{inputs, session::Session, value::Tensor};

use crate::{mode, postprocess, preprocess, types::PoseDetection};

pub struct PoseInferencer {
    session: Session,
    imgsz: u32,
    conf: f32,
    kpt_conf: f32,
}

impl PoseInferencer {
    pub fn new(
        model: &std::path::Path,
        device: i32,
        fp16: bool,
        cache: &std::path::Path,
        imgsz: u32,
        conf: f32,
        kpt_conf: f32,
    ) -> Result<Self> {
        Ok(Self {
            session: mode::build_session(model, device, fp16, cache)?,
            imgsz,
            conf,
            kpt_conf,
        })
    }

    pub fn infer(&mut self, image: &DynamicImage) -> Result<Vec<PoseDetection>> {
        let (blob, letterbox) = preprocess::letterbox_to_tensor(image, self.imgsz);
        let input = Tensor::from_array(([1i64, 3, self.imgsz as i64, self.imgsz as i64], blob))
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        let outputs = self.session.run(inputs!["images" => input])?;
        let (shape, data) = outputs["output0"].try_extract_tensor::<f32>()?;

        Ok(postprocess::decode_pose(
            data,
            &shape.to_vec(),
            &letterbox,
            self.conf,
            self.kpt_conf,
        ))
    }
}
