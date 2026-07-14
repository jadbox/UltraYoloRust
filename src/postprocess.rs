//! YOLO26 pose postprocessor.
//!
//! YOLO26 is NMS-free (end-to-end head). The exported ONNX / TensorRT engine
//! produces a single output tensor shaped  [1, 300, 57]  where 300 is the
//! maximum number of detections (already filtered by the model's built-in
//! confidence/NMS logic) and 57 columns are:
//!
//!   col 0..=3  : x1, y1, x2, y2  in model-pixel space (0..model_size)
//!   col 4      : detection confidence
//!   col 5      : class_id  (always 0 = person for COCO pose)
//!   col 6..=8  : kpt0  (x, y, vis)  in model-pixel space
//!   col 9..=11 : kpt1  (x, y, vis)
//!   …
//!   col 54..=56: kpt16 (x, y, vis)
//!
//! Total: 6 + 17 * 3 = 57  ✓
//!
//! Reference: ultralytics examples/cpp/common/yolo_postprocess.hpp,
//!            PostprocessPose(), end2end branch (dim1 > dim2).

use crate::preprocess::LetterboxInfo;
use crate::types::{BBox, Keypoint, PoseDetection};

pub const NUM_KEYPOINTS: usize = 17;
const COLS: usize = 6 + NUM_KEYPOINTS * 3; // 57

/// Decode the raw `[1, max_det, 57]` output from YOLO26-pose into
/// [`PoseDetection`] values mapped back to original-image pixel space.
///
/// # Arguments
/// * `data`      – flat slice of the output tensor in row-major order.
/// * `shape`     – tensor shape, expected `[1, N, 57]`.
/// * `info`      – letterbox metadata from preprocessing.
/// * `conf_thr`  – minimum detection confidence (e.g. 0.25).
/// * `kpt_thr`   – minimum keypoint visibility to include (e.g. 0.5).
pub fn decode_pose(
    data: &[f32],
    shape: &[i64],
    info: &LetterboxInfo,
    conf_thr: f32,
    kpt_thr: f32, // kept for caller convenience; stored on each kpt
) -> Vec<PoseDetection> {
    // shape: [batch=1, max_det, 57]
    assert_eq!(shape.len(), 3, "expected 3-D output [1, N, 57]");
    assert_eq!(
        shape[2] as usize, COLS,
        "expected 57 columns (6 + 17*3), got {}",
        shape[2]
    );

    let max_det = shape[1] as usize;
    let inv = 1.0 / info.scale; // model-pixel → original-pixel
    let _ = kpt_thr; // stored on Keypoint.conf; caller decides the threshold

    let mut detections = Vec::new();

    for i in 0..max_det {
        let row = &data[i * COLS..(i + 1) * COLS];

        // col 4 = detection score
        let conf = row[4];
        if conf < conf_thr {
            continue; // the model sorts by confidence desc, so we can break
                      // early if we want, but filtering is safer.
        }

        // Bounding box: model-pixel → original-pixel.
        // Clamp to [0, orig_dim] to handle slight boundary overflows.
        let x1 = ((row[0] - info.pad_left as f32) * inv)
            .max(0.0)
            .min(info.orig_w as f32);
        let y1 = ((row[1] - info.pad_top as f32) * inv)
            .max(0.0)
            .min(info.orig_h as f32);
        let x2 = ((row[2] - info.pad_left as f32) * inv)
            .max(0.0)
            .min(info.orig_w as f32);
        let y2 = ((row[3] - info.pad_top as f32) * inv)
            .max(0.0)
            .min(info.orig_h as f32);

        let bbox = BBox {
            x1,
            y1,
            x2,
            y2,
            conf,
            class_id: row[5] as u32,
        };

        // Keypoints: columns 6, 7, 8 | 9, 10, 11 | … (stride 3)
        let mut keypoints = [Keypoint {
            x: 0.0,
            y: 0.0,
            conf: 0.0,
        }; NUM_KEYPOINTS];
        for k in 0..NUM_KEYPOINTS {
            let base = 6 + k * 3;
            keypoints[k] = Keypoint {
                x: ((row[base] - info.pad_left as f32) * inv)
                    .max(0.0)
                    .min(info.orig_w as f32),
                y: ((row[base + 1] - info.pad_top as f32) * inv)
                    .max(0.0)
                    .min(info.orig_h as f32),
                conf: row[base + 2],
            };
        }

        detections.push(PoseDetection { bbox, keypoints });
    }

    detections
}
