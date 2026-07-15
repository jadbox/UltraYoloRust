/// A detected bounding box in original-image pixel space.
#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub conf: f32,
    pub class_id: u32,
}

/// One COCO keypoint.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    /// Pixel x in original image space.
    pub x: f32,
    /// Pixel y in original image space.
    pub y: f32,
    /// Visibility / confidence score [0, 1].
    pub conf: f32,
}

/// Full pose detection result for one person.
#[derive(Debug, Clone)]
pub struct PoseDetection {
    pub bbox: BBox,
    /// Always 17 for COCO pose models.
    pub keypoints: [Keypoint; 17],
}

/// COCO keypoint index → body-part name (for logging/display).
pub const COCO_KPT_NAMES: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

/// COCO skeleton limb pairs (0-indexed).
pub const COCO_SKELETON: [(usize, usize); 19] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
];

#[derive(serde::Serialize)]
pub struct OpenSimDoc {
    pub version: f64,
    pub people: Vec<OpenSimPerson>,
}

#[derive(serde::Serialize)]
pub struct OpenSimPerson {
    pub person_id: Vec<i32>,
    pub pose_keypoints_2d: Vec<f32>,
    pub face_keypoints_2d: Vec<f32>,
    pub hand_left_keypoints_2d: Vec<f32>,
    pub hand_right_keypoints_2d: Vec<f32>,
    pub pose_keypoints_3d: Vec<f32>,
    pub face_keypoints_3d: Vec<f32>,
    pub hand_left_keypoints_3d: Vec<f32>,
    pub hand_right_keypoints_3d: Vec<f32>,
}

pub fn coco17_to_halpe26(keypoints: &[Keypoint; 17]) -> Vec<f32> {
    let mut out = vec![0.0; 26 * 3];
    for (i, kp) in keypoints.iter().enumerate() {
        out[i * 3] = kp.x;
        out[i * 3 + 1] = kp.y;
        out[i * 3 + 2] = kp.conf;
    }
    out
}

pub fn select_best_person(detections: &[PoseDetection]) -> Option<&PoseDetection> {
    let mut best_det: Option<&PoseDetection> = None;
    let mut best_score = -1.0;
    for det in detections {
        let score: f32 = det.keypoints.iter()
            .map(|kp| if !kp.conf.is_nan() && kp.conf > 0.0 { kp.conf } else { 0.0 })
            .sum();
        if score > best_score {
            best_score = score;
            best_det = Some(det);
        }
    }
    if best_score <= 0.0 {
        None
    } else {
        best_det
    }
}

