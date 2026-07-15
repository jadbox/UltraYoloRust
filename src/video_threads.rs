//! Pipeline thread stages for parallel video processing.

use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, ChildStdout};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::time::{Duration, Instant};
use anyhow::{bail, Context, Result};

use crate::{
    render::PoseRenderer,
    types::{coco17_to_halpe26, select_best_person, OpenSimDoc, OpenSimPerson, PoseDetection},
    video::SavePayload,
};

/// Runs the background thread that saves OpenSim JSON payloads to files.
///
/// # Arguments
/// - `rx_json`: Channel receiver containing OpenSim documents and target write paths.
pub fn run_json_writer(rx_json: Receiver<SavePayload>) {
    while let Ok(payload) = rx_json.recv() {
        if let Err(e) = std::fs::File::create(&payload.path)
            .context("creating opensim json file")
            .and_then(|file| serde_json::to_writer(file, &payload.doc).context("writing opensim json"))
        {
            eprintln!("Error saving OpenSim JSON to {}: {:?}", payload.path.display(), e);
        }
    }
}

/// Runs the video decoder thread.
///
/// # Arguments
/// - `decoder_stdout`: Reader pipe from FFmpeg raw video decoder.
/// - `frame_bytes`: Bytes size of a single RGBA frame.
/// - `tx_dec`: Bounded sync channel to enqueue decoded frame buffers.
pub fn run_decoder(
    decoder_stdout: ChildStdout,
    frame_bytes: usize,
    tx_dec: SyncSender<(Vec<u8>, usize)>,
) -> Result<Duration> {
    let mut local_read_time = Duration::ZERO;
    let mut reader = BufReader::new(decoder_stdout);
    let mut f_num = 0usize;
    loop {
        let mut frame = vec![0; frame_bytes];
        let read_started = Instant::now();
        let mut read = 0;
        while read < frame.len() {
            let count = reader.read(&mut frame[read..])?;
            if count == 0 {
                if read == 0 {
                    return Ok(local_read_time); // EOF
                }
                bail!("ffmpeg decoder ended in the middle of frame {f_num}");
            }
            read += count;
        }
        local_read_time += read_started.elapsed();
        if tx_dec.send((frame, f_num)).is_err() {
            break;
        }
        f_num += 1;
    }
    Ok(local_read_time)
}

/// Helper to construct and queue an OpenSim JSON export payload for a frame's detections.
///
/// # Arguments
/// - `detections`: YOLO pose detections for the frame.
/// - `dir`: OpenSim output directory path.
/// - `stem`: Filename base/stem (e.g. video file name).
/// - `f_num`: Frame index.
/// - `tx_json`: Channel sender to queue the JSON save task.
fn export_opensim_frame(
    detections: &[PoseDetection],
    dir: &Path,
    stem: &str,
    f_num: usize,
    tx_json: &Sender<SavePayload>,
) {
    let filename = format!("{}_{:06}.json", stem, f_num);
    let path = dir.join(filename);
    let best = select_best_person(detections).cloned();
    let people = if let Some(det) = best {
        vec![OpenSimPerson {
            person_id: vec![-1],
            pose_keypoints_2d: coco17_to_halpe26(&det.keypoints),
            face_keypoints_2d: vec![],
            hand_left_keypoints_2d: vec![],
            hand_right_keypoints_2d: vec![],
            pose_keypoints_3d: vec![],
            face_keypoints_3d: vec![],
            hand_left_keypoints_3d: vec![],
            hand_right_keypoints_3d: vec![],
        }]
    } else {
        vec![]
    };
    let doc = OpenSimDoc {
        version: 1.3,
        people,
    };
    let _ = tx_json.send(SavePayload { path, doc });
}

/// Runs the overlay rendering and NVENC encoding thread.
///
/// # Arguments
/// - `rx_inf`: Channel receiver containing inferred frame buffers and keypoint detections.
/// - `opensim`: Optional directory to output OpenSim JSON files.
/// - `stem`: Base filename for exported JSON files.
/// - `tx_json`: Channel sender to enqueue JSON write operations.
/// - `writer`: Writer pipe to the FFmpeg NVENC encoder.
/// - `renderer`: Mutable reference to the GPU compositor.
/// - `kpt_conf`: Minimum confidence score for a keypoint to be rendered.
/// - `frame_bytes`: Byte size of a single RGBA frame.
/// - `width`: Frame width in pixels.
/// - `height`: Frame height in pixels.
pub fn run_renderer(
    rx_inf: Receiver<(Vec<u8>, Vec<PoseDetection>, usize)>,
    opensim: Option<PathBuf>,
    stem: String,
    tx_json: Sender<SavePayload>,
    mut writer: Option<ChildStdin>,
    renderer: &mut PoseRenderer,
    kpt_conf: f32,
    frame_bytes: usize,
    width: u32,
    height: u32,
) -> Result<(Duration, Duration)> {
    let mut local_render_time = Duration::ZERO;
    let mut local_write_time = Duration::ZERO;
    let mut annotated = Vec::with_capacity(frame_bytes);

    while let Ok((frame, detections, f_num)) = rx_inf.recv() {
        if let Some(ref dir) = opensim {
            export_opensim_frame(&detections, dir, &stem, f_num, &tx_json);
        }

        if let Some(ref mut w) = writer {
            let render_started = Instant::now();
            renderer.render_into(
                &frame,
                width,
                height,
                &detections,
                kpt_conf,
                &mut annotated,
            )?;
            local_render_time += render_started.elapsed();

            let write_started = Instant::now();
            w.write_all(&annotated)
                .context("writing annotated frame to NVENC")?;
            local_write_time += write_started.elapsed();
        }
    }
    Ok((local_render_time, local_write_time))
}
