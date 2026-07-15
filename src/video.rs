use std::{
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use crate::{
    inference::PoseInferencer,
    render::PoseRenderer,
    types::{coco17_to_halpe26, select_best_person, OpenSimDoc, OpenSimPerson},
};
use anyhow::{bail, Context, Result};

struct SavePayload {
    path: PathBuf,
    doc: OpenSimDoc,
}

struct VideoInfo {
    width: u32,
    height: u32,
    fps: String,
}

pub fn annotate_mp4(
    source: &Path,
    output: Option<&Path>,
    inferencer: &mut PoseInferencer,
    renderer: &mut PoseRenderer,
    kpt_conf: f32,
    opensim: Option<&Path>,
) -> Result<()> {
    let info = probe(source)?;
    let frame_bytes = info.width as usize * info.height as usize * 4;
    let mut decoder = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(source)
        .args([
            "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("starting ffmpeg video decoder")?;

    let mut encoder = if let Some(out) = output {
        let e = Command::new("ffmpeg")
            .args([
                "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
            ])
            .arg(format!("{}x{}", info.width, info.height))
            .args(["-r"])
            .arg(&info.fps)
            .args(["-i", "pipe:0", "-i"])
            .arg(source)
            .args([
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-cq",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-shortest",
            ])
            .arg(out)
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting ffmpeg NVENC video encoder")?;
        Some(e)
    } else {
        None
    };

    let mut reader = BufReader::new(
        decoder
            .stdout
            .take()
            .context("ffmpeg decoder has no stdout")?,
    );
    let mut writer = encoder
        .as_mut()
        .and_then(|enc| enc.stdin.take());

    let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("video").to_string();
    if let Some(ref dir) = opensim {
        std::fs::create_dir_all(dir)?;
    }
    let (tx, rx) = std::sync::mpsc::channel::<SavePayload>();
    let writer_thread = std::thread::spawn(move || {
        while let Ok(payload) = rx.recv() {
            if let Err(e) = std::fs::File::create(&payload.path)
                .context("creating opensim json file")
                .and_then(|file| serde_json::to_writer(file, &payload.doc).context("writing opensim json"))
            {
                eprintln!("Error saving OpenSim JSON to {}: {:?}", payload.path.display(), e);
            }
        }
    });

    let mut frame = vec![0; frame_bytes];
    let mut frame_number = 0usize;
    let started = Instant::now();
    let mut read_time = Duration::ZERO;
    let mut inference_time = Duration::ZERO;
    let mut render_time = Duration::ZERO;
    let mut write_time = Duration::ZERO;
    let mut annotated = Vec::with_capacity(frame_bytes);
    loop {
        let read_started = Instant::now();
        let mut read = 0;
        while read < frame.len() {
            let count = reader
                .read(&mut frame[read..])
                .context("reading decoded frame")?;
            if count == 0 {
                if read == 0 {
                    break;
                }
                bail!("ffmpeg decoder ended in the middle of frame {frame_number}");
            }
            read += count;
        }
        if read == 0 {
            break;
        }
        read_time += read_started.elapsed();

        let inference_started = Instant::now();
        let detections = inferencer.infer_rgba(&mut frame, info.width, info.height)?;
        inference_time += inference_started.elapsed();

        if let Some(ref dir) = opensim {
            let filename = format!("{}_{:06}.json", stem, frame_number);
            let path = dir.join(filename);
            let best = select_best_person(&detections).cloned();
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
            let _ = tx.send(SavePayload { path, doc });
        }

        if let Some(ref mut w) = writer {
            let render_started = Instant::now();
            renderer.render_into(
                &frame,
                info.width,
                info.height,
                &detections,
                kpt_conf,
                &mut annotated,
            )?;
            render_time += render_started.elapsed();
            
            let write_started = Instant::now();
            w.write_all(&annotated)
                .context("writing annotated frame to NVENC")?;
            write_time += write_started.elapsed();
        }
        
        frame_number += 1;
        if frame_number % 30 == 0 {
            let elapsed = started.elapsed().as_secs_f64();
            let frames = frame_number as f64;
            if writer.is_some() {
                eprintln!(
                    "Processed {frame_number} video frames in {elapsed:.1}s ({:.2} fps): read {:.1} ms/frame, infer {:.1} ms/frame, render {:.1} ms/frame, write {:.1} ms/frame",
                    frames / elapsed,
                    read_time.as_secs_f64() * 1_000.0 / frames,
                    inference_time.as_secs_f64() * 1_000.0 / frames,
                    render_time.as_secs_f64() * 1_000.0 / frames,
                    write_time.as_secs_f64() * 1_000.0 / frames,
                );
            } else {
                eprintln!(
                    "Processed {frame_number} video frames in {elapsed:.1}s ({:.2} fps): read {:.1} ms/frame, infer {:.1} ms/frame",
                    frames / elapsed,
                    read_time.as_secs_f64() * 1_000.0 / frames,
                    inference_time.as_secs_f64() * 1_000.0 / frames,
                );
            }
        }
    }
    drop(writer);
    drop(tx);
    let _ = writer_thread.join();
    if !decoder.wait()?.success() {
        bail!("ffmpeg video decoder failed");
    }
    if let Some(mut enc) = encoder {
        if !enc.wait()?.success() {
            bail!("ffmpeg NVENC encoder failed");
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let frames = frame_number as f64;
    if output.is_some() {
        println!(
            "Wrote {frame_number} annotated frames to {} in {elapsed:.1}s ({:.2} fps; infer {:.1} ms/frame, render {:.1} ms/frame)",
            output.unwrap().display(),
            frames / elapsed,
            inference_time.as_secs_f64() * 1_000.0 / frames,
            render_time.as_secs_f64() * 1_000.0 / frames,
        );
    } else {
        println!(
            "Processed {frame_number} video frames in {elapsed:.1}s ({:.2} fps; infer {:.1} ms/frame)",
            frames / elapsed,
            inference_time.as_secs_f64() * 1_000.0 / frames,
        );
    }
    println!(
        "Rust Inference Breakdown:\n\
         - Preprocess (CPU Resize + Normalization): {:.2} ms/frame\n\
         - TensorRT Engine Inference: {:.2} ms/frame\n\
         - Postprocess (Decode Pose): {:.2} ms/frame",
        inferencer.preprocess_accum.as_secs_f64() * 1_000.0 / frames,
        inferencer.engine_accum.as_secs_f64() * 1_000.0 / frames,
        inferencer.postprocess_accum.as_secs_f64() * 1_000.0 / frames
    );
    Ok(())
}

fn probe(source: &Path) -> Result<VideoInfo> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(source)
        .output()
        .context("running ffprobe")?;
    if !output.status.success() {
        bail!("ffprobe could not read {}", source.display());
    }
    let text = String::from_utf8(output.stdout).context("ffprobe returned non-UTF-8 output")?;
    let mut width = None;
    let mut height = None;
    let mut fps = None;
    for line in text.lines() {
        if let Some((key, value)) = line.split_once('=') {
            match key {
                "width" => width = value.parse().ok(),
                "height" => height = value.parse().ok(),
                "r_frame_rate" => fps = Some(value.to_owned()),
                _ => {}
            }
        }
    }
    Ok(VideoInfo {
        width: width.context("video width missing from ffprobe")?,
        height: height.context("video height missing from ffprobe")?,
        fps: fps.context("video frame rate missing from ffprobe")?,
    })
}
