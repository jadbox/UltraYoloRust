use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use crate::{
    inference::PoseInferencer,
    render::PoseRenderer,
    types::{OpenSimDoc, PoseDetection},
};
use anyhow::{bail, Context, Result};

pub struct SavePayload {
    pub path: PathBuf,
    pub doc: OpenSimDoc,
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

    let decoder_stdout = decoder.stdout.take().context("ffmpeg decoder has no stdout")?;
    let writer = encoder
        .as_mut()
        .and_then(|enc| enc.stdin.take());

    let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("video").to_string();
    if let Some(ref dir) = opensim {
        std::fs::create_dir_all(dir)?;
    }

    let (tx_json, rx_json) = std::sync::mpsc::channel::<SavePayload>();
    let (tx_dec, rx_dec) = std::sync::mpsc::sync_channel::<(Vec<u8>, usize)>(4);
    let (tx_inf, rx_inf) = std::sync::mpsc::sync_channel::<(Vec<u8>, Vec<PoseDetection>, usize)>(4);

    let started = Instant::now();
    let mut read_time = Duration::ZERO;
    let mut render_time = Duration::ZERO;
    let mut write_time = Duration::ZERO;
    let mut frame_number = 0usize;

    std::thread::scope(|s| {
        // 1. JSON Writer Thread
        s.spawn(move || {
            crate::video_threads::run_json_writer(rx_json);
        });

        // 2. Decoder Thread
        let t_dec = s.spawn(move || -> Result<Duration> {
            crate::video_threads::run_decoder(decoder_stdout, frame_bytes, tx_dec)
        });

        // 3. Render/Encoder Thread
        let opensim_owned = opensim.map(|p| p.to_path_buf());
        let t_render = s.spawn(move || -> Result<(Duration, Duration)> {
            crate::video_threads::run_renderer(
                rx_inf,
                opensim_owned,
                stem,
                tx_json,
                writer,
                renderer,
                kpt_conf,
                frame_bytes,
                info.width,
                info.height,
            )
        });

        // 4. Main Thread (Runs Inference)
        let width = info.width;
        let height = info.height;

        while let Ok((mut frame, f_num)) = rx_dec.recv() {
            frame_number = f_num + 1;
            let detections = inferencer.infer_rgba(&mut frame, width, height)?;
            if tx_inf.send((frame, detections, f_num)).is_err() {
                break;
            }

            if frame_number % 30 == 0 {
                let elapsed = started.elapsed().as_secs_f64();
                let frames = frame_number as f64;
                eprintln!(
                    "Processed {frame_number} video frames in {elapsed:.1}s ({:.2} fps)",
                    frames / elapsed
                );
            }
        }

        // Drop tx_inf to close rx_inf and signal render thread to exit
        drop(tx_inf);

        // Collect decoder and render thread results
        read_time = t_dec.join().unwrap()?;
        let (r_time, w_time) = t_render.join().unwrap()?;
        render_time = r_time;
        write_time = w_time;
        Ok::<(), anyhow::Error>(())
    })?;

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
            "Wrote {frame_number} annotated frames to {} in {elapsed:.1}s ({:.2} fps)",
            output.unwrap().display(),
            frames / elapsed
        );
    } else {
        println!(
            "Processed {frame_number} video frames in {elapsed:.1}s ({:.2} fps)",
            frames / elapsed
        );
    }

    println!(
        "Rust Inference Pipeline Timings (Thread-Local Average):\n\
         - Decode: {:.2} ms/frame\n\
         - Preprocess (CPU Resize + Normalization): {:.2} ms/frame\n\
         - TensorRT Engine Inference: {:.2} ms/frame\n\
         - Postprocess (Decode Pose): {:.2} ms/frame\n\
         - Render Overlay: {:.2} ms/frame\n\
         - NVENC Write: {:.2} ms/frame",
        read_time.as_secs_f64() * 1_000.0 / frames,
        inferencer.preprocess_accum.as_secs_f64() * 1_000.0 / frames,
        inferencer.engine_accum.as_secs_f64() * 1_000.0 / frames,
        inferencer.postprocess_accum.as_secs_f64() * 1_000.0 / frames,
        render_time.as_secs_f64() * 1_000.0 / frames,
        write_time.as_secs_f64() * 1_000.0 / frames
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
