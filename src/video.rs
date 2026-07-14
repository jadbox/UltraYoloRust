use std::{
    io::{BufReader, Read, Write},
    path::Path,
    process::{Command, Stdio},
};

use anyhow::{bail, Context, Result};
use image::{DynamicImage, RgbaImage};

use crate::{inference::PoseInferencer, render::PoseRenderer};

struct VideoInfo {
    width: u32,
    height: u32,
    fps: String,
}

pub fn annotate_mp4(
    source: &Path,
    output: &Path,
    inferencer: &mut PoseInferencer,
    renderer: &PoseRenderer,
    kpt_conf: f32,
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
    let mut encoder = Command::new("ffmpeg")
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
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("starting ffmpeg NVENC video encoder")?;

    let mut reader = BufReader::new(
        decoder
            .stdout
            .take()
            .context("ffmpeg decoder has no stdout")?,
    );
    let mut writer = encoder
        .stdin
        .take()
        .context("ffmpeg encoder has no stdin")?;
    let mut frame = vec![0; frame_bytes];
    let mut frame_number = 0usize;
    loop {
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

        let image = RgbaImage::from_raw(info.width, info.height, frame.clone())
            .context("decoded frame dimensions do not match its buffer")?;
        let detections = inferencer.infer(&DynamicImage::ImageRgba8(image))?;
        let annotated = renderer.render(&frame, info.width, info.height, &detections, kpt_conf)?;
        writer
            .write_all(&annotated)
            .context("writing annotated frame to NVENC")?;
        frame_number += 1;
        if frame_number % 30 == 0 {
            eprintln!("Processed {frame_number} video frames");
        }
    }
    drop(writer);
    if !decoder.wait()?.success() {
        bail!("ffmpeg video decoder failed");
    }
    if !encoder.wait()?.success() {
        bail!("ffmpeg NVENC encoder failed");
    }
    println!(
        "Wrote {frame_number} annotated frames to {}",
        output.display()
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
