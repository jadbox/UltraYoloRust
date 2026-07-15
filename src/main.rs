mod inference;
mod postprocess;
mod preprocess;
mod render;
mod tensorrt;
mod types;
mod video;

use anyhow::{bail, Context, Result};
use clap::Parser;
use image::{open as open_image, DynamicImage, RgbaImage};
use std::path::PathBuf;

use crate::{inference::PoseInferencer, render::PoseRenderer, types::COCO_KPT_NAMES};

#[derive(Parser)]
#[command(name = "yolo26-pose", about = "YOLO26 pose inference via ORT/TensorRT")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    /// Input JPEG, PNG, or MP4 file. `--image` remains an alias for images.
    #[arg(long, visible_alias = "image")]
    source: PathBuf,
    /// Annotated JPEG, PNG, or MP4 destination.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = false)]
    fp16: bool,
    #[arg(long, default_value = "./trt_cache")]
    cache: PathBuf,
    #[arg(long, default_value_t = 0.25)]
    conf: f32,
    #[arg(long, default_value_t = 0.5)]
    kpt_conf: f32,
    #[arg(long, default_value_t = 640)]
    imgsz: u32,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    std::fs::create_dir_all(&args.cache)?;
    let mut inferencer = PoseInferencer::new(
        &args.model,
        args.device,
        args.fp16,
        &args.cache,
        args.imgsz,
        args.conf,
        args.kpt_conf,
    )?;
    let mut renderer = PoseRenderer::new()?;
    match extension(&args.source).as_deref() {
        Some("mp4") => {
            if extension(&args.output).as_deref() != Some("mp4") {
                bail!("an MP4 source requires an .mp4 output");
            }
            video::annotate_mp4(
                &args.source,
                &args.output,
                &mut inferencer,
                &mut renderer,
                args.kpt_conf,
            )
        }
        Some("jpg") | Some("jpeg") | Some("png") => {
            annotate_image(&args, &mut inferencer, &mut renderer)
        }
        _ => bail!(
            "unsupported source {}; expected JPEG, PNG, or MP4",
            args.source.display()
        ),
    }
}

fn annotate_image(
    args: &Cli,
    inferencer: &mut PoseInferencer,
    renderer: &mut PoseRenderer,
) -> Result<()> {
    match extension(&args.output).as_deref() {
        Some("jpg") | Some("jpeg") | Some("png") => {}
        _ => bail!("an image source requires a .jpg, .jpeg, or .png output"),
    }
    let image = open_image(&args.source)?;
    println!("Image: {}x{}", image.width(), image.height());
    let detections = inferencer.infer(&image)?;
    print_detections(&detections, args.kpt_conf);
    let rgba = image.to_rgba8();
    let annotated = renderer.render(
        rgba.as_raw(),
        image.width(),
        image.height(),
        &detections,
        args.kpt_conf,
    )?;
    let rendered = RgbaImage::from_raw(image.width(), image.height(), annotated)
        .context("GPU renderer returned an invalid RGBA frame")?;
    match extension(&args.output).as_deref() {
        Some("jpg") | Some("jpeg") => DynamicImage::ImageRgba8(rendered)
            .to_rgb8()
            .save(&args.output)?,
        Some("png") => rendered.save(&args.output)?,
        _ => unreachable!("validated before rendering"),
    }
    println!("Wrote annotated image to {}", args.output.display());
    Ok(())
}

fn print_detections(detections: &[types::PoseDetection], kpt_conf: f32) {
    println!("Detected {} person(s):", detections.len());
    for (i, det) in detections.iter().enumerate() {
        let b = &det.bbox;
        println!(
            "  [{i}] class={} conf={:.2}  bbox=({:.0},{:.0})-({:.0},{:.0})",
            b.class_id, b.conf, b.x1, b.y1, b.x2, b.y2
        );
        for (k, point) in det
            .keypoints
            .iter()
            .enumerate()
            .filter(|(_, point)| point.conf >= kpt_conf)
        {
            println!(
                "       {:>15}  x={:6.1}  y={:6.1}  vis={:.2}",
                COCO_KPT_NAMES[k], point.x, point.y, point.conf
            );
        }
    }
}

fn extension(path: &std::path::Path) -> Option<String> {
    path.extension()?
        .to_str()
        .map(|value| value.to_ascii_lowercase())
}
