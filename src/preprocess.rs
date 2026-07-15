//! Letterbox resize + CHW f32 normalization — mirrors ultralytics LetterBox.
//!
//! Algorithm (from ultralytics/data/augment.py LetterBox):
//!   scale = min(target_h / src_h, target_w / src_w)
//!   new_w = round(src_w * scale),  new_h = round(src_h * scale)
//!   center-pad the remainder with pixel value 114
//!   convert RGB → CHW f32 / 255.0

use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};

/// Output of the preprocessor, needed to unscale coordinates afterwards.
pub struct LetterboxInfo {
    /// The scale applied: new_side / original_side  (same for both axes).
    pub scale: f32,
    /// Left-padding in letterboxed pixels.
    pub pad_left: u32,
    /// Top-padding in letterboxed pixels.
    pub pad_top: u32,
    pub orig_w: u32,
    pub orig_h: u32,
}

/// Resize `img` into a `model_size × model_size` square with centered padding
/// using pixel value 114. Returns a `Vec<f32>` in NCHW layout (batch=1) and
/// the [`LetterboxInfo`] needed to map detections back to the original image.
pub fn letterbox_to_tensor(img: &DynamicImage, model_size: u32) -> (Vec<f32>, LetterboxInfo) {
    let (src_w, src_h) = img.dimensions();

    // 1. Compute scale while preserving aspect ratio.
    let scale = f32::min(
        model_size as f32 / src_w as f32,
        model_size as f32 / src_h as f32,
    );
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;

    // 2. Resize to (new_w, new_h) using bilinear interpolation.
    let resized = img
        .resize_exact(new_w, new_h, FilterType::Triangle)
        .to_rgb8();

    // 3. Create a (model_size × model_size) canvas filled with 114 (grey).
    let mut canvas = RgbImage::from_pixel(model_size, model_size, image::Rgb([114u8, 114, 114]));

    // Match Ultralytics LetterBox(center=True), including its rounding bias.
    let pad_left = (((model_size - new_w) as f32 / 2.0) - 0.1).round() as u32;
    let pad_top = (((model_size - new_h) as f32 / 2.0) - 0.1).round() as u32;

    // 4. Copy resized image into the centered letterbox region.
    for y in 0..new_h {
        for x in 0..new_w {
            canvas.put_pixel(x + pad_left, y + pad_top, *resized.get_pixel(x, y));
        }
    }

    // 5. HWC u8 → CHW f32 / 255  (RGB already, ORT model expects RGB CHW)
    let area = (model_size * model_size) as usize;
    let mut blob = vec![0f32; 3 * area];
    for y in 0..model_size as usize {
        for x in 0..model_size as usize {
            let px = canvas.get_pixel(x as u32, y as u32);
            let idx = y * model_size as usize + x;
            blob[0 * area + idx] = px[0] as f32 / 255.0; // R
            blob[1 * area + idx] = px[1] as f32 / 255.0; // G
            blob[2 * area + idx] = px[2] as f32 / 255.0; // B
        }
    }

    let info = LetterboxInfo {
        scale,
        pad_left,
        pad_top,
        orig_w: src_w,
        orig_h: src_h,
    };

    (blob, info)
}

/// Convert a borrowed RGBA frame without cloning its full-resolution buffer.
pub fn letterbox_rgba_to_tensor(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    model_size: u32,
) -> (Vec<f32>, LetterboxInfo) {
    let frame = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(src_w, src_h, rgba)
        .expect("caller must provide exactly width * height * 4 RGBA bytes");
    let scale = f32::min(
        model_size as f32 / src_w as f32,
        model_size as f32 / src_h as f32,
    );
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;
    let resized = image::imageops::resize(&frame, new_w, new_h, FilterType::Triangle);
    let pad_left = (((model_size - new_w) as f32 / 2.0) - 0.1).round() as u32;
    let pad_top = (((model_size - new_h) as f32 / 2.0) - 0.1).round() as u32;
    let area = (model_size * model_size) as usize;
    let mut blob = vec![114.0 / 255.0; 3 * area];
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let index = (y + pad_top as usize) * model_size as usize + x + pad_left as usize;
            blob[index] = pixel[0] as f32 / 255.0;
            blob[area + index] = pixel[1] as f32 / 255.0;
            blob[2 * area + index] = pixel[2] as f32 / 255.0;
        }
    }
    (
        blob,
        LetterboxInfo {
            scale,
            pad_left,
            pad_top,
            orig_w: src_w,
            orig_h: src_h,
        },
    )
}
