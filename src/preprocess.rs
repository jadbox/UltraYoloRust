//! Letterbox resize + CHW f32 normalization — mirrors ultralytics LetterBox.
//!
//! Algorithm (from ultralytics/data/augment.py LetterBox):
//!   scale = min(target_h / src_h, target_w / src_w)
//!   new_w = round(src_w * scale),  new_h = round(src_h * scale)
//!   center-pad the remainder with pixel value 114
//!   convert RGB → CHW f32 / 255.0


use fast_image_resize as fr;
use image::{DynamicImage, GenericImageView};

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

    // 2. Resize using fast_image_resize
    let mut raw_rgba = img.to_rgba8();
    let src_image = fr::images::Image::from_slice_u8(
        src_w,
        src_h,
        raw_rgba.as_mut(),
        fr::PixelType::U8x4,
    ).unwrap();

    let mut dst_image = fr::images::Image::new(
        new_w,
        new_h,
        fr::PixelType::U8x4,
    );

    let mut resizer = fr::Resizer::new();
    resizer.resize(&src_image, &mut dst_image, None).unwrap();
    let resized_raw = dst_image.buffer();

    // Match Ultralytics LetterBox(center=True), including its rounding bias.
    let pad_left = (((model_size - new_w) as f32 / 2.0) - 0.1).round() as u32;
    let pad_top = (((model_size - new_h) as f32 / 2.0) - 0.1).round() as u32;

    let area = (model_size * model_size) as usize;
    let mut blob = vec![114.0 / 255.0; 3 * area];

    let inv_255 = 1.0 / 255.0;
    let pad_left = pad_left as usize;
    let pad_top = pad_top as usize;
    let model_size = model_size as usize;

    for y in 0..new_h as usize {
        let row_offset = (y + pad_top) * model_size + pad_left;
        let raw_row_offset = y * (new_w as usize) * 4;
        for x in 0..new_w as usize {
            let px_idx = raw_row_offset + x * 4;
            let blob_idx = row_offset + x;
            blob[blob_idx] = resized_raw[px_idx] as f32 * inv_255;
            blob[area + blob_idx] = resized_raw[px_idx + 1] as f32 * inv_255;
            blob[2 * area + blob_idx] = resized_raw[px_idx + 2] as f32 * inv_255;
        }
    }

    let info = LetterboxInfo {
        scale,
        pad_left: pad_left as u32,
        pad_top: pad_top as u32,
        orig_w: src_w,
        orig_h: src_h,
    };

    (blob, info)
}

/// Convert a borrowed RGBA frame without cloning its full-resolution buffer.
pub fn letterbox_rgba_to_tensor(
    rgba: &mut [u8],
    src_w: u32,
    src_h: u32,
    model_size: u32,
) -> (Vec<f32>, LetterboxInfo) {
    let scale = f32::min(
        model_size as f32 / src_w as f32,
        model_size as f32 / src_h as f32,
    );
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;

    let src_image = fr::images::Image::from_slice_u8(
        src_w,
        src_h,
        rgba,
        fr::PixelType::U8x4,
    ).unwrap();

    let mut dst_image = fr::images::Image::new(
        new_w,
        new_h,
        fr::PixelType::U8x4,
    );

    let mut resizer = fr::Resizer::new();
    resizer.resize(&src_image, &mut dst_image, None).unwrap();
    let resized_raw = dst_image.buffer();

    let pad_left = (((model_size - new_w) as f32 / 2.0) - 0.1).round() as u32;
    let pad_top = (((model_size - new_h) as f32 / 2.0) - 0.1).round() as u32;
    let area = (model_size * model_size) as usize;
    let mut blob = vec![114.0 / 255.0; 3 * area];

    let inv_255 = 1.0 / 255.0;
    let pad_left = pad_left as usize;
    let pad_top = pad_top as usize;
    let model_size = model_size as usize;

    for y in 0..new_h as usize {
        let row_offset = (y + pad_top) * model_size + pad_left;
        let raw_row_offset = y * (new_w as usize) * 4;
        for x in 0..new_w as usize {
            let px_idx = raw_row_offset + x * 4;
            let blob_idx = row_offset + x;
            blob[blob_idx] = resized_raw[px_idx] as f32 * inv_255;
            blob[area + blob_idx] = resized_raw[px_idx + 1] as f32 * inv_255;
            blob[2 * area + blob_idx] = resized_raw[px_idx + 2] as f32 * inv_255;
        }
    }

    (
        blob,
        LetterboxInfo {
            scale,
            pad_left: pad_left as u32,
            pad_top: pad_top as u32,
            orig_w: src_w,
            orig_h: src_h,
        },
    )
}
