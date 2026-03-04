use std::fs;

use anyhow::Result;
use image::{DynamicImage, GenericImageView, GrayImage, Luma};

pub fn line_segemenation(page: &str) -> Result<Vec<String>> {
    fs::create_dir_all("data/handwritten")?;

    let original = image::open(page)?;
    let gray = original.to_luma8();
    let binary = adaptive_threshold_like(&gray, 41, 5);
    let merged = dilate_binary(&binary, 1);

    let (width, height) = merged.dimensions();
    let mut histogram = vec![0i32; height as usize];
    for y in 0..height {
        let mut sum = 0i32;
        for x in 0..width {
            if merged.get_pixel(x, y)[0] > 0 {
                sum += 1;
            }
        }
        histogram[y as usize] = sum;
    }

    let smoothed_histogram = smooth_histogram(&histogram, 3);
    let threshold = (width as f32 * 0.008) as i32;
    let min_line_height = 10i32;

    let mut lines = Vec::new();
    let mut start: Option<i32> = None;
    for (y, &value) in smoothed_histogram.iter().enumerate() {
        if value > threshold && start.is_none() {
            start = Some(y as i32);
        } else if value <= threshold && start.is_some() {
            let y_start = start.expect("line start missing");
            if (y as i32 - y_start) >= min_line_height {
                lines.push((y_start, y as i32));
            }
            start = None;
        }
    }
    if let Some(y_start) = start {
        if (height as i32 - y_start) >= min_line_height {
            lines.push((y_start, height as i32));
        }
    }
    lines.sort_by_key(|(y1, _)| *y1);

    let mut image_names = vec![];
    for (index, (y1, y2)) in lines.iter().enumerate() {
        let crop = crop_line(&original, *y1 as u32, *y2 as u32);
        let name = format!("data/handwritten/line_{}.png", index);
        crop.save(&name)?;
        image_names.push(name);
    }

    Ok(image_names)
}

fn adaptive_threshold_like(gray: &GrayImage, window: u32, bias: i32) -> GrayImage {
    let (width, height) = gray.dimensions();
    let radius = (window / 2) as i32;
    let mut out = GrayImage::new(width, height);

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut sum = 0u32;
            let mut count = 0u32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    sum += gray.get_pixel(nx as u32, ny as u32)[0] as u32;
                    count += 1;
                }
            }

            let mean = if count > 0 { (sum / count) as i32 } else { 255 };
            let current = gray.get_pixel(x as u32, y as u32)[0] as i32;
            let is_foreground = current < (mean - bias);
            out.put_pixel(x as u32, y as u32, Luma([if is_foreground { 255 } else { 0 }]));
        }
    }

    out
}

fn dilate_binary(image: &GrayImage, radius: i32) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut out = GrayImage::new(width, height);

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut has_foreground = false;
            'neighbors: for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    if image.get_pixel(nx as u32, ny as u32)[0] > 0 {
                        has_foreground = true;
                        break 'neighbors;
                    }
                }
            }
            out.put_pixel(x as u32, y as u32, Luma([if has_foreground { 255 } else { 0 }]));
        }
    }

    out
}

fn smooth_histogram(histogram: &[i32], window: usize) -> Vec<i32> {
    if histogram.len() < (window * 2 + 1) {
        return histogram.to_vec();
    }

    let mut smoothed = histogram.to_vec();
    for i in window..(histogram.len() - window) {
        let sum: i32 = histogram[(i - window)..=(i + window)].iter().sum();
        smoothed[i] = sum / (2 * window as i32 + 1);
    }
    smoothed
}

fn crop_line(original: &DynamicImage, y1: u32, y2: u32) -> DynamicImage {
    let (width, height) = original.dimensions();
    let top = y1.min(height.saturating_sub(1));
    let bottom = y2.min(height).max(top + 1);
    original.crop_imm(0, top, width, bottom - top)
}