use image::{DynamicImage, GrayImage};
use opencv::{core::{Rect, Vector}, imgcodecs, imgproc, prelude::*};
use opencv::core::{Size, Point, Scalar};

pub fn line_segemenation(page: &str) -> opencv::Result<Vec<String>> {

    let img = imgcodecs::imread(page, imgcodecs::IMREAD_COLOR)?;
    //convert to grayscale
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0, opencv::core::AlgorithmHint::ALGO_HINT_APPROX)?;


    // Adaptive threeshold 
    let mut bin = Mat::default();
    imgproc::adaptive_threshold(&gray, &mut bin, 
        255.0, 
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C, 
        imgproc::THRESH_BINARY_INV, 
        41, 
        5.0
    )?;

    //horizontal projection profile 
    let rows = bin.rows();
    let cols = bin.cols();

    let mut histogram = vec![0; rows as usize];

    for y in 0..rows{
        let mut sum = 0;
        for x in 0..cols {
            let val = *bin.at_2d::<u8>(y, x)?; // get the pixel value at y and x
            if val > 0 { // if pixel value not zero add the sum
                sum += 1;
            }
        }
        histogram[y as usize] = sum;
    }
    
    let window_size = 3;
    let mut smoothed_histogram = histogram.clone();
    for i in window_size..histogram.len() - window_size {
        let sum: i32 = histogram[i - window_size..=i + window_size].iter().sum();
        smoothed_histogram[i] = sum / (2 * window_size as i32 + 1);
    }

    //detect line ranges
    let mut lines = Vec::new();
    let mut start:Option<i32> = None;
    let threshold = (cols as f32 * 0.01) as i32;

    for (y, &v) in smoothed_histogram.iter().enumerate(){
        if v > threshold && start.is_none(){
            start = Some(y as i32);
        } else if v <= threshold && start.is_some() {
            lines.push((start.expect("Failed to get image line start"), y as i32));
            start = None;
        }
    }

    if let Some(s) = start {
        lines.push((s, rows));
    }   

    let mut image_names: Vec<String> = vec![];
    for (i, (y1, y2)) in lines.iter().enumerate() {
        let rect = Rect::new(0, *y1, cols, y2 - y1); 
        let roi = Mat::roi(&img, rect)?;
        let name = format!("data/handwritten/line_{}.png", i);
        image_names.push(name.clone());
        imgcodecs::imwrite(&name, &roi, &opencv::core::Vector::new())?;
    }   

    Ok(image_names)

}