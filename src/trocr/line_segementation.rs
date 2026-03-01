use image::{DynamicImage, GrayImage};
use opencv::{core::{Rect, Vector}, imgcodecs, imgproc, prelude::*};

pub fn line_segemenation(page: &str) -> opencv::Result<()> {

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
        31, 
        15.0
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

    //detect line ranges
    let mut lines = Vec::new();
    let mut start:Option<i32> = None;
    let threshold = 10;

    for (y, &v) in histogram.iter().enumerate(){
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

    for (i, (y1, y2)) in lines.iter().enumerate() {
        let rect = Rect::new(0, *y1, cols, y2 - y1); 
        let roi = Mat::roi(&img, rect)?;
        imgcodecs::imwrite(&format!("line_{}.png", i), &roi, &opencv::core::Vector::new())?;
    }

    Ok(())

}