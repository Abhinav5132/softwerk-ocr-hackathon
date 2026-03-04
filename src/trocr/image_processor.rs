use candle_core::{DType, Device, Tensor};
use image::DynamicImage;
use anyhow::Result;

pub struct PreProcessorConfig {
    do_resize: bool,
    height: u32,
    width: u32,
    do_rescale: bool,
    do_thicken: bool,
    do_normalize: bool,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    thicken_threshold: u8,
    thicken_radius: u32,
}

impl Default for PreProcessorConfig {
    fn default() -> Self {
        Self { do_resize: true,
            height: 384, 
            width: 384, 
            do_rescale: true,
            do_thicken: true,
            do_normalize: true, 
            image_mean: vec![0.5,0.5,0.5], 
            image_std: vec![0.5,0.5,0.5],
            thicken_threshold: 170,
            thicken_radius: 1,
        }
    }
}

pub struct VITImageProcessor {
    do_resize: bool,
    height: u32,
    width:u32,
    do_thicken: bool,
    do_normalize: bool,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    thicken_threshold: u8,
    thicken_radius: u32,
}

impl VITImageProcessor {
    pub fn new(config: PreProcessorConfig) -> Self {
        Self { do_resize: config.do_resize, 
            height: config.height, 
            width: config.width, 
            do_thicken: config.do_thicken,
            do_normalize: config.do_normalize, 
            image_mean: config.image_mean,
            image_std: config.image_std,
            thicken_threshold: config.thicken_threshold,
            thicken_radius: config.thicken_radius,
        }
    }


    pub fn preprocess(&self, mut images: Vec<image::DynamicImage>, device: &Device, dtype: DType) -> Result<Tensor> {
        let resized_images: Vec<DynamicImage> = if self.do_resize {
            images.iter_mut()
            .map(|image| self.resize(image.clone()))
            .collect()
        } else {
            images
        };

        let enhanced_images: Vec<DynamicImage> = if self.do_thicken {
            resized_images
                .into_iter()
                .map(|image| self.thicken_characters(image))
                .collect()
        } else {
            resized_images
        };

        let mut normalized_images: Vec<Tensor> = vec![];
        for img in enhanced_images{
            let normal_img = self.normalize(img, device, dtype)?;
            normalized_images.push(normal_img);
        }
        Ok(Tensor::stack(&normalized_images, 0)?)
    }

    fn resize(&self, image: image::DynamicImage) -> image::DynamicImage {

        let resized_image = image.resize_exact(
            self.width, 
            self.height, 
            image::imageops::FilterType::Triangle
        );

        resized_image
    }

    fn thicken_characters(&self, image: image::DynamicImage) -> image::DynamicImage {
        let rgb = image.to_rgb8();
        let gray = image::DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
        let (width, height) = gray.dimensions();
        let mut mask = gray.clone();

        for y in 0..height {
            for x in 0..width {
                let pixel = gray.get_pixel(x, y)[0];
                let mut has_ink = pixel < self.thicken_threshold;

                if !has_ink {
                    'neighbors: for dy in -(self.thicken_radius as i32)..=(self.thicken_radius as i32) {
                        for dx in -(self.thicken_radius as i32)..=(self.thicken_radius as i32) {
                            if dx == 0 && dy == 0 {
                                continue;
                            }

                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;

                            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                                continue;
                            }

                            if gray.get_pixel(nx as u32, ny as u32)[0] < self.thicken_threshold {
                                has_ink = true;
                                break 'neighbors;
                            }
                        }
                    }
                }

                mask.get_pixel_mut(x, y)[0] = if has_ink { 0 } else { 255 };
            }
        }

        let mut thickened = rgb;
        for y in 0..height {
            for x in 0..width {
                if mask.get_pixel(x, y)[0] == 0 {
                    thickened.put_pixel(x, y, image::Rgb([0, 0, 0]));
                }
            }
        }

        image::DynamicImage::ImageRgb8(thickened)
    }

    fn normalize(&self, image: image::DynamicImage, device: &Device, dtype: DType) -> Result<Tensor> {

        let mean = Tensor::from_vec(self.image_mean.clone(), (3,1,1), device)?;
        let std = Tensor::from_vec(self.image_std.clone(), (3,1,1), device)?;
        
        let image = image.to_rgb8();
        let data = image.into_raw();

        let height = self.height as usize;
        let width = self.width as usize;

        let channels = 3; 

        let data = Tensor::from_vec(data, &[height, width, channels], device)?
        .permute((2,0,1))?;

        Ok((data.to_dtype(dtype)? / 255.)?.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }
}