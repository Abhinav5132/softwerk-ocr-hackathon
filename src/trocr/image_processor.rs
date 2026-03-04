use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use image::DynamicImage;
use anyhow::Result;

pub struct PreProcessorConfig {
    do_resize: bool,
    height: u32,
    width: u32,
    do_rescale: bool,
    do_normalize: bool,
    image_mean: Vec<f32>,
    image_std: Vec<f32>
}

impl Default for PreProcessorConfig {
    fn default() -> Self {
        Self { do_resize: true,
            height: 384, 
            width: 384, 
            do_rescale: true,
            do_normalize: true, 
            image_mean: vec![0.5,0.5,0.5], 
            image_std: vec![0.5,0.5,0.5] 
        }
    }
}

pub struct VITImageProcessor {
    do_resize: bool,
    height: u32,
    width:u32,
    do_normalize: bool,
    image_mean: Vec<f32>,
    image_std: Vec<f32>
}

impl VITImageProcessor {
    pub fn new(config: PreProcessorConfig) -> Self {
        Self { do_resize: config.do_resize, 
            height: config.height, 
            width: config.width, 
            do_normalize: config.do_normalize, 
            image_mean: config.image_mean,
            image_std: config.image_std 
        }
    }


    pub fn preprocess(&self, mut images: Vec<image::DynamicImage>, device: &Device, dtype: DType) -> Result<Tensor> {
        let resized_images:Vec<DynamicImage> = if self.do_resize {
            images.iter_mut()
            .map(|image| self.resize(image.clone()))
            .collect()
        } else {
            images
        };

        let mut normalized_images: Vec<Tensor> = vec![];
        for img in resized_images{
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