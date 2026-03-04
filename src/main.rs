use std::time;
use std::io::Write;
use candle_nn::VarBuilder;
use candle_core::{DType, Device, safetensors::*};
use regex::Regex;
use tokenizers::Tokenizer;

pub mod Light_on_ocr;
pub mod trocr;
pub mod moondream;
pub mod pdf;
use crate::pdf::convert_pdf_to_image;

mod page_struct;
use crate::page_struct::{ImageCoordinates, Page};
use crate::Light_on_ocr::preprocess::preprocess;

fn main() {
    let start_time = time::Instant::now();
    let device = select_device();
    let dtype = DType::F32;
    /* 
        let _ = convert_pdf_to_image(); 
    */
    let mut unprocessed_outputs = vec![];
    // Build the LightOnOcr and get the unprocessed output from it. 
    let mut pages = vec![]; // Empty vector for now add actual loading later,

    //TODO dont manually load one page load all pages in data/converted
    pages.push(
        Page { 
            path: "data/images/pol_1994_03_24_SÄPO_PM_Swedenborgskyrkan_HE_15241_02_pdf_page_3.png".to_string(), 
            name: "test".to_string() 
        }
    );
    // the { } ensure the model goes out of memory once its finished transcribing. We dont want multiple models loaded in memeory at the same time
    {
        if let Ok((model, tokenizer)) = Light_on_ocr::model_functions::build_model(&device){
            match Light_on_ocr::model_functions::run_model(model, tokenizer, &device, pages) {
                Ok(output) => {
                    println!("Transcription finished successfully");
                    unprocessed_outputs = output;
                }

                Err(e) => {
                    dbg!(e);
                    println!("Failed to transcribe. Please fix the error above.")
                }
            }
        }
        else {
            println!("Failed to build model");
        }
    }

    let mut processed_outputs = vec![];
    {
        if let Ok(pipeline) = moondream::build_model(&device){
            match moondream::run_moondream(pipeline, &device, unprocessed_outputs){
                Ok(output) => {
                    println!("Description finished succesfully");
                    processed_outputs = output
                }
                Err(e) => {
                    dbg!(e);
                    println!("failed to describe images please fix the above error.")
                }
            }
        } else {
            println!("Failed to build moondream model")
        }
    }
    let elapsed = start_time.elapsed().as_secs();
    println!("{elapsed}");
    


    /*
    let mut images = vec![];

    let image_path = "data/images/akl-2017-02-27-AM-2017-1099-SA-Brev-till-KP.pdf-10.png";
    let image_names = line_segemenation(image_path).unwrap();

    for img_path in image_names {
        let image = image::ImageReader::open(img_path).unwrap().decode().unwrap();
        images.push(image);

    }
    
    let trocr_swedish_model = 
    TrocrSwedishHandwritten::build_handwritten_trocr(&device, dtype).unwrap();
    
    match trocr_swedish_model.run_handwritten_trocr(images, &device, dtype){
        Ok(_) => {
            println!("Transcription finished succesfully");
        }
        Err(e) => {
            dbg!(e);
        }
    }*/

    

}

/*Selects the device to you, falls back to cpu if no CUDA or METAL devices found */
pub fn select_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            let cuda = match Device::new_cuda(0) {
                Ok(c) => c,
                Err(e) => {
                    dbg!(e);
                    Device::Cpu
                }
            };
            return cuda;
        }
    }

    #[cfg(feature = "metal")]
    {
        if candle_core::utils::metal_is_available() {
            let metal = match Device::new_metal(0) {
                Ok(m) => m,
                Err(e) => {
                    dbg!(e);
                    Device::Cpu
                }
            };
            return metal;
        }
    }

    Device::Cpu
}

pub fn get_dtype(device: &Device) -> DType{
    let _ = device;
    DType::BF16
}