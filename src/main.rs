use std::time;
use std::fs;
use candle_core::{DType, Device};

pub mod Light_on_ocr;
pub mod trocr;
pub mod moondream;
pub mod pdf;
use crate::pdf::convert_pdf_to_image;

mod page_struct;
use crate::page_struct::Page;
#[cfg(feature = "opencv")]
use crate::page_struct::ImageCoordinates;

#[cfg(feature = "opencv")]
const HANDWRITING_CONFIDENCE_THRESHOLD: f32 = 0.28;


//TODO: IF AN IMAGE HAS HANDWRITING USE THE HANDWRITTEN MODEL, IF NOT USE THE OTHER MODEL. AND IMPROVE THE OUTPUT OF THE HANDWRITTEN MODEL. BY IMPROVEING LINE SEGEMENTATION AND IMAGE PREPROCESSING (THIKEN the characters).
fn main() {
    let start_time = time::Instant::now();
    let device = select_device();
    #[cfg(feature = "opencv")]
    let dtype = candle_core::DType::F32;
    let mut unprocessed_outputs = vec![];
    let mut pages = load_pages_from_images_dir();

    if pages.is_empty() {
        match convert_pdf_to_image() {
            Ok(_) => {
                pages = load_pages_from_images_dir();
            }
            Err(e) => {
                dbg!(e);
                println!("Failed to convert PDFs to images");
            }
        }
    }

    if pages.is_empty() {
        println!("No pages found in data/images");
        return;
    }
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

    #[cfg(feature = "opencv")]
    {
        match trocr::TrocrSwedishHandwritten::build_handwritten_trocr(&device, dtype) {
            Ok(mut trocr_model) => {
                for output in unprocessed_outputs.iter_mut() {
                    if should_use_handwritten_output(output.lighton_confidence, &output.unprocessed_output) {
                        match trocr_model.transcribe_page(&output.page.path, &device, dtype) {
                            Ok(handwritten_text) => {
                                let merged_text = merge_handwritten_with_placeholders(
                                    &handwritten_text,
                                    &output.image_regions,
                                );
                                if !merged_text.trim().is_empty() {
                                    output.unprocessed_output = merged_text;
                                }
                                output.is_handwritten = true;
                            }
                            Err(e) => {
                                dbg!(e);
                                println!(
                                    "Failed handwritten transcription for page: {}",
                                    output.page.path
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                dbg!(e);
                println!("Failed to build TroCR handwritten model");
            }
        }
    }

    #[cfg(not(feature = "opencv"))]
    {
        println!(
            "OpenCV feature is disabled; skipping handwritten routing. Build with --features opencv to enable it."
        );
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

fn load_pages_from_images_dir() -> Vec<Page> {
    let mut pages = vec![];

    let entries = match fs::read_dir("data/images") {
        Ok(entries) => entries,
        Err(_) => return pages,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let extension = match path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) => ext.to_ascii_lowercase(),
            None => continue,
        };

        if extension != "png" && extension != "jpg" && extension != "jpeg" {
            continue;
        }

        let page_path = path.to_string_lossy().to_string();
        let page_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        pages.push(Page {
            path: page_path,
            name: page_name,
        });
    }

    pages.sort_by(|left, right| left.path.cmp(&right.path));
    pages
}

#[cfg(feature = "opencv")]
fn should_use_handwritten_output(confidence: f32, text: &str) -> bool {
    let non_ws_chars = text.chars().filter(|c| !c.is_whitespace()).count();
    if non_ws_chars == 0 {
        return true;
    }

    let alphabetic_chars = text
        .chars()
        .filter(|c| !c.is_whitespace() && c.is_alphabetic())
        .count();
    let alpha_ratio = alphabetic_chars as f32 / non_ws_chars as f32;

    let handwriting_signal = 0.8 * (1.0 - confidence) + 0.2 * (1.0 - alpha_ratio);

    handwriting_signal > 0.72 || confidence < HANDWRITING_CONFIDENCE_THRESHOLD
}

#[cfg(feature = "opencv")]
fn merge_handwritten_with_placeholders(
    handwritten_text: &str,
    image_regions: &[ImageCoordinates],
) -> String {
    let mut merged = handwritten_text.trim().to_string();

    for region in image_regions {
        if !merged.contains(&region.lable) {
            if !merged.is_empty() {
                merged.push('\n');
            }
            merged.push_str(&region.lable);
        }
    }

    merged
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