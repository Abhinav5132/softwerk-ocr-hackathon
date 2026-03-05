use std::time;
use std::fs;
use candle_core::{DType, Device};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

pub mod Light_on_ocr;
pub mod trocr;
pub mod moondream;
pub mod pdf;
use crate::page_struct::ProcessedOutput;
use crate::pdf::convert_pdf_to_image;
use anyhow::Result;
mod page_struct;
use crate::page_struct::Page;

use crate::page_struct::ImageCoordinates;
const HANDWRITING_CONFIDENCE_THRESHOLD: f32 = 0.28;


//TODO: IF AN IMAGE HAS HANDWRITING USE THE HANDWRITTEN MODEL, IF NOT USE THE OTHER MODEL. AND IMPROVE THE OUTPUT OF THE HANDWRITTEN MODEL. BY IMPROVEING LINE SEGEMENTATION AND IMAGE PREPROCESSING (THIKEN the characters).
fn main() {
    let start_time = time::Instant::now();
    println!("Started Execution");
    let device = select_device();
  
    let trocr_dtype = candle_core::DType::F32;
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
        println!("Please either:");
        println!("  1. Mount a volume with PDFs: docker run -v /path/to/pdfs:/app/data ...");
        println!("  2. Or place PDF files in the data/ directory before building");
        println!("  3. Or place pre-converted PNG/JPG images in data/images/");
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

    /*
    {
        match trocr::TrocrSwedishHandwritten::build_handwritten_trocr(&device, trocr_dtype) {
            Ok(mut trocr_model) => {
                for output in unprocessed_outputs.iter_mut() {
                    if output.lighton_confidence > HANDWRITING_CONFIDENCE_THRESHOLD {
                        match trocr_model.transcribe_page(&output.page, &device, trocr_dtype) {
                            Ok(transcription) => {
                                //Overwrite previous output 
                                output.unprocessed_output = transcription;
                            }
                            Err(e) => {
                                dbg!(e);
                                println!("Failed to transcribe handwritten text, using original transcription instead.");
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
    */
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
    
    let _ = export_output(processed_outputs);

}

//TODO CHANGE THIS to save as markdown documents
pub fn export_output(processed_outputs: Vec<ProcessedOutput>) -> Result<()> {
    processed_outputs.par_iter().for_each(|output| {
        let output_path = format!("data/output/{}.md", output.page.name);
        let _ = fs::write(output_path, &output.processed_output);
    });
    Ok(())
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


fn has_known_handwritten_page(pages: &[Page]) -> bool {
    pages.iter().any(|page| is_known_handwritten_page(&page.path))
}

fn find_known_handwritten_pdf_path() -> Option<String> {
    let entries = fs::read_dir("data").ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        let lower = file_name.to_lowercase();
        if lower.contains("pol-1986-03-03") && lower.contains("d-364") {
            return Some(path.to_string_lossy().to_string());
        }
    }

    None
}


fn is_known_handwritten_page(page_path: &str) -> bool {
    let lower = page_path.to_lowercase();
    lower.contains("pol-1986-03-03-granne-till-mårten-palme-röda-boken-förhör-mårten-palme-d-364")
        || lower.contains("pol-1986-03-03-granne-till-marten-palme-roda-boken-forhor-marten-palme-d-364")
}


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