use std::{fs, time};
use std::io::Write;
use candle_nn::VarBuilder;
use candle_core::{DType, Device, safetensors::*};
use regex::Regex;
use tokenizers::Tokenizer;

pub mod Light_on_ocr;
pub mod trocr;
pub mod moondream;
pub mod pdf;

mod page_struct;
use crate::page_struct::{ImageCoordinates, Page};
use crate::Light_on_ocr::preprocess::preprocess;
use crate::pdf::get_pdfs_converted_as_images;

fn main() {
    let start_time = time::Instant::now();
    let device = select_device();
    // Build the LightOnOcr and get the unprocessed output from it. 

    println!("Converting pdfs to pngs");
    let mut pages = get_pdfs_converted_as_images();

    // the { } ensure the model goes out of memory once its finished transcribing. We dont want multiple models loaded in memeory at the same time
    {
        if let Ok((model, tokenizer)) = Light_on_ocr::model_functions::build_model(&device){
            match Light_on_ocr::model_functions::run_model(model, tokenizer, &device, &mut pages) {
                Ok(_) => {
                    println!("Transcription finished successfully");
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

    {
        if let Ok(pipeline) = moondream::build_model(&device){
            match moondream::run_moondream(pipeline, &device, &mut pages){
                Ok(_) => {
                    println!("Description finished succesfully");
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

    for page in pages.iter() {
    if let Some(ref processed) = page.processed {
        let output_path = format!(
            "./data/output/{}.txt",
            std::path::Path::new(&page.path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
        );
        fs::create_dir_all("./data/output").expect("failed to create output directory");

        let mut file = fs::File::create(&output_path)
            .expect("failed to create output file");

        writeln!(file, "{}", processed.processed_output)
            .expect("failed to write transcription");

    } else {
        println!("Skipping page with no processed output: {}", page.path);
    }
}
    let elapsed = start_time.elapsed().as_secs();
    println!("{elapsed}");


    /*
    let mut images = vec![];

    let image_path = "data/images/akl-2017-02-27-AM-2017-1099-SA-Brev-till-KP.pdf-10.png";
    let image_names = line_segemenation(image_path).unwrap();
pub fn run_model(mut model: LightOnOCR, tokenizer: Tokenizer, device: &Device, pages: &mut Vec<Page>) -> Result<()> {
    let image_regex = Regex::new(r"!\[image\]\(image_(\d+)\.png\)\s*(\d+),(\d+),(\d+),(\d+)")
    .expect("Failed to generate image extraction regex");

    for page in pages.iter_mut(){
        let image_path = &page.path;
        let img = image::open(image_path)?;
        let image_dimentions = ImageDimentions{
            img_h: img.height(),
            img_w: img.width()
        };
        let preprocessed = preprocess(&img, device)?;

        // Merged patch grid dimensions after 2x2 spatial merge
        let merged_ph = preprocessed.ph / 2;
        let merged_pw = preprocessed.pw / 2;
        let num_image_tokens = merged_ph * merged_pw; // IMAGE_PAD count = 2200

        println!("ph={} pw={} merged_ph={} merged_pw={} num_image_tokens={}",
            preprocessed.ph, preprocessed.pw, merged_ph, merged_pw, num_image_tokens);

        // Encode only plain text — special tokens inserted by id
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("{}", e))?
                .get_ids()
                .to_vec())
        };

        let system_tokens    = encode("system")?;
        let user_tokens      = encode("user\n")?;
        let assistant_tokens = encode("assistant\n")?;
        let newline_tokens   = encode("\n")?;
        let prompt = encode("describe this image\n")?;

        let mut image_tokens: Vec<u32> = vec![IMAGE_PAD; num_image_tokens];

        // Full prompt:
        // <|im_start|>system<|im_end|>\n
        // <|im_start|>user\n[image tokens]<|im_end|>\n
        // <|im_start|>assistant\n
        let mut input_ids: Vec<u32> = Vec::new();

        input_ids.push(IM_START);
        input_ids.extend_from_slice(&system_tokens);
        input_ids.push(IM_END);
        input_ids.extend_from_slice(&newline_tokens);

        input_ids.push(IM_START);
        input_ids.extend_from_slice(&user_tokens);
        input_ids.extend_from_slice(&image_tokens);
        input_ids.extend_from_slice(&prompt);
        input_ids.push(IM_END);
        input_ids.extend_from_slice(&newline_tokens);

        input_ids.push(IM_START);
        input_ids.extend_from_slice(&assistant_tokens);

        let seq_len = input_ids.len();
        println!("Sequence length: {} ({} IMAGE_PAD + {} row tokens)",
            seq_len, num_image_tokens, merged_ph);

        let input_tensor = candle_core::Tensor::from_vec(
            input_ids,
            (1, seq_len),
            device,
        )?;

        println!("Prefilling...");
        let logits = model.forward(&input_tensor, &preprocessed.pixel_values, 0)?;
        println!("logits shape: {:?}", logits.shape());

        let mut generated: Vec<u32> = Vec::new();
        let mut offset = seq_len;

        let first_token = greedy(&logits)?;
        generated.push(first_token);
        println!("first token id={} decoded={:?}",
            first_token,
            tokenizer.decode(&[first_token], false));

        println!("Generating...");
        let max_new_tokens = 1024usize;

        for _ in 1..max_new_tokens {
            let last = *generated.last().unwrap();

            if last == IM_END {
                break;
            }

            let input = candle_core::Tensor::from_vec(
                vec![last],
                (1, 1),
                device,
            )?;

            let logits = model.decode_step(&input, offset)?;
            let token = greedy(&logits)?;
            generated.push(token);
            offset += 1;
        }

        let decode_ids: Vec<u32> = generated.iter()
            .copied()
            .filter(|&t| t != IM_END)
            .collect();

        let output = tokenizer
            .decode(&decode_ids, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        println!("\n=== Output ===");
        println!("{}", output);

        let image_regions = get_image_regions(&output, &image_regex, &image_dimentions);
        
        page.unprocessed = Some(UnprocessedOutput { 
            loaded_image: img,
            image_dimentions, 
            unprocessed_output: 
            output, 
            image_regions, 
            is_handwritten: false // TODO Add a function to determine if its handwritten or not.
        });
    }
    Ok(())
}

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
    if candle_core::utils::cuda_is_available() {
        let cuda = match Device::new_cuda(0) {
            Ok(c) => c,
            Err(e) => {
                dbg!(e);
                Device::Cpu
            }
        };
        cuda
    }
    else if candle_core::utils::metal_is_available() {
        let metal = match Device::new_metal(0) {
            Ok(m) => m,
            Err(e) => {
                dbg!(e);
                Device::Cpu
            }
        };
        metal
    } else {
        Device::Cpu
    }

}

pub fn get_dtype(device: &Device) -> DType{
    let dtype = match device {
        Device::Cpu => DType::F32,
        _ => DType::BF16,
    };
    dtype
}