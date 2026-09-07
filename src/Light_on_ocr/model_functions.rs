use std::io::Write;

use candle_core::{DType, Device, safetensors::load};
use candle_nn::VarBuilder;
use regex::Regex;
use tokenizers::Tokenizer;

use crate::{
    Light_on_ocr::{config_structs::ModelConfig, model::LightOnOCR, preprocess::preprocess},
    get_dtype,
    page_struct::{ImageCoordinates, ImageDimentions, Page, UnprocessedOutput},
};
use anyhow::{Context, Result};

const IM_START:   u32 = 151644; // <|im_start|>
const IM_END:     u32 = 151645; // <|im_end|> — EOS token
const IMAGE_PAD:  u32 = 151655; // <|image_pad|> — one image patch token

pub fn build_model(device: &Device) -> Result<(LightOnOCR, Tokenizer)> {
    
    let config_path    = "models/LightOnOCR/config.json";
    let weights_path   = "models/LightOnOCR/model.safetensors";
    let tokenizer_path = "models/LightOnOCR/tokenizer.json";

    let config_str = std::fs::read_to_string(config_path)?;
    let model_config: ModelConfig = serde_json::from_str(&config_str)?;

    let dtype = get_dtype(device);

    println!("Loading weights");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
    }?;

    println!("Building model");
    let mut model = LightOnOCR::new(&model_config, vb)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;

    Ok((model, tokenizer))
}


/*image path is hardcoded for now should later use the pages vector. */
pub fn run_model(mut model: LightOnOCR, tokenizer: Tokenizer, device: &Device, pages: Vec<Page>) -> Result<Vec<UnprocessedOutput>> {
    let image_regex = Regex::new(r"!\[image\]\(image_(\d+)\.png\)\s*(\d+),(\d+),(\d+),(\d+)")
    .expect("Failed to generate image extraction regex");

    let mut unpocessed_output: Vec<UnprocessedOutput> = vec![];
    for page in pages.iter(){
        model.clear_kv_cache();
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

        let mut image_tokens: Vec<u32> = vec![IMAGE_PAD; num_image_tokens];
        let mut input_ids: Vec<u32> = Vec::new();

        input_ids.push(IM_START);
        input_ids.extend_from_slice(&system_tokens);
        input_ids.push(IM_END);
        input_ids.extend_from_slice(&newline_tokens);

        input_ids.push(IM_START);
        input_ids.extend_from_slice(&user_tokens);
        input_ids.extend_from_slice(&image_tokens);
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
        let logits = model.forward(&input_tensor, &preprocessed.pixel_values, 0).context("MODEL FORWARD FAIL")?;
        println!("logits shape: {:?}", logits.shape());

        let mut generated: Vec<u32> = Vec::new();
        let mut offset = seq_len;

        let (first_token, first_confidence) = greedy(&logits)?;
        generated.push(first_token);
        let mut token_confidences: Vec<f32> = vec![];
        if first_token != IM_END {
            token_confidences.push(first_confidence);
        }
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

            let logits = model.decode_step(&input, offset).context("DECODE STEP ERROR")?;
            let (token, confidence) = greedy(&logits)?;
            generated.push(token);
            if token != IM_END {
                token_confidences.push(confidence);
            }
            offset += 1;
        }

        let lighton_confidence = if token_confidences.is_empty() {
            0.0
        } else {
            token_confidences.iter().sum::<f32>() / token_confidences.len() as f32
        };

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
        
        unpocessed_output.push(UnprocessedOutput { 
            page: page.clone(), 
            loaded_image: img,
            image_dimentions, 
            unprocessed_output: 
            output, 
            image_regions, 
            is_handwritten: false,
            lighton_confidence
        }
        );
    }
    Ok(unpocessed_output)
}


fn greedy(logits: &candle_core::Tensor) -> Result<(u32, f32)> {
    let logits = logits.squeeze(0)?;
    let seq = logits.dim(0)?;
    let last = logits.narrow(0, seq - 1, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
    let logits_vec = last.to_vec1::<f32>()?;

    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, value) in logits_vec.iter().enumerate() {
        if *value > max_val {
            max_val = *value;
            max_idx = idx;
        }
    }

    let mut exp_sum = 0f32;
    for value in &logits_vec {
        exp_sum += (*value - max_val).exp();
    }
    let top_probability = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };

    Ok((max_idx as u32, top_probability))
}

pub fn print_safetensors() -> Result<()> {
    let tensor1 = "models/LightOnOCR/model.safetensors";
    
    let tensors = load(tensor1, &candle_core::Device::Cpu)?;

    let mut out = std::fs::File::create("Keys.txt")?;
    
    for (name, tensor) in &tensors{
        writeln!(out, "{}\t{:?}\t{:?}", name, tensor.shape(), tensor.dtype())?;
    }

    Ok(())
}

pub fn get_image_regions(output: &str, regex: &Regex, img_dimentions: &ImageDimentions) -> Vec<ImageCoordinates>{
    let mut image_regions = vec![];
    for imgs in regex.captures_iter(output){
        let full = imgs[0].to_string();
        let x1_norm= imgs[2].parse().unwrap_or(0.0);
        let y1_norm= imgs[3].parse().unwrap_or(0.0);
        let x2_norm= imgs[4].parse().unwrap_or(0.0);
        let y2_norm= imgs[5].parse().unwrap_or(0.0);

        let x1 = ((x1_norm / 1000.0)*img_dimentions.img_w as f32) as u32; 
        let y1 = ((y1_norm / 1000.0)*img_dimentions.img_h as f32) as u32;
        let x2 = ((x2_norm / 1000.0)*img_dimentions.img_w as f32) as u32;
        let y2 = ((y2_norm / 1000.0)*img_dimentions.img_h as f32) as u32;

        image_regions.push(ImageCoordinates{
            lable: full,
            x1,
            y1,
            x2,
            y2,
        });
    }
    image_regions
}