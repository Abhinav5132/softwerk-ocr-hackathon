use candle_core::Device;
use crate::{Light_on_ocr::{config_structs::ModelConfig, model::LightOnOCR}, page_struct::{ImageDimentions, UnprocessedOutput}, *};
use anyhow::Result;

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
pub fn run_model(mut model: LightOnOCR, tokenizer: Tokenizer, device: &Device, pages: &mut Vec<Page>) -> Result<()> {
    let image_regex = Regex::new(r"!\[image\]\(image_(\d+)\.png\)\s*(\d+),(\d+),(\d+),(\d+)")
    .expect("Failed to generate image extraction regex");

    for page in pages.iter_mut(){
        model.clear_kv_cache();

        println!("Processing page: {}", page.path);
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

        let input_tensor = candle_core::Tensor::from_vec(
            input_ids,
            (1, seq_len),
            device,
        )?;

        println!("Prefilling...");
        let logits = model.forward(&input_tensor, &preprocessed.pixel_values, 0)?;

        let mut generated: Vec<u32> = Vec::new();
        let mut offset = seq_len;

        let first_token = greedy(&logits)?;
        generated.push(first_token);
        println!("first token id={} decoded={:?}",
            first_token,
            tokenizer.decode(&[first_token], false));

        // Explicitly drop large tensors to free memory
        drop(logits);
        drop(input_tensor);

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
            
            // Explicitly drop logits after each step
            drop(logits);
            drop(input);
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


fn greedy(logits: &candle_core::Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?;
    let seq = logits.dim(0)?;
    let last = logits.narrow(0, seq - 1, 1)?.squeeze(0)?;
    Ok(last.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?)
}

pub fn print_safetensors() -> Result<()> {
    let tensor1 = "models/moondream/model.safetensors";
    
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