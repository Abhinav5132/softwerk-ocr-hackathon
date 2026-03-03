use candle_core::{DType, Device, Tensor};
use candle_transformers::{
    generation::LogitsProcessor,
    models::moondream::{self, Model}
};
use image::DynamicImage;
use tokenizers::Tokenizer;
use anyhow::Result;
use anyhow::Error as E;
use std::io::Write;
use candle_nn::VarBuilder;
use crate::get_dtype;

/*adapted from https://github.com/huggingface/candle/blob/main/candle-examples/examples/moondream/main.rs#L35 */
struct TextGeneration {
    model: moondream::Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: moondream::Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, image_embeds: &Tensor, sample_len: usize) -> Result<()> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the Moondream model.")
        }

        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;

        // Moondream tokenizer bos_token and eos_token is "<|endoftext|>"
        // https://huggingface.co/vikhyatk/moondream2/blob/main/special_tokens_map.json
        let special_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the special token"),
        };
        let (bos_token, eos_token) = (special_token, special_token);

        let start_gen = std::time::Instant::now();
        let mut load_t = std::time::Duration::from_secs_f64(0f64);
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = if index > 0 {
                self.model.text_model.forward(&input)?
            } else {
                let bos_token = Tensor::new(&[bos_token], &self.device)?.unsqueeze(0)?;
                let logits = self.model
                        .text_model
                        .forward_with_img(&bos_token, &input, image_embeds)?;
                load_t = start_gen.elapsed();
                println!("load_t: {load_t:?}");
                logits
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || tokens.ends_with(&[27, 10619, 29] /* <END> */) {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            print!("{token}");
            std::io::stdout().flush()?;
        }

        let dt = start_gen.elapsed() - load_t;
        println!(
            "\ngenerated in {} seconds\n{generated_tokens} tokens generated ({:.2} token/s)",
            dt.as_secs_f64(),
            (generated_tokens - 1) as f64 / dt.as_secs_f64()
        );

        Ok(())
    }

}

/*Convert imaage into a tensor with shape (3, 378, 378) */
pub fn convert_image(image: DynamicImage, device: &Device) -> Result<Tensor>{
    let img = image.resize_to_fill(378, 378, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (378, 378, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], device)?.reshape((3, 1, 1))?;

    Ok((data.to_dtype(DType::F32)? / 255.0)?.broadcast_sub(&mean)?.broadcast_div(&std)?)
} 

pub fn run_moondream(device: &Device, image: DynamicImage, context: String) -> Result<()> {
    let seed = 1337;
    let temp= Some(0.00);
    let top_p = None; // try 0.9 later
    let repeat_penalty = 1.1;
    let repeat_last_n = 64;

    let tokenizer = Tokenizer::from_file("models/moondream/tokenizer.json")
    .map_err(E::msg).expect("Failed to load tokenizer for image model");

    let config = moondream::Config::v2();
    let dtype = get_dtype(device);

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&["models/moondream/model.safetensors"], dtype, device)?
    };

    let model = moondream::Model::new(&config, vb.pp("Model"))?;

    let mut pipeline = TextGeneration::new(
        model, 
        tokenizer, 
        seed, 
        temp, 
        top_p, 
        repeat_penalty, 
        repeat_last_n, 
        device
    );

    let image = convert_image(image, device)?.to_device(device)?.to_dtype(dtype)?;

    let image_embeds = image.unsqueeze(0)?;
    let image_embeds = image_embeds.apply(pipeline.model.vision_encoder())?;

    let prompt = format!("
    \n\nQuestion: Describe this image in Swedish.
     It appears in a document with the following surrounding text: {}\n\nAnswer:", context);
    pipeline.run(&prompt, &image_embeds, 250usize)
    

}