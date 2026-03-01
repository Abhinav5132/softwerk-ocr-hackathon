use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::{generation, models::{trocr::{self, TrOCRConfig}, vit}};
use image::DynamicImage;
use tokenizers::Tokenizer;
use anyhow::Result;
use std::io::Write;

use crate::{trocr::config_structs::ModelConfig};
pub mod config_structs;
pub mod image_processor;
pub mod line_segementation;

pub struct TrocrSwedishHandwritten{
    encoder_config: vit::Config,
    decoder_config: TrOCRConfig,
    model: trocr::TrOCRModel,
}
impl TrocrSwedishHandwritten {
    pub fn build_handwritten_trocr(device: &Device, dtype: DType) 
    -> Result<Self> {
        let weights_path = "models/trocr/model.safetensors";
        let config = TrocrSwedishHandwritten::load_config()?;

        let encoder = config.encoder;
        let encoder_config = vit::Config{
            hidden_size: encoder.hidden_size,
            num_hidden_layers: encoder.num_hidden_layers,
            num_attention_heads: encoder.num_attention_heads,
            intermediate_size: encoder.intermediate_size,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: encoder.layer_norm_eps,
            image_size: encoder.image_size,
            patch_size: encoder.patch_size,
            num_channels: encoder.num_channels,
            qkv_bias: encoder.qkv_bias,
        };

        let decoder = config.decoder;
        let decoder_config = trocr::TrOCRConfig{
            vocab_size: decoder.vocab_size,
            d_model: decoder.d_model,
            cross_attention_hidden_size: decoder.cross_attention_hidden_size,
            decoder_layers: decoder.decoder_layers,
            decoder_attention_heads: decoder.decoder_attention_heads,
            decoder_ffn_dim: decoder.decoder_ffn_dim,
            activation_function: candle_nn::Activation::Relu,
            max_position_embeddings: decoder.max_position_embeddings,
            dropout: decoder.dropout,
            attention_dropout: decoder.attention_dropout,
            activation_dropout: decoder.activation_dropout,
            decoder_start_token_id: decoder.decoder_start_token_id,
            init_std: decoder.init_std,
            decoder_layerdrop: decoder.decoder_layerdrop,
            use_cache: decoder.use_cache,
            scale_embedding: decoder.scale_embedding,
            pad_token_id: decoder.pad_token_id,
            bos_token_id: decoder.bos_token_id,
            eos_token_id: decoder.eos_token_id,
            decoder_vocab_size: Some(decoder.vocab_size),
            use_learned_position_embeddings: decoder.use_learned_position_embeddings,
            tie_word_embeddings: decoder.tie_word_embeddings,
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
        }?;

        println!("Built model");

        let mut model = trocr::TrOCRModel::new(&encoder_config, &decoder_config, vb)?;
        Ok(Self{
            model,
            encoder_config,
            decoder_config
        })
    }

    pub fn run_handwritten_trocr(mut self, images: Vec<DynamicImage>, device: &Device, dtype: DType) -> Result<()> {
        let pre_process_config = image_processor::PreProcessorConfig::default();
        let preprocessor = image_processor::VITImageProcessor::new(pre_process_config);

        let image = preprocessor.preprocess(images, device, dtype)?.to_device(device)?;

        let encoder = self.model.encoder().forward(&image)?;
        self.model.reset_kv_cache();

        let mut logits_processor = generation::LogitsProcessor::new(1337, None, None);

        let mut token_ids: Vec<u32> = vec![self.decoder_config.decoder_start_token_id];
        let mut tokenizer = TrocrSwedishHandwritten::get_tokenizer()?;

        println!(
            "Starting decode (decoder_start={}, bos={}, pad={}, eos={})",
            self.decoder_config.decoder_start_token_id,
            self.decoder_config.bos_token_id,
            self.decoder_config.pad_token_id,
            self.decoder_config.eos_token_id,
        );

        /*This iterates to 200 and force stops if an EOS token is never hit. 
        we shouldnt have more thatn 200 tokens per line of text so this limit is still overkill */
        for index in 0..1000{
            let context_size = if index >= 1 {1} else {
                token_ids.len()
            };

            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], device)?.unsqueeze(0)?;
            let logits = self.model.decode(&input_ids, &encoder, start_pos)?;

            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let mut logits_vec = logits.to_vec1::<f32>()?;
            if index == 0 {
                let mut top: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
                top.sort_by(|a, b| b.1.total_cmp(&a.1));
                println!("Step 0 top-5 logits: {:?}", &top[..5]);
            }
            if index >= 1 {
                // Avoid getting stuck in special-token loops once decoding has started.
                logits_vec[self.decoder_config.bos_token_id] = f32::NEG_INFINITY;
                logits_vec[self.decoder_config.pad_token_id] = f32::NEG_INFINITY;
            }
            let logits = Tensor::from_vec(logits_vec, self.decoder_config.vocab_size, device)?;
            let token = logits_processor.sample(&logits)?;
            token_ids.push(token);
            

            if let Some(t) =  tokenizer.next_token(token)?{
                print!("{t}");
                let _ = std::io::stdout().flush();
            }

            if token == self.decoder_config.eos_token_id {
                break;
            }
        }   

        if let Ok(Some(rest)) = tokenizer.decode_rest() {
            print!("{rest}");
        }
        println!();

        Ok(())
    }

    pub fn get_tokenizer() -> Result<TokenOutputStream>{
        let path = "models/trocr/tokenizer.json";

        let tokenizer = Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;

        Ok(TokenOutputStream::new(tokenizer))
    }

    pub fn load_config() -> Result<ModelConfig> {
        let config_path= "models/trocr/config.json";
        let config_str = std::fs::read_to_string(config_path)?;
        let model_config: ModelConfig = serde_json::from_str(&config_str)?;

        Ok(model_config)
    }
}
