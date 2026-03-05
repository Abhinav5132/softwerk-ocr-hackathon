use candle_core::{DType, Device, Tensor};
use candle_transformers::{
    models::moondream::{self, Model}
};
use image::DynamicImage;
use tokenizers::Tokenizer;
use anyhow::Result;
use anyhow::Error as E;
use candle_nn::VarBuilder;
use crate::{get_dtype, moondream::text_generation::TextGeneration, page_struct::{ImageCoordinates, ImageDimentions, Page, ProcessedOutput, UnprocessedOutput}};

mod text_generation;

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

pub fn build_model(device: &Device) -> Result<TextGeneration>{
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

    let model = moondream::Model::new(&config, vb)?;

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

    Ok(pipeline)

}

pub fn run_moondream(mut pipeline: TextGeneration, device: &Device, mut unprocessed_outputs: Vec<UnprocessedOutput>) -> Result<Vec<ProcessedOutput>> {
    let dtype = get_dtype(device);
    let mut processed_outputs:Vec<ProcessedOutput> = vec![];
    for output in unprocessed_outputs{
        let mut context = output.unprocessed_output;
        let page = output.page;
        for image_region in output.image_regions{
            //extract the image here 
            let image = extract_image(&image_region, &output.image_dimentions, &output.loaded_image);
            let image = convert_image(image, device)?.to_device(device)?.to_dtype(dtype)?;

            let image_embeds = image.unsqueeze(0)?;
            let image_embeds = image_embeds.apply(pipeline.model.vision_encoder())?;

            let prompt = build_prompt(&context);
            pipeline.model.text_model.clear_kv_cache();
            let image_descrption = pipeline.run(&prompt, &image_embeds, 250usize)?;
            let label = image_region.lable;
            context = context.replace(&label, &image_descrption); // TODO maybe we can embed the image snipped back in. 
        }
        
        let processed_output = ProcessedOutput{
            page,
            processed_output: context,
        };
        processed_outputs.push(processed_output);
    }
    
    Ok(processed_outputs)

}

pub fn extract_image(
    region: &ImageCoordinates, 
    dimentions: &ImageDimentions, 
    image: &DynamicImage
) -> DynamicImage{
    
    image.crop_imm(
        region.x1, 
        region.y1, 
        (region.x2 - region.x1).min(dimentions.img_w - region.x1), 
        (region.y2 - region.y1).min(dimentions.img_h - region.y1)
    )
}

/* Builds the prompt to be used by moondream1 */
pub fn build_prompt(context: &str) -> String {
    let mut ctx = context.trim();
    if ctx.is_empty() {
        ctx = "No surrounding text content found.";
    }
    return format!(
"\n\nQuestion: You are an expert forensic document and image analyzer. Your task is to perform precise Optical Character Recognition (OCR) and detailed visual analysis on the provided case file image.

You will be provided with the text surrounding this image for context. Use this context to inform your analysis, but do not hallucinate details that are not visible in the image.

Surrounding Text Context:
{}

Analyze the image and output your findings EXACTLY in the following XML format. Do not include conversational filler. If a section is not applicable or not visible, write ‘None detected’.

<extracted_text>
[Transcribe all visible text in the image exactly as written, preserving line breaks, typos, and capitalization. Differentiate between handwritten and printed text if possible.]
</extracted_text>

<people_and_subjects>
[Describe all visible persons. Include estimated age, sex, clothing, distinct physical features, facial expressions, and physical positioning/actions.]
</people_and_subjects>

<objects_and_evidence>
[List and describe all distinct objects. Include colors, makes/models, conditions (e.g., damaged, pristine), and spatial relationships to other objects.]
</objects_and_evidence>

<location_and_environment>
[Describe the setting. Note indoor/outdoor, lighting conditions, weather, architectural details, and any identifiable signage or landmarks.]
</location_and_environment>

<forensic_anomalies_and_metadata>
[Describe anything else of investigative value. This includes timestamps, logos, official stamps, signatures, watermarks, damage to the physical document/photograph (e.g., tears, stains), or inconsistencies.]
</forensic_anomalies_and_metadata>
\n\nAnswer:", ctx.chars().take(300).collect::<String>());
}
