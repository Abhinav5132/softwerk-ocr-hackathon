use image::DynamicImage;



/*
path -> path to the PNG file
img_coordinates -> if there is an image present in the page a image descriptor model will be run, t
hese coordianes are gotten from LightOnOCr 
// TODO maybe add the pdf name so we can lump all the pdfs together
*/

#[derive(Clone)]
pub struct Page {
    pub path: String, 
    pub name: String,
}

#[derive(Clone)]
pub struct ImageCoordinates {
    pub lable: String,
    pub x1: u32,
    pub x2: u32,
    pub y1: u32,
    pub y2: u32,
}

pub struct ImageDimentions{
    pub img_h: u32,
    pub img_w: u32
}

/* 
This contains the output of the LightOnOcr model. Unprocessed out is the out put from the model. 
It still contains the ![image] coordiantes.
If handwriting model is implemented it would check if the page is handwritten and if it is, 
then the Trocr model will change unprocessed_output to the string it generated. 
This is quite wastefull as it first goes through lightOnOcr -> handwriting check -> Trocr.
But this removes the need for a classifier step at the start. 
This struct is then passed to the moondream model which uses the unprocessed output as context 
for image description. Since we havent stripped the ![image] coordinates we can use that to give the model 
context of the text above and below. 
If the text is only images unprocessed_output will be empty and we then run the model with generic context.
*/
pub struct UnprocessedOutput{
    pub page: Page,
    pub image_dimentions: ImageDimentions,
    pub loaded_image: DynamicImage,
    pub unprocessed_output: String,
    pub image_regions: Vec<ImageCoordinates>,
    pub is_handwritten: bool,
    pub lighton_confidence: f32
}

pub struct ProcessedOutput {
    pub page: Page,
    pub processed_output: String // This contains the output from transcription(lightonocr || trocr) and the ![image]coordinates replaced with moondream output
}