use std::fs::{self};
use std::path::Path;
use std::process::Command;
use std::time;
use std::io::Write;
use candle_nn::VarBuilder;
use rayon::prelude::*;
use anyhow::Result;
use candle_core::{DType, Device, safetensors::*};
use tokenizers::Tokenizer;

pub mod Light_on_ocr;
pub mod trocr;
pub mod moondream;
use crate::Light_on_ocr::model_functions::{build_model, print_safetensors, run_model};

mod page_struct;
use moondream::run_moondream;
use crate::page_struct::Page;
use crate::Light_on_ocr::preprocess::preprocess;
use crate::trocr::TrocrSwedishHandwritten;
use crate::trocr::line_segementation::line_segemenation;

fn main() {
    let start_time = time::Instant::now();
    let device = select_device();
    let dtype = DType::F32;
    /* 
    let dir = "./data/images";
    

    if !Path::new(dir).exists() {
        fs::create_dir_all(dir).expect("failed to create dir");
    } else{
        fs::read_dir(dir).expect("failed to read images directory")
        .map(|entry| entry.expect("failed to get dirEntry"))
        .filter(|ent| ent.file_type().expect("unable to get file type").is_file())
        .for_each(|ent| { fs::remove_file(ent.path()).expect("failed to remove file"); });
    }

    let paths: Vec<_> = fs::read_dir("./data")
        .expect("failed to read data directory")
        .map(|ent| ent.expect("failed to get path"))
        .filter(|ent| ent.file_type().expect("").is_file())
        .map(|ent| ent.path())
        .collect();

    paths.into_par_iter().for_each(|path| {
        let name = path.as_path().file_name().unwrap().display();
        let _ = Command::new("pdftoppm")
            .arg("-png")
            .arg("-r").arg("200")
            .arg(path.as_os_str())
            .arg(format!("./data/images/{name}"))
            .status().unwrap_or_else(|_| panic!("failed to convert to png {name}"));
    });
 
       
    
    let pages = vec![]; // Empty vector for now add actual loading later,

    if let Ok((model, tokenizer)) = build_model(&device){
        match run_model(model, tokenizer, device, pages) {
            Ok(_) => {
                println!("Transcription finished successfully")
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
    */
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

    let context = "LÖRDAG 24 APRIL 1993

# Mystikern bakom den hemliga sekten

![image](image_1.png)100,210,620,750

Arne Weise har varit Swedenborgare helt sit liv. Han döptes i församlingen och han blev också konfirmerad där. — Men jag är inte aktiv medlem. Jag jobbar helt enkelt för mycket, konstaterar han.

---

**STOCKHOLM (IDAG) TV-kändisen Arne Weise är medlem i en okänd religiös sekt med starkt fläste i Sydafrika. Varje söndag samlas ett total av medlemmarna i den svenska avdelningen till egna gudstjänster.**

— Vi tror på 1700-tals-mystikern Swedenborgs läror, säger Arne Weise.

---

**Av OISIN CANTWELL**

I USA, England och Sydafrika har Nya kyrkan — som grundades 1787 i London — sommanligt 50 000 medlemmar.

Men i Sverige är sekten nästan helt okänd. Den har ett 50-tal medlemmar — och det är bara ett total av dessa som varje söndag samlas till gudstjänster i den egna kyrkan i Stockholms innerstad.

Arne Weise, 63, har varit med i sekten i hela sitt liv. Hans mamma var Swedenborgare, han är djup och konfirmerad i församlingen.

— Men jag är inte aktiv medlem. Jag jobbar helt enkelt för mycket, säger han till IDAG. Sekten är inspirerad av Emanuel Swedenborg, känd ytterekapsman och mystiker som levde på 1700-tålet.

Han är en mycket sympatisk måstare med en vildigt human inställning till mänskliga svagheter.

---

**INGET HOPP FÖR ONDA MÄNNISKOR**

Sekten trar inte på himmel eller helvete på samma sätt som kristendomen. Goda och onda människor kommer till olika platser, men de fortsätter att leva på samma sätt som de garde under det första livet.

— En en medlemsa fortsätter att varn and. Men huv eller hon skulle ju ända inte finna sig till rö i himlen. Han trivs ju med att supa och hona eller vad han nu har gjort.

Ibland, säger Arne Weise, känner han en oerhård nähet till ett högre väsen.

— Det kan vara på savannen i Kenya en sjärmklar natt. Eller när jag är uta vid min sommanstuga. Det är svårt att förklara kanslan, men den är måttig.

Arne Weise säger att han ibland varit tveksam till att behöva leva för eviga.

— Men enligt läran gör vi nyttta i det eviga livet. Man fortsätter att fylla en funktion.

Sekten trar att det var God — inte Jesus — som steg ner till jorden. Därför ses inte heller korrekt som symbol för iblander.

Swedenborg ansag att det inte räckte med att tre — man måste hövsa sin tre med goda gärningar. Även det är ett grunddrag i läran.

Församlingen är annorlunda även på det sättet att begravningen inte bara ses som en sörgestand. I stället inlever den det eviga livet och prästen av därför klädd i vit.

Swedenborg är en av de mest kända svenskarerna över huvud taget. I Sverige nästan bara som vetenskapeman, men utomlands även som mystiker.

Hans korrespondenslåra — att alla ting på jorden är en skuggbild av ett andligt — har inspirerat mänga filosofer och färfattare som Strindberg och Håtze.

---

Emanuel Swedenborg, vars läror förs vidare av en liten okänd sekt, var en framstående vetenskapeman och mystiker.";

    let image = image::open("data/image.png").unwrap();

    let _= run_moondream(&select_device(), image, context.to_string()).unwrap();
    let elapsed = start_time.elapsed().as_secs();
    println!("{elapsed}");

}

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