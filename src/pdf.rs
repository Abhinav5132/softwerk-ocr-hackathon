use std::{fs, path::Path, process::Command};
use anyhow::Result;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

//TODO If the input is already and image dont convert it, move a copy of it to the images folder to convert.
pub fn convert_pdf_to_image() -> Result<()>{
    let dir = "./data/images";

    // Clean the images folder first before any conversions
    if Path::new(dir).exists() {
        fs::read_dir(dir)
            .expect("failed to read images directory")
            .filter_map(|entry| entry.ok())
            .filter(|ent| ent.file_type().ok().map(|ft| ft.is_file()).unwrap_or(false))
            .for_each(|ent| {
                if let Err(e) = fs::remove_file(ent.path()) {
                    eprintln!("Warning: failed to remove file {:?}: {}", ent.path(), e);
                }
            });
    } else {
        fs::create_dir_all(dir).expect("failed to create dir");
    }

    let paths: Vec<_> = fs::read_dir("./data")
        .expect("failed to read data directory")
        .filter_map(|ent| ent.ok())
        .map(|ent| ent.path())
        .filter(|p| p.is_file())
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
    Ok(())
}

pub fn convert_single_pdf_to_image(pdf_path: &str) -> Result<()> {
    let dir = "./data/images";
    if !Path::new(dir).exists() {
        fs::create_dir_all(dir)?;
    }

    let name = Path::new(pdf_path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid PDF path: {pdf_path}"))?;

    Command::new("pdftoppm")
        .arg("-png")
        .arg("-r").arg("200")
        .arg(pdf_path)
        .arg(format!("./data/images/{name}"))
        .status()
        .map_err(|e| anyhow::anyhow!("failed to convert to png {name}: {e}"))?;

    Ok(())
}