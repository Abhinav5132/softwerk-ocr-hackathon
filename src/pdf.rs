use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::page_struct::Page;

pub fn get_pdfs_converted_as_images() -> Vec<Page> {
    fs::create_dir_all("./data/images").expect("failed to create images directory");

    let entries: Vec<PathBuf> = fs::read_dir("./data")
        .expect("failed to read data directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| e.path())
        .collect();

    let mut pages: Vec<Page> = entries
        .par_iter()
        .flat_map(|path| {
            match path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).as_deref() {
                Some("pdf") => convert_pdf_to_images(path).unwrap_or_default(),
                Some("png") | Some("jpg") | Some("jpeg") => {
                    copy_image(path).map(|p| vec![p]).unwrap_or_default()
                }
                _ => vec![],
            }
        })
        .collect();

    // sort all pages by filename at the end
    pages
}

fn convert_pdf_to_images(path: &PathBuf) -> Option<Vec<Page>> {
    let name = path.file_stem()?.to_str()?.to_string();
    let output_prefix = format!("./data/images/{}", name);

    let status = Command::new("pdftoppm")
        .args(["-png", "-r", "200"])
        .arg(path.as_os_str())
        .arg(&output_prefix)
        .status()
        .unwrap_or_else(|_| panic!("failed to run pdftoppm for {}", name));

    if !status.success() {
        eprintln!("pdftoppm failed for {}", name);
        return None;
    }

    let pages: Vec<Page> = fs::read_dir("./data/images")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("png")
            && p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with(&name))
                .unwrap_or(false)
        })
        .filter_map(|p| p.to_str().map(|s| Page {
            path: s.to_string(),
            unprocessed: None,
            processed: None,
        }))
        .collect();

    Some(pages)
}

fn copy_image(path: &PathBuf) -> Option<Page> {
    let filename = path.file_name()?.to_str()?.to_string();
    let dest = format!("./data/images/{}", filename);
    fs::copy(path, &dest).ok()?;
    println!("Copied: {}", filename);
    Some(Page { path: dest, unprocessed: None, processed: None })
}