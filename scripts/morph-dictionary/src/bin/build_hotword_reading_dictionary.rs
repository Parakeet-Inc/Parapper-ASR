use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use parapper_morph_dictionary::hotword_reading_dictionary::build_hotword_reading_dictionaries;

fn main() -> Result<()> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        arguments.len() == 3,
        "usage: build_hotword_reading_dictionary <sudachi-full-winfo.csv> <cmudict.dict> <output-dir>"
    );
    let sudachi_full = PathBuf::from(&arguments[0]);
    let cmudict = PathBuf::from(&arguments[1]);
    let output = PathBuf::from(&arguments[2]);
    build_hotword_reading_dictionaries(&sudachi_full, &cmudict, &output)
        .with_context(|| format!("failed to build dictionaries in {}", output.display()))
}
