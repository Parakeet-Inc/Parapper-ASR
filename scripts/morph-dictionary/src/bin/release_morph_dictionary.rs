use std::{env, path::Path};

use anyhow::{Context, Result, bail};
use parapper_morph_dictionary::morph_dictionary_release::{
    ArtifactSource, ReleaseContract, package_release, prepare_native_dictionary,
    verify_official_expanded_dictionary, verify_official_release, verify_official_source,
};

fn main() -> Result<()> {
    let mut args = env::args_os();
    let _program = args.next();
    let Some(command) = args.next() else {
        bail!("{}", usage());
    };
    match command.to_string_lossy().as_ref() {
        "prepare-native" => {
            let expanded = args.next().context("missing expanded system.dic path")?;
            let compressed = args.next().context("missing system.dic.zst output path")?;
            ensure_no_more(args)?;
            verify_official_expanded_dictionary(Path::new(&expanded))?;
            let record = prepare_native_dictionary(Path::new(&expanded), Path::new(&compressed))?;
            println!("prepared {} bytes sha256={}", record.size, record.sha256);
        }
        "package" => {
            let source = args.next().context("missing source artifact directory")?;
            let output = args.next().context("missing Release output directory")?;
            ensure_no_more(args)?;
            let source = ArtifactSource::new(source);
            verify_official_source(&source)?;
            package_release(&source, output, &ReleaseContract::official())?;
        }
        "verify" => {
            let directory = args.next().context("missing Release directory")?;
            ensure_no_more(args)?;
            verify_official_release(Path::new(&directory))?;
        }
        _ => bail!("{}", usage()),
    }
    Ok(())
}

fn ensure_no_more(mut args: impl Iterator<Item = std::ffi::OsString>) -> Result<()> {
    if let Some(argument) = args.next() {
        bail!("unexpected argument: {}", argument.to_string_lossy());
    }
    Ok(())
}

fn usage() -> &'static str {
    "usage:
  release_morph_dictionary prepare-native EXPANDED_SYSTEM_DIC OUTPUT_SYSTEM_DIC_ZST
  release_morph_dictionary package SOURCE_ARTIFACT_DIR RELEASE_OUTPUT_DIR
  release_morph_dictionary verify RELEASE_OUTPUT_DIR"
}
