use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    io::{BufReader, Cursor, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tar::{Archive, Builder, Header};
use vibrato_rkyv::Dictionary;

pub const RELEASE_TAG: &str = "morph-dictionary-unidic-cwj-3.1.1-v1";
pub const NATIVE_ASSET_NAME: &str = "parapper-unidic-cwj-3_1_1-compact-raw-v1.tar.zst";
pub const AUTHORS_ASSET_NAME: &str = "AUTHORS";
pub const BSD_ASSET_NAME: &str = "BSD";
pub const NOTICE_ASSET_NAME: &str = "NOTICE";
pub const DICTIONARY_DIR: &str = "unidic-cwj-3_1_1";
pub const RELEASE_MANIFEST_NAME: &str = "release-manifest.json";
pub const RELEASE_CHECKSUMS_NAME: &str = "SHA256SUMS";

const NATIVE_EXPANDED_SIZE: u64 = 44_497_344;
const NATIVE_EXPANDED_SHA256: &str =
    "1d7f0a194ae1f296f740fdb08433ef1fce81c0a353e7ce3585146b295ca87ef6";
const NATIVE_COMPRESSED_SIZE: u64 = 7_469_299;
const NATIVE_COMPRESSED_SHA256: &str =
    "92fbc19982ada2d565115819d11083d51dd1285c4dfa059112fafcad79bcca86";
const AUTHORS_SIZE: u64 = 34;
const AUTHORS_SHA256: &str = "d3cde34eb1ec1f6b4b4d78b77d2ff909f812a3c71d451cf97adb48ce7b1b0ab5";
const BSD_SIZE: u64 = 1_547;
const BSD_SHA256: &str = "19ed65bc4230f2c786d955247da91d21f609149accd49715971d14319f97af8d";
const NOTICE_SIZE: u64 = 851;
const NOTICE_SHA256: &str = "6ab9139ca40daa086c2e47e07eed7de1be983c07fb1d529d0e096f08fa3d65e0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseContract {
    pub release_tag: String,
    pub native_asset_name: String,
    pub dictionary_dir: String,
}

impl ReleaseContract {
    #[must_use]
    pub fn official() -> Self {
        Self {
            release_tag: RELEASE_TAG.into(),
            native_asset_name: NATIVE_ASSET_NAME.into(),
            dictionary_dir: DICTIONARY_DIR.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArtifactSource {
    pub directory: PathBuf,
}

impl ArtifactSource {
    #[must_use]
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    #[must_use]
    pub fn native_dictionary(&self) -> PathBuf {
        self.directory.join("system.dic.zst")
    }

    #[must_use]
    pub fn native_expanded_dictionary(&self) -> PathBuf {
        self.directory.join("system.dic")
    }

    #[must_use]
    pub fn authors(&self) -> PathBuf {
        self.directory.join("AUTHORS")
    }

    #[must_use]
    pub fn bsd_license(&self) -> PathBuf {
        self.directory.join("BSD")
    }

    #[must_use]
    pub fn notice(&self) -> PathBuf {
        self.directory.join("NOTICE")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct FileRecord {
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct NativeManifest {
    schema_version: u32,
    dictionary_id: String,
    representation: String,
    feature_encoding: String,
    expanded_dictionary: FileRecord,
    files: BTreeMap<String, FileRecord>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ReleaseManifest {
    schema_version: u32,
    release_tag: String,
    assets: BTreeMap<String, FileRecord>,
}

/// Validates an expanded rkyv dictionary, then compresses its already
/// serialized bytes through the same Rust zstd level-19 path used by the
/// dictionary builder.
///
/// # Errors
///
/// Returns an error when the input is not a valid rkyv dictionary or the
/// output cannot be written.
pub fn prepare_native_dictionary(expanded: &Path, compressed: &Path) -> Result<FileRecord> {
    let validation_input = fs::File::open(expanded)
        .with_context(|| format!("failed to open expanded dictionary {}", expanded.display()))?;
    Dictionary::read(BufReader::new(validation_input))
        .with_context(|| format!("invalid rkyv dictionary {}", expanded.display()))?;

    if let Some(parent) = compressed.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temporary = compressed.with_extension("zst.writing");
    let output = fs::File::create(&temporary)
        .with_context(|| format!("failed to create {}", temporary.display()))?;
    let mut encoder = zstd::Encoder::new(output, 19)
        .with_context(|| format!("failed to start zstd encoder for {}", compressed.display()))?;
    let mut input = fs::File::open(expanded)
        .with_context(|| format!("failed to reopen {}", expanded.display()))?;
    std::io::copy(&mut input, &mut encoder)
        .with_context(|| format!("failed to compress {}", expanded.display()))?;
    encoder
        .finish()
        .with_context(|| format!("failed to finish {}", temporary.display()))?;
    replace_file(&temporary, compressed)?;
    file_record(compressed)
}

/// Builds the immutable GitHub Release assets and both levels of integrity
/// metadata.
///
/// # Errors
///
/// Returns an error if a required source artifact is missing, the destination
/// is not empty, or an asset cannot be packaged and verified.
pub fn package_release(
    source: &ArtifactSource,
    output_directory: impl AsRef<Path>,
    contract: &ReleaseContract,
) -> Result<()> {
    let output_directory = output_directory.as_ref();
    require_source_file(&source.native_dictionary())?;
    require_source_file(&source.authors())?;
    require_source_file(&source.bsd_license())?;
    require_source_file(&source.notice())?;
    ensure_empty_output_directory(output_directory)?;

    let expanded_dictionary = expanded_dictionary_record(&source.native_dictionary())?;
    let internal_files = BTreeMap::from([
        (
            "AUTHORS".into(),
            file_record(&source.authors()).context("failed to hash AUTHORS")?,
        ),
        (
            "BSD".into(),
            file_record(&source.bsd_license()).context("failed to hash BSD")?,
        ),
        (
            "NOTICE".into(),
            file_record(&source.notice()).context("failed to hash NOTICE")?,
        ),
        (
            "system.dic.zst".into(),
            file_record(&source.native_dictionary()).context("failed to hash system.dic.zst")?,
        ),
    ]);
    let native_manifest = NativeManifest {
        schema_version: 1,
        dictionary_id: contract.dictionary_dir.clone(),
        representation: "compact-raw".into(),
        feature_encoding: "[PP][S][F]".into(),
        expanded_dictionary,
        files: internal_files.clone(),
    };
    let native_manifest_bytes = json_bytes(&native_manifest)?;
    let internal_checksums = checksum_text(&internal_files);
    let native_asset = output_directory.join(&contract.native_asset_name);
    write_native_package(
        source,
        &native_asset,
        contract,
        &native_manifest_bytes,
        internal_checksums.as_bytes(),
    )?;

    let authors_asset = output_directory.join(AUTHORS_ASSET_NAME);
    fs::copy(source.authors(), &authors_asset)
        .with_context(|| format!("failed to copy {}", source.authors().display()))?;
    let bsd_asset = output_directory.join(BSD_ASSET_NAME);
    fs::copy(source.bsd_license(), &bsd_asset)
        .with_context(|| format!("failed to copy {}", source.bsd_license().display()))?;
    let notice_asset = output_directory.join(NOTICE_ASSET_NAME);
    fs::copy(source.notice(), &notice_asset)
        .with_context(|| format!("failed to copy {}", source.notice().display()))?;

    let assets = BTreeMap::from([
        (
            contract.native_asset_name.clone(),
            file_record(&native_asset)?,
        ),
        (AUTHORS_ASSET_NAME.into(), file_record(&authors_asset)?),
        (BSD_ASSET_NAME.into(), file_record(&bsd_asset)?),
        (NOTICE_ASSET_NAME.into(), file_record(&notice_asset)?),
    ]);
    let release_manifest = ReleaseManifest {
        schema_version: 1,
        release_tag: contract.release_tag.clone(),
        assets: assets.clone(),
    };
    fs::write(
        output_directory.join(RELEASE_MANIFEST_NAME),
        json_bytes(&release_manifest)?,
    )
    .context("failed to write release manifest")?;
    fs::write(
        output_directory.join(RELEASE_CHECKSUMS_NAME),
        checksum_text(&assets),
    )
    .context("failed to write release checksums")?;

    verify_release(output_directory, contract)
}

/// Verifies the outer Release manifest and checksums, then verifies every
/// member of the native package against its internal manifest.
///
/// # Errors
///
/// Returns an error for missing, extra, truncated, or modified content.
pub fn verify_release(directory: &Path, contract: &ReleaseContract) -> Result<()> {
    let expected_names = BTreeSet::from([
        contract.native_asset_name.clone(),
        AUTHORS_ASSET_NAME.into(),
        BSD_ASSET_NAME.into(),
        NOTICE_ASSET_NAME.into(),
        RELEASE_MANIFEST_NAME.into(),
        RELEASE_CHECKSUMS_NAME.into(),
    ]);
    let actual_names = fs::read_dir(directory)
        .with_context(|| format!("failed to read release directory {}", directory.display()))?
        .map(|entry| {
            let entry = entry?;
            ensure!(
                entry.file_type()?.is_file(),
                "unexpected directory {}",
                entry.path().display()
            );
            entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF-8 release asset name"))
        })
        .collect::<Result<BTreeSet<_>>>()?;
    ensure!(
        actual_names == expected_names,
        "release asset set mismatch: expected {expected_names:?}, found {actual_names:?}"
    );

    let manifest_path = directory.join(RELEASE_MANIFEST_NAME);
    let manifest: ReleaseManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("invalid {}", manifest_path.display()))?;
    ensure!(
        manifest.schema_version == 1,
        "unsupported Release manifest schema"
    );
    ensure!(
        manifest.release_tag == contract.release_tag,
        "Release tag mismatch: expected {}, found {}",
        contract.release_tag,
        manifest.release_tag
    );
    let expected_asset_names = BTreeSet::from([
        contract.native_asset_name.clone(),
        AUTHORS_ASSET_NAME.into(),
        BSD_ASSET_NAME.into(),
        NOTICE_ASSET_NAME.into(),
    ]);
    ensure!(
        manifest.assets.keys().cloned().collect::<BTreeSet<_>>() == expected_asset_names,
        "Release manifest asset set mismatch"
    );
    verify_records(directory, &manifest.assets)?;

    let checksums_path = directory.join(RELEASE_CHECKSUMS_NAME);
    let checksums = fs::read_to_string(&checksums_path)
        .with_context(|| format!("failed to read {}", checksums_path.display()))?;
    ensure!(
        checksums == checksum_text(&manifest.assets),
        "{RELEASE_CHECKSUMS_NAME} does not match the Release manifest"
    );
    verify_native_package(&directory.join(&contract.native_asset_name), contract)
}

/// Verifies the exact v0.4.0 Release contents after generic integrity checks.
///
/// # Errors
///
/// Returns an error when any audited dictionary or license record differs.
pub fn verify_official_release(directory: &Path) -> Result<()> {
    let contract = ReleaseContract::official();
    verify_release(directory, &contract)?;
    let manifest = read_native_manifest(&directory.join(&contract.native_asset_name), &contract)?;
    verify_expected(
        "expanded system.dic",
        &manifest.expanded_dictionary,
        NATIVE_EXPANDED_SIZE,
        NATIVE_EXPANDED_SHA256,
    )?;
    verify_expected(
        "packaged AUTHORS",
        &manifest.files["AUTHORS"],
        AUTHORS_SIZE,
        AUTHORS_SHA256,
    )?;
    verify_expected("packaged BSD", &manifest.files["BSD"], BSD_SIZE, BSD_SHA256)?;
    verify_expected(
        "packaged NOTICE",
        &manifest.files["NOTICE"],
        NOTICE_SIZE,
        NOTICE_SHA256,
    )?;
    verify_expected(
        AUTHORS_ASSET_NAME,
        &file_record(&directory.join(AUTHORS_ASSET_NAME))?,
        AUTHORS_SIZE,
        AUTHORS_SHA256,
    )?;
    verify_expected(
        BSD_ASSET_NAME,
        &file_record(&directory.join(BSD_ASSET_NAME))?,
        BSD_SIZE,
        BSD_SHA256,
    )?;
    verify_expected(
        NOTICE_ASSET_NAME,
        &file_record(&directory.join(NOTICE_ASSET_NAME))?,
        NOTICE_SIZE,
        NOTICE_SHA256,
    )
}

/// Enforces the exact v0.4.0 source artifact contract before publication.
///
/// # Errors
///
/// Returns an error when the generated dictionary or license inputs differ
/// from the audited `UniDic` CWJ 3.1.1 compact-Raw artifacts.
pub fn verify_official_source(source: &ArtifactSource) -> Result<()> {
    verify_expected(
        "AUTHORS",
        &file_record(&source.authors())?,
        AUTHORS_SIZE,
        AUTHORS_SHA256,
    )?;
    verify_expected(
        "BSD",
        &file_record(&source.bsd_license())?,
        BSD_SIZE,
        BSD_SHA256,
    )?;
    verify_expected(
        "NOTICE",
        &file_record(&source.notice())?,
        NOTICE_SIZE,
        NOTICE_SHA256,
    )?;
    verify_expected(
        "system.dic.zst",
        &file_record(&source.native_dictionary())?,
        NATIVE_COMPRESSED_SIZE,
        NATIVE_COMPRESSED_SHA256,
    )?;
    let expanded = expanded_dictionary_record(&source.native_dictionary())?;
    verify_expected(
        "expanded system.dic",
        &expanded,
        NATIVE_EXPANDED_SIZE,
        NATIVE_EXPANDED_SHA256,
    )
}

/// Checks the known expanded native dictionary before re-serialization.
///
/// # Errors
///
/// Returns an error if the input differs from the audited compact-Raw rkyv
/// dictionary prepared for v0.4.0.
pub fn verify_official_expanded_dictionary(path: &Path) -> Result<()> {
    verify_expected(
        "expanded system.dic",
        &file_record(path)?,
        NATIVE_EXPANDED_SIZE,
        NATIVE_EXPANDED_SHA256,
    )
}

fn verify_expected(name: &str, actual: &FileRecord, size: u64, sha256: &str) -> Result<()> {
    ensure!(
        actual.size == size,
        "{name} size mismatch: expected {size}, found {}",
        actual.size
    );
    ensure!(
        actual.sha256 == sha256,
        "{name} SHA-256 mismatch: expected {sha256}, found {}",
        actual.sha256
    );
    Ok(())
}

fn require_source_file(path: &Path) -> Result<()> {
    ensure!(
        path.is_file(),
        "required Release source file is missing: {}",
        path.display()
    );
    Ok(())
}

fn ensure_empty_output_directory(directory: &Path) -> Result<()> {
    if directory.exists() {
        ensure!(
            fs::read_dir(directory)?.next().is_none(),
            "Release output directory must be empty: {}",
            directory.display()
        );
    } else {
        fs::create_dir_all(directory)
            .with_context(|| format!("failed to create {}", directory.display()))?;
    }
    Ok(())
}

fn replace_file(temporary: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        fs::remove_file(destination)
            .with_context(|| format!("failed to replace {}", destination.display()))?;
    }
    fs::rename(temporary, destination).with_context(|| {
        format!(
            "failed to move {} to {}",
            temporary.display(),
            destination.display()
        )
    })
}

fn file_record(path: &Path) -> Result<FileRecord> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let size = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash)
        .with_context(|| format!("failed to hash {}", path.display()))?;
    Ok(FileRecord {
        size,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn bytes_record(bytes: &[u8]) -> FileRecord {
    FileRecord {
        size: u64::try_from(bytes.len()).expect("byte slice length should fit u64"),
        sha256: format!("{:x}", Sha256::digest(bytes)),
    }
}

fn expanded_dictionary_record(compressed: &Path) -> Result<FileRecord> {
    let bytes =
        fs::read(compressed).with_context(|| format!("failed to read {}", compressed.display()))?;
    expanded_dictionary_record_from_bytes(&bytes, &compressed.display().to_string())
}

fn expanded_dictionary_record_from_bytes(compressed: &[u8], label: &str) -> Result<FileRecord> {
    let mut decoder = zstd::Decoder::new(Cursor::new(compressed))
        .with_context(|| format!("failed to start zstd decoder for {label}"))?;
    let mut expanded = Vec::new();
    decoder
        .read_to_end(&mut expanded)
        .with_context(|| format!("failed to decompress {label}"))?;
    Dictionary::read(Cursor::new(&expanded))
        .with_context(|| format!("invalid native rkyv dictionary {label}"))?;
    Ok(bytes_record(&expanded))
}

fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).context("failed to serialize manifest")?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn checksum_text(records: &BTreeMap<String, FileRecord>) -> String {
    records
        .iter()
        .fold(String::new(), |mut output, (path, record)| {
            writeln!(output, "{}  {path}", record.sha256).expect("writing to a String cannot fail");
            output
        })
}

fn write_native_package(
    source: &ArtifactSource,
    output: &Path,
    contract: &ReleaseContract,
    manifest: &[u8],
    checksums: &[u8],
) -> Result<()> {
    let temporary = output.with_extension("zst.writing");
    let file = fs::File::create(&temporary)
        .with_context(|| format!("failed to create {}", temporary.display()))?;
    let encoder = zstd::Encoder::new(file, 19)
        .with_context(|| format!("failed to start zstd encoder for {}", output.display()))?;
    let mut archive = Builder::new(encoder);
    archive.mode(tar::HeaderMode::Deterministic);
    for (name, path) in [
        ("AUTHORS", source.authors()),
        ("BSD", source.bsd_license()),
        ("NOTICE", source.notice()),
        ("system.dic.zst", source.native_dictionary()),
    ] {
        append_path(
            &mut archive,
            &format!("{}/{name}", contract.dictionary_dir),
            &path,
        )?;
    }
    append_bytes(
        &mut archive,
        &format!("{}/manifest.json", contract.dictionary_dir),
        manifest,
    )?;
    append_bytes(
        &mut archive,
        &format!("{}/SHA256SUMS", contract.dictionary_dir),
        checksums,
    )?;
    let encoder = archive
        .into_inner()
        .context("failed to finish native tar archive")?;
    encoder
        .finish()
        .context("failed to finish native zstd package")?;
    replace_file(&temporary, output)
}

fn append_path<W: Write>(
    archive: &mut Builder<W>,
    archive_path: &str,
    source_path: &Path,
) -> Result<()> {
    let mut source = fs::File::open(source_path)
        .with_context(|| format!("failed to open {}", source_path.display()))?;
    let size = source.metadata()?.len();
    let mut header = deterministic_header(size);
    archive
        .append_data(&mut header, archive_path, &mut source)
        .with_context(|| format!("failed to append {archive_path}"))
}

fn append_bytes<W: Write>(
    archive: &mut Builder<W>,
    archive_path: &str,
    bytes: &[u8],
) -> Result<()> {
    let mut header =
        deterministic_header(u64::try_from(bytes.len()).expect("byte slice length should fit u64"));
    archive
        .append_data(&mut header, archive_path, bytes)
        .with_context(|| format!("failed to append {archive_path}"))
}

fn deterministic_header(size: u64) -> Header {
    let mut header = Header::new_gnu();
    header.set_size(size);
    header.set_mode(0o644);
    header.set_uid(0);
    header.set_gid(0);
    header.set_mtime(0);
    header.set_cksum();
    header
}

fn verify_records(base: &Path, records: &BTreeMap<String, FileRecord>) -> Result<()> {
    for (name, expected) in records {
        let path = base.join(name);
        require_source_file(&path)?;
        let actual = file_record(&path)?;
        ensure!(
            actual.size == expected.size,
            "{name} size mismatch: expected {}, found {}",
            expected.size,
            actual.size
        );
        ensure!(
            actual.sha256 == expected.sha256,
            "{name} SHA-256 mismatch: expected {}, found {}",
            expected.sha256,
            actual.sha256
        );
    }
    Ok(())
}

/// Verifies the native package independently from its outer Release metadata.
///
/// # Errors
///
/// Returns an error for a malformed archive, member-set drift, or an internal
/// size/checksum mismatch.
pub fn verify_native_package(package: &Path, contract: &ReleaseContract) -> Result<()> {
    read_native_manifest(package, contract).map(|_| ())
}

fn read_native_manifest(package: &Path, contract: &ReleaseContract) -> Result<NativeManifest> {
    let input =
        fs::File::open(package).with_context(|| format!("failed to open {}", package.display()))?;
    let decoder = zstd::Decoder::new(BufReader::new(input))
        .with_context(|| format!("failed to decompress {}", package.display()))?;
    let mut archive = Archive::new(decoder);
    let mut files = BTreeMap::new();
    for entry in archive.entries().context("failed to read native package")? {
        let mut entry = entry.context("failed to read native package entry")?;
        ensure!(
            entry.header().entry_type().is_file(),
            "native package contains a non-file entry"
        );
        let path = entry
            .path()
            .context("invalid native package path")?
            .to_string_lossy()
            .replace('\\', "/");
        ensure!(
            !files.contains_key(&path),
            "native package contains duplicate path {path}"
        );
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .with_context(|| format!("failed to read {path}"))?;
        files.insert(path, bytes);
    }

    let prefix = &contract.dictionary_dir;
    let expected_paths = BTreeSet::from([
        format!("{prefix}/AUTHORS"),
        format!("{prefix}/BSD"),
        format!("{prefix}/NOTICE"),
        format!("{prefix}/system.dic.zst"),
        format!("{prefix}/manifest.json"),
        format!("{prefix}/SHA256SUMS"),
    ]);
    ensure!(
        files.keys().cloned().collect::<BTreeSet<_>>() == expected_paths,
        "native package member set mismatch"
    );
    let manifest_path = format!("{prefix}/manifest.json");
    let manifest: NativeManifest = serde_json::from_slice(&files[&manifest_path])
        .context("invalid native package manifest")?;
    ensure!(
        manifest.schema_version == 1,
        "unsupported native manifest schema"
    );
    ensure!(
        manifest.dictionary_id == contract.dictionary_dir,
        "native dictionary ID mismatch"
    );
    ensure!(
        manifest.representation == "compact-raw",
        "native dictionary representation must be compact-raw"
    );
    ensure!(
        manifest.feature_encoding == "[PP][S][F]",
        "native feature encoding mismatch"
    );
    let expected_file_names = BTreeSet::from([
        "AUTHORS".to_owned(),
        "BSD".to_owned(),
        "NOTICE".to_owned(),
        "system.dic.zst".to_owned(),
    ]);
    ensure!(
        manifest.files.keys().cloned().collect::<BTreeSet<_>>() == expected_file_names,
        "native manifest file set mismatch"
    );
    for (name, expected) in &manifest.files {
        let actual = bytes_record(&files[&format!("{prefix}/{name}")]);
        ensure!(
            actual == *expected,
            "native package {name} size or SHA-256 mismatch"
        );
    }
    let checksums_path = format!("{prefix}/SHA256SUMS");
    ensure!(
        files[&checksums_path] == checksum_text(&manifest.files).as_bytes(),
        "native package SHA256SUMS does not match its manifest"
    );
    let compressed_path = format!("{prefix}/system.dic.zst");
    let actual_expanded =
        expanded_dictionary_record_from_bytes(&files[&compressed_path], &compressed_path)?;
    ensure!(
        actual_expanded == manifest.expanded_dictionary,
        "expanded dictionary size or SHA-256 does not match packaged system.dic.zst"
    );
    Ok(manifest)
}
