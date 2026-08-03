use std::{
    fs,
    io::{Cursor, Read, Write},
};

use parapper_morph_dictionary::morph_dictionary_release::{
    ArtifactSource, ReleaseContract, package_release, verify_native_package,
    verify_official_source, verify_release,
};
use tempfile::tempdir;
use vibrato_rkyv::{Dictionary, SystemDictionaryBuilder};

fn release_contract() -> ReleaseContract {
    ReleaseContract {
        release_tag: "morph-dictionary-test-v1".into(),
        native_asset_name: "unidic-test.tar.zst".into(),
        dictionary_dir: "unidic-test".into(),
    }
}

fn write_source_fixture(source: &ArtifactSource) {
    fs::create_dir_all(&source.directory).unwrap();
    let inner = SystemDictionaryBuilder::from_readers(
        "東京,1,1,1,0100\n".as_bytes(),
        "2 2\n0 0 0\n0 1 0\n1 0 0\n1 1 0\n".as_bytes(),
        "DEFAULT 0 1 0\n".as_bytes(),
        "DEFAULT,1,1,100,0100\n".as_bytes(),
    )
    .unwrap();
    let dictionary = Dictionary::from_inner(inner);
    let output = fs::File::create(source.native_dictionary()).unwrap();
    let mut encoder = zstd::Encoder::new(output, 1).unwrap();
    dictionary.write(&mut encoder).unwrap();
    encoder.finish().unwrap().flush().unwrap();
    fs::write(source.authors(), b"AUTHORS fixture").unwrap();
    fs::write(source.bsd_license(), b"BSD fixture").unwrap();
    fs::write(source.notice(), b"NOTICE fixture").unwrap();
}

fn rewrite_expanded_dictionary_record(package: &std::path::Path) {
    let input = fs::File::open(package).unwrap();
    let decoder = zstd::Decoder::new(input).unwrap();
    let mut archive = tar::Archive::new(decoder);
    let mut files = Vec::new();
    for entry in archive.entries().unwrap() {
        let mut entry = entry.unwrap();
        let path = entry.path().unwrap().into_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        if path.ends_with("manifest.json") {
            let mut manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            manifest["expanded_dictionary"] = serde_json::json!({
                "size": 123,
                "sha256": "00".repeat(32),
            });
            bytes = serde_json::to_vec_pretty(&manifest).unwrap();
            bytes.push(b'\n');
        }
        files.push((path, bytes));
    }

    let output = fs::File::create(package).unwrap();
    let encoder = zstd::Encoder::new(output, 1).unwrap();
    let mut archive = tar::Builder::new(encoder);
    for (path, bytes) in files {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, path, Cursor::new(bytes))
            .unwrap();
    }
    let encoder = archive.into_inner().unwrap();
    encoder.finish().unwrap();
}

#[test]
fn package_release_for_v0_4_0_does_not_require_or_publish_a_wasm_dictionary() {
    let dir = tempdir().unwrap();
    let source = ArtifactSource::new(dir.path().join("source"));
    write_source_fixture(&source);
    fs::write(
        source.directory.join("system.wasm.dic.gz"),
        b"future v0.5.0 artifact",
    )
    .unwrap();
    let release_dir = dir.path().join("release");

    package_release(&source, &release_dir, &release_contract())
        .expect("the v0.4.0 dictionary Release is native-only");

    assert_eq!(
        fs::read_dir(&release_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([
            "AUTHORS".to_owned(),
            "BSD".to_owned(),
            "NOTICE".to_owned(),
            "SHA256SUMS".to_owned(),
            "release-manifest.json".to_owned(),
            "unidic-test.tar.zst".to_owned(),
        ])
    );
    assert_eq!(
        fs::read(release_dir.join("AUTHORS")).unwrap(),
        b"AUTHORS fixture",
        "the published attribution must preserve the source bytes exactly"
    );
}

#[test]
fn package_release_rejects_a_missing_unidic_authors_attribution_file() {
    let dir = tempdir().unwrap();
    let source = ArtifactSource::new(dir.path().join("source"));
    write_source_fixture(&source);
    fs::remove_file(source.authors()).unwrap();

    let error = package_release(&source, dir.path().join("release"), &release_contract())
        .expect_err("a UniDic derivative must retain the AUTHORS attribution file");

    assert!(
        error.to_string().contains("AUTHORS"),
        "the failure must identify the missing attribution file: {error:#}"
    );
}

#[test]
fn official_source_rejects_unidic_authors_with_an_added_trailing_newline() {
    let dir = tempdir().unwrap();
    let source = ArtifactSource::new(dir.path().join("source"));
    fs::create_dir_all(&source.directory).unwrap();
    fs::write(source.authors(), b"The UniDic Consortium,\nTeruaki Oka\n").unwrap();

    let error = verify_official_source(&source)
        .expect_err("the audited attribution must match the official 34 bytes exactly");

    assert!(
        error.to_string().contains("AUTHORS size mismatch"),
        "the failure must identify source attribution drift: {error:#}"
    );
}

#[test]
fn verify_release_rejects_an_asset_whose_bytes_no_longer_match_the_manifest() {
    let dir = tempdir().unwrap();
    let source = ArtifactSource::new(dir.path().join("source"));
    write_source_fixture(&source);
    let release_dir = dir.path().join("release");
    package_release(&source, &release_dir, &release_contract()).unwrap();

    fs::write(release_dir.join("BSD"), b"bad fixture").unwrap();

    let error = verify_release(&release_dir, &release_contract())
        .expect_err("a hash mismatch must fail before publication or installation");
    assert!(
        error.to_string().contains("SHA-256 mismatch"),
        "the failure must distinguish an integrity mismatch: {error:#}"
    );
}

#[test]
fn verify_release_rejects_a_native_package_with_tampered_internal_content() {
    let dir = tempdir().unwrap();
    let source = ArtifactSource::new(dir.path().join("source"));
    write_source_fixture(&source);
    let release_dir = dir.path().join("release");
    package_release(&source, &release_dir, &release_contract()).unwrap();

    let native_package = release_dir.join("unidic-test.tar.zst");
    let bytes = fs::read(&native_package).unwrap();
    let truncated = &bytes[..bytes.len() - 1];
    fs::write(&native_package, truncated).unwrap();

    let error = verify_native_package(&native_package, &release_contract())
        .expect_err("a damaged native package must never pass verification");
    assert!(
        error.to_string().contains("failed")
            || error.to_string().contains("incomplete")
            || error.to_string().contains("corrupt"),
        "the failure must report damaged package content: {error:#}"
    );
}

#[test]
fn verify_native_package_rejects_an_expanded_record_not_derived_from_its_dictionary() {
    let dir = tempdir().unwrap();
    let source = ArtifactSource::new(dir.path().join("source"));
    write_source_fixture(&source);
    let release_dir = dir.path().join("release");
    package_release(&source, &release_dir, &release_contract()).unwrap();

    let native_package = release_dir.join("unidic-test.tar.zst");
    rewrite_expanded_dictionary_record(&native_package);

    let error = verify_native_package(&native_package, &release_contract())
        .expect_err("the expanded record must be recomputed from packaged system.dic.zst");
    assert!(
        error.to_string().contains("expanded"),
        "the failure must identify the unbound expanded dictionary record: {error:#}"
    );
}
