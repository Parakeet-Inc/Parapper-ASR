//! JVS + sherpa-onnx の ASR engine 精度を検証する ignored test。
//!
//! ここでは `verify_jvs_asr` 診断コマンドを起動し、ReazonSpeech 単体の認識精度と
//! 連結 wav の ASR 結果を確認する。`TurnRuntime` の segment / interim / final /
//! UI event 順序を検証する統合テストではない。

use std::{path::PathBuf, process::Command};

#[path = "../../diagnostics/src/verify_jvs_asr_constants.rs"]
mod verify_jvs_asr_constants;

use verify_jvs_asr_constants::LEGACY_INTERIM_SUMMARY_PREFIX;

#[test]
#[ignore = "requires local JVS corpus and downloaded ReazonSpeech model"]
fn jvs_nonparallel_wav_is_recognized_close_to_reference_text() {
    let Some(jvs_root) = existing_jvs_root_from_env() else {
        return;
    };

    let output = verify_jvs_asr_command()
        .arg("--jvs-root")
        .arg(&jvs_root)
        .arg("--max-speakers")
        .arg("1")
        .arg("--max-utterances-per-speaker")
        .arg("5")
        .arg("--max-cer")
        .arg("0.15")
        .arg("--min-exact-rate")
        .arg("0.0")
        .output()
        .expect("failed to run verify_jvs_asr");

    assert!(
        output.status.success(),
        "verify_jvs_asr failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires local JVS corpus and downloaded ReazonSpeech model"]
fn jvs_nonparallel_concatenated_wavs_keep_each_component_visible_after_internal_silence() {
    let Some(jvs_root) = existing_jvs_root_from_env() else {
        return;
    };

    let output = verify_jvs_asr_command()
        .arg("--jvs-root")
        .arg(&jvs_root)
        .arg("--max-speakers")
        .arg("1")
        .arg("--ids")
        .arg("BASIC5000_0408,BASIC5000_0518")
        .arg("--concat-size")
        .arg("2")
        .arg("--concat-silence-ms")
        .arg("200")
        .arg("--max-cer")
        .arg("0.15")
        .arg("--min-exact-rate")
        .arg("0.0")
        .arg("--verbose")
        .output()
        .expect("failed to run verify_jvs_asr");

    assert!(
        output.status.success(),
        "verify_jvs_asr failed for concatenated JVS wavs\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires local JVS corpus and downloaded ReazonSpeech model"]
fn jvs_concat_interim_like_silence_reports_rerecognition_cer_for_continuous_and_legacy_audio() {
    let Some(jvs_root) = existing_jvs_root_from_env() else {
        return;
    };

    let output = verify_jvs_asr_command()
        .arg("--jvs-root")
        .arg(&jvs_root)
        .arg("--max-speakers")
        .arg("1")
        .arg("--ids")
        .arg("BASIC5000_0408,BASIC5000_0518")
        .arg("--concat-size")
        .arg("2")
        .arg("--concat-silence-ms")
        .arg("128")
        .arg("--production-asr-padding")
        .arg("--compare-interim-rerecognition")
        .arg("--interim-result-silence-ms")
        .arg("64")
        .arg("--turn-check-silence-ms")
        .arg("192")
        .arg("--max-cer")
        .arg("0.15")
        .arg("--min-exact-rate")
        .arg("0.0")
        .arg("--verbose")
        .output()
        .expect("failed to run verify_jvs_asr");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "verify_jvs_asr failed for interim-like JVS rerecognition comparison\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains(LEGACY_INTERIM_SUMMARY_PREFIX),
        "comparison output should include the legacy duplicated-padding CER summary\nstdout:\n{stdout}"
    );
}

fn existing_jvs_root_from_env() -> Option<PathBuf> {
    let Some(jvs_root) = std::env::var_os("JVS_ROOT").map(PathBuf::from) else {
        eprintln!("skipping JVS ASR diagnostic: set JVS_ROOT to run this ignored test");
        return None;
    };
    if !jvs_root.is_dir() {
        eprintln!(
            "skipping JVS ASR diagnostic: JVS_ROOT does not exist or is not a directory: {}",
            jvs_root.display()
        );
        return None;
    }
    Some(jvs_root)
}

fn verify_jvs_asr_command() -> Command {
    let mut command = Command::new(env!("CARGO"));
    command
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("run")
        .arg("-p")
        .arg("parapper-diagnostics")
        .arg("--features")
        .arg("real-asr-tests")
        .arg("--bin")
        .arg("verify_jvs_asr")
        .arg("--");
    command
}
