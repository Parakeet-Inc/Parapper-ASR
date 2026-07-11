fn main() {
    copy_sherpa_onnx_runtime_dlls();
    copy_macos_sherpa_runtime_libraries();
    configure_macos_runtime_library_path();
    configure_windows_common_controls_manifest();
    tauri_build::try_build(tauri_build_attributes()).expect("failed to run tauri build helpers");
}

#[cfg(windows)]
fn copy_sherpa_onnx_runtime_dlls() {
    use std::{env, fs, path::PathBuf};

    const SHERPA_ONNX_VERSION: &str = "1.13.3";
    const SHERPA_PREBUILT_DIR: &str = "sherpa-onnx-v1.13.3-win-x64-shared-MT-Release-lib";

    let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") else {
        return;
    };
    let manifest_dir = PathBuf::from(manifest_dir);
    let Some(workspace_root) = manifest_dir.parent() else {
        return;
    };

    let source_dir = workspace_root
        .join("target")
        .join("sherpa-onnx-prebuilt")
        .join(SHERPA_PREBUILT_DIR)
        .join("lib");
    let Some(profile_dir) = target_profile_dir() else {
        return;
    };

    println!("cargo:rerun-if-changed={}", source_dir.display());
    if !source_dir.is_dir() {
        println!(
            "cargo:warning=sherpa-onnx {SHERPA_ONNX_VERSION} runtime DLL dir not found: {}",
            source_dir.display()
        );
        return;
    }

    let dlls: Vec<_> = match fs::read_dir(&source_dir) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "dll"))
            .collect(),
        Err(err) => {
            println!(
                "cargo:warning=failed to read sherpa-onnx runtime DLL dir {}: {err}",
                source_dir.display()
            );
            return;
        }
    };

    for dest_dir in [&profile_dir, &profile_dir.join("deps")] {
        if let Err(err) = fs::create_dir_all(dest_dir) {
            println!(
                "cargo:warning=failed to create runtime DLL dir {}: {err}",
                dest_dir.display()
            );
            continue;
        }
        for source in &dlls {
            let Some(dll_name) = source.file_name() else {
                continue;
            };
            let destination = dest_dir.join(dll_name);
            if let Err(err) = fs::copy(source, &destination) {
                println!(
                    "cargo:warning=failed to copy {} to {}: {err}",
                    source.display(),
                    destination.display()
                );
            }
        }
    }
}

#[cfg(windows)]
fn target_profile_dir() -> Option<std::path::PathBuf> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").ok()?);
    out_dir.ancestors().nth(3).map(std::path::Path::to_path_buf)
}

#[cfg(not(windows))]
fn copy_sherpa_onnx_runtime_dlls() {}

#[cfg(target_os = "macos")]
fn copy_macos_sherpa_runtime_libraries() {
    use std::{env, fs, path::PathBuf};

    const SHERPA_PREBUILT_DIR: &str = "sherpa-onnx-v1.13.3-osx-arm64-shared-lib";
    const LIBRARIES: &[&str] = &[
        "libsherpa-onnx-c-api.dylib",
        "libsherpa-onnx-cxx-api.dylib",
        "libonnxruntime.dylib",
        "libonnxruntime.1.24.4.dylib",
    ];

    let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") else {
        return;
    };
    let manifest_dir = PathBuf::from(manifest_dir);
    let Some(workspace_root) = manifest_dir.parent() else {
        return;
    };

    let runtime_dir = manifest_dir.join("macos-runtime");
    if let Err(err) = fs::create_dir_all(&runtime_dir) {
        println!(
            "cargo:warning=failed to create macOS runtime dir {}: {err}",
            runtime_dir.display()
        );
        return;
    }

    let target_triple = env::var("TARGET_TRIPLE")
        .ok()
        .or_else(|| env::var("CARGO_BUILD_TARGET").ok());
    let mut source_dirs = Vec::new();
    if let Some(target_triple) = target_triple {
        source_dirs.push(
            workspace_root
                .join("target")
                .join(&target_triple)
                .join("release"),
        );
        source_dirs.push(
            workspace_root
                .join("target")
                .join(target_triple)
                .join("debug"),
        );
    }
    source_dirs.push(workspace_root.join("target").join("release"));
    source_dirs.push(workspace_root.join("target").join("debug"));
    source_dirs.push(
        workspace_root
            .join("target")
            .join("sherpa-onnx-prebuilt")
            .join(SHERPA_PREBUILT_DIR)
            .join("lib"),
    );

    for library in LIBRARIES {
        let Some(source) = source_dirs
            .iter()
            .map(|dir| dir.join(library))
            .find(|path| path.is_file())
        else {
            println!("cargo:warning=macOS runtime library not found: {library}");
            continue;
        };
        let destination = runtime_dir.join(library);
        if let Err(err) = fs::copy(&source, &destination) {
            println!(
                "cargo:warning=failed to copy {} to {}: {err}",
                source.display(),
                destination.display()
            );
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn copy_macos_sherpa_runtime_libraries() {}

#[cfg(target_os = "macos")]
fn configure_macos_runtime_library_path() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Resources/macos-runtime");
}

#[cfg(not(target_os = "macos"))]
fn configure_macos_runtime_library_path() {}

#[cfg(windows)]
fn tauri_build_attributes() -> tauri_build::Attributes {
    tauri_build::Attributes::new()
        .windows_attributes(tauri_build::WindowsAttributes::new_without_app_manifest())
}

#[cfg(not(windows))]
fn tauri_build_attributes() -> tauri_build::Attributes {
    tauri_build::Attributes::new()
}

#[cfg(windows)]
fn configure_windows_common_controls_manifest() {
    use std::{env, fs};

    let Some(out_dir) = env::var_os("OUT_DIR") else {
        return;
    };
    let manifest_path = std::path::PathBuf::from(out_dir).join("common-controls-v6.manifest");
    let manifest = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency>
    <dependentAssembly>
      <assemblyIdentity
        type="win32"
        name="Microsoft.Windows.Common-Controls"
        version="6.0.0.0"
        processorArchitecture="*"
        publicKeyToken="6595b64144ccf1df"
        language="*" />
    </dependentAssembly>
  </dependency>
</assembly>
"#;
    if let Err(err) = fs::write(&manifest_path, manifest) {
        println!(
            "cargo:warning=failed to write Windows common-controls manifest {}: {err}",
            manifest_path.display()
        );
        return;
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
    println!(
        "cargo:rustc-link-arg=/MANIFESTINPUT:{}",
        manifest_path.display()
    );
}

#[cfg(not(windows))]
fn configure_windows_common_controls_manifest() {}
