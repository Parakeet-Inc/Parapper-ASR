use std::process::Command;

use scripts::find_current_project_dir;

fn main() {
    let current_project_dir =
        find_current_project_dir().expect("Failed to find current project directory");
    let src_tauri_path = current_project_dir.join("src-tauri");
    let dest_path = current_project_dir.join("licenses/rust.json");

    let status = Command::new("cargo")
        .current_dir(&src_tauri_path)
        .arg("about")
        .arg("generate")
        .arg("--format")
        .arg("json")
        .arg("-o")
        .arg(dest_path)
        .status()
        .expect("Failed to execute cargo about command");
    assert!(
        status.success(),
        "cargo about command failed with status: {status}"
    );
}
