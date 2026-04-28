#[must_use]
pub fn find_current_project_dir() -> Option<std::path::PathBuf> {
    let mut current_dir = std::env::current_dir().ok()?;

    loop {
        if current_dir.join("package.json").exists() {
            return Some(current_dir);
        }
        if !current_dir.pop() {
            break;
        }
    }

    None
}
