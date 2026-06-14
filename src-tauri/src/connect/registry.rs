#[cfg(windows)]
use std::{os::windows::process::CommandExt, process::Command};

#[cfg(windows)]
pub(crate) fn detect_dword_value_u16(subkey: &str, value_name: &str) -> Option<u16> {
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let mut command = Command::new("reg");
    command
        .args(["query", subkey, "/v", value_name])
        .creation_flags(CREATE_NO_WINDOW);
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(|line| parse_reg_dword(line, value_name))
}

#[cfg(windows)]
fn parse_reg_dword(line: &str, value_name: &str) -> Option<u16> {
    let mut parts = line.split_whitespace();
    if !parts.any(|part| part == value_name) {
        return None;
    }
    let value = parts.find(|part| part.starts_with("0x"))?;
    u16::from_str_radix(value.trim_start_matches("0x"), 16).ok()
}

#[cfg(test)]
#[cfg(windows)]
mod tests {
    #[test]
    fn parse_reg_dword_reads_named_port() {
        let line = "    HTTP    REG_DWORD    0x3ca0";

        assert_eq!(super::parse_reg_dword(line, "HTTP"), Some(15520));
        assert_eq!(super::parse_reg_dword(line, "Other"), None);
    }
}
