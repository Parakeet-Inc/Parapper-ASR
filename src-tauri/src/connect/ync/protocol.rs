use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub(super) struct CommandRequest<T> {
    pub(super) operation: &'static str,
    pub(super) params: Vec<T>,
}

#[derive(Debug, Serialize)]
pub(super) struct IdParams<'a> {
    pub(super) id: &'a str,
}

#[derive(Debug, Deserialize)]
pub(super) struct RawVersionResponse {
    pub(super) operation: String,
    pub(super) status: String,
}

pub(super) fn ensure_success(
    operation: &str,
    status: &str,
    expected_operation: &str,
) -> Result<()> {
    if operation != expected_operation {
        return Err(anyhow!(
            "YNC plugin returned operation {operation}, expected {expected_operation}"
        ));
    }
    if status != "success" {
        return Err(anyhow!(
            "YNC plugin command {operation} returned status {status}"
        ));
    }
    Ok(())
}

pub(super) fn ensure_sended(operation: &str, status: &str, expected_operation: &str) -> Result<()> {
    ensure_status(operation, status, expected_operation, &["sended"])
}

pub(super) fn ensure_sended_or_success(
    operation: &str,
    status: &str,
    expected_operation: &str,
) -> Result<()> {
    ensure_status(
        operation,
        status,
        expected_operation,
        &["sended", "success"],
    )
}

fn ensure_status(
    operation: &str,
    status: &str,
    expected_operation: &str,
    allowed_statuses: &[&str],
) -> Result<()> {
    if operation != expected_operation {
        return Err(anyhow!(
            "YNC plugin returned operation {operation}, expected {expected_operation}"
        ));
    }
    if !allowed_statuses.contains(&status) {
        return Err(anyhow!(
            "YNC plugin command {operation} returned status {status}, expected one of {}",
            allowed_statuses.join(", ")
        ));
    }
    Ok(())
}
