use anyhow::{Result, bail};
use cpal::{
    Device, HostId, SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait},
};
use serde::{Deserialize, Serialize};

use crate::config::ParapperConfig;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceInfo {
    pub id: String,
    pub host: String,
    pub display_name: String,
    pub channels: u16,
    pub sample_rate: u32,
}

pub(crate) struct InputDeviceSelection {
    pub device: Device,
    pub stream_config: StreamConfig,
    pub sample_format: SampleFormat,
    pub device_info: DeviceInfo,
}

pub fn collect_input_devices() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();
    for host_id in available_non_asio_hosts() {
        let Ok(host) = cpal::host_from_id(host_id) else {
            continue;
        };
        let Ok(input_devices) = host.input_devices() else {
            continue;
        };

        for device in input_devices {
            let Some(device_info) = device_info(host_id, &device) else {
                continue;
            };
            devices.push(device_info);
        }
    }

    devices
}

pub(crate) fn selected_input_device(config: &ParapperConfig) -> Result<InputDeviceSelection> {
    if let (Some(host), Some(id)) = (
        config.input_device_host.as_deref(),
        config.input_device_id.as_deref(),
    ) && let Some(selected) = find_input_device(host, id)?
    {
        return Ok(selected);
    }

    default_input_device()
}

fn find_input_device(host_name: &str, device_id: &str) -> Result<Option<InputDeviceSelection>> {
    for host_id in available_non_asio_hosts() {
        if host_name_from_id(host_id) != host_name {
            continue;
        }
        let host = cpal::host_from_id(host_id)?;
        let input_devices = host.input_devices()?;
        for device in input_devices {
            let Some(device_info) = device_info(host_id, &device) else {
                continue;
            };
            if device_info.id == device_id {
                let default_config = device.default_input_config()?;
                return Ok(Some(InputDeviceSelection {
                    device,
                    stream_config: default_config.config(),
                    sample_format: default_config.sample_format(),
                    device_info,
                }));
            }
        }
    }

    Ok(None)
}

fn default_input_device() -> Result<InputDeviceSelection> {
    for host_id in available_non_asio_hosts() {
        let host = cpal::host_from_id(host_id)?;
        let Some(device) = host.default_input_device() else {
            continue;
        };
        let Some(device_info) = device_info(host_id, &device) else {
            continue;
        };
        let default_config = device.default_input_config()?;
        return Ok(InputDeviceSelection {
            device,
            stream_config: default_config.config(),
            sample_format: default_config.sample_format(),
            device_info,
        });
    }

    bail!("No input audio device is available")
}

fn device_info(host_id: HostId, device: &Device) -> Option<DeviceInfo> {
    let default_config = device.default_input_config().ok()?;
    let stream_config = default_config.config();
    Some(DeviceInfo {
        id: device_id(device),
        host: host_name_from_id(host_id),
        display_name: device_name(device),
        channels: stream_config.channels,
        sample_rate: stream_config.sample_rate,
    })
}

fn available_non_asio_hosts() -> impl Iterator<Item = HostId> {
    cpal::available_hosts()
        .into_iter()
        .filter(|host| host_name_from_id(*host) != "Asio")
}

fn host_name_from_id(host_id: HostId) -> String {
    format!("{host_id:?}")
}

fn device_id(device: &Device) -> String {
    device
        .id()
        .map_or_else(|_| device_name(device), |id| id.to_string())
}

fn device_name(device: &Device) -> String {
    if let Ok(description) = device.description() {
        if let Some(extended_name) = description.extended().first()
            && !extended_name.trim().is_empty()
        {
            return normalize_device_name(extended_name);
        }

        let name = description.name();

        #[cfg(target_os = "windows")]
        {
            if let Some(manufacturer) = description.manufacturer() {
                return normalize_device_name(&format!("{name} ({manufacturer})"));
            }
            if let Some(driver) = description.driver() {
                return normalize_device_name(&format!("{name} ({driver})"));
            }
        }

        return normalize_device_name(name);
    }

    #[allow(deprecated)]
    device.name().map_or_else(
        |_| "Unknown Device".to_string(),
        |name| normalize_device_name(&name),
    )
}

#[cfg(target_os = "windows")]
fn normalize_device_name(name: &str) -> String {
    if name.contains(" - ") {
        name.replace(" - ", " (") + ")"
    } else {
        name.to_string()
    }
}

#[cfg(not(target_os = "windows"))]
fn normalize_device_name(name: &str) -> String {
    name.to_string()
}
