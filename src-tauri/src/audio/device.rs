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
}

pub(crate) struct OutputDeviceSelection {
    pub device: Device,
}

pub fn collect_input_devices() -> Vec<DeviceInfo> {
    collect_devices(DeviceDirection::Input)
}

pub fn collect_output_devices() -> Vec<DeviceInfo> {
    collect_devices(DeviceDirection::Output)
}

fn collect_devices(direction: DeviceDirection) -> Vec<DeviceInfo> {
    let mut devices = Vec::new();
    for host_id in available_non_asio_hosts() {
        let Ok(host) = cpal::host_from_id(host_id) else {
            continue;
        };
        let Ok(host_devices) = direction.devices(&host) else {
            continue;
        };

        for device in host_devices {
            let Some(device_info) = device_info(host_id, &device, direction) else {
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

pub(crate) fn selected_output_device_by_id(
    host: Option<&str>,
    id: Option<&str>,
) -> Result<OutputDeviceSelection> {
    if let (Some(host), Some(id)) = (host, id)
        && let Some(selected) = find_output_device(host, id)?
    {
        return Ok(selected);
    }

    default_output_device()
}

fn find_input_device(host_name: &str, device_id: &str) -> Result<Option<InputDeviceSelection>> {
    for host_id in available_non_asio_hosts() {
        if host_name_from_id(host_id) != host_name {
            continue;
        }
        let host = cpal::host_from_id(host_id)?;
        let input_devices = host.input_devices()?;
        for device in input_devices {
            let Some(device_info) = device_info(host_id, &device, DeviceDirection::Input) else {
                continue;
            };
            if device_info.id == device_id {
                let default_config = device.default_input_config()?;
                return Ok(Some(InputDeviceSelection {
                    device,
                    stream_config: default_config.config(),
                    sample_format: default_config.sample_format(),
                }));
            }
        }
    }

    Ok(None)
}

fn find_output_device(host_name: &str, device_id: &str) -> Result<Option<OutputDeviceSelection>> {
    for host_id in available_non_asio_hosts() {
        if host_name_from_id(host_id) != host_name {
            continue;
        }
        let host = cpal::host_from_id(host_id)?;
        let output_devices = host.output_devices()?;
        for device in output_devices {
            let Some(device_info) = device_info(host_id, &device, DeviceDirection::Output) else {
                continue;
            };
            if device_info.id == device_id {
                return Ok(Some(OutputDeviceSelection { device }));
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
        if device_info(host_id, &device, DeviceDirection::Input).is_none() {
            continue;
        }
        let default_config = device.default_input_config()?;
        return Ok(InputDeviceSelection {
            device,
            stream_config: default_config.config(),
            sample_format: default_config.sample_format(),
        });
    }

    bail!("No input audio device is available")
}

fn default_output_device() -> Result<OutputDeviceSelection> {
    for host_id in available_non_asio_hosts() {
        let host = cpal::host_from_id(host_id)?;
        let Some(device) = host.default_output_device() else {
            continue;
        };
        if device_info(host_id, &device, DeviceDirection::Output).is_none() {
            continue;
        }
        return Ok(OutputDeviceSelection { device });
    }

    bail!("No output audio device is available")
}

fn device_info(host_id: HostId, device: &Device, direction: DeviceDirection) -> Option<DeviceInfo> {
    let default_config = direction.default_config(device)?;
    let stream_config = default_config.config();
    Some(DeviceInfo {
        id: device_id(device),
        host: host_name_from_id(host_id),
        display_name: device_name(device),
        channels: stream_config.channels,
        sample_rate: stream_config.sample_rate,
    })
}

#[derive(Debug, Clone, Copy)]
enum DeviceDirection {
    Input,
    Output,
}

impl DeviceDirection {
    fn devices(self, host: &cpal::Host) -> Result<Box<dyn Iterator<Item = Device>>> {
        match self {
            Self::Input => Ok(Box::new(host.input_devices()?)),
            Self::Output => Ok(Box::new(host.output_devices()?)),
        }
    }

    fn default_config(self, device: &Device) -> Option<cpal::SupportedStreamConfig> {
        match self {
            Self::Input => device.default_input_config().ok(),
            Self::Output => device.default_output_config().ok(),
        }
    }
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
