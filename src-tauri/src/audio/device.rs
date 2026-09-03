use anyhow::{Context, Result, bail};
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
    let mut devices = collect_devices(DeviceDirection::Input);
    if LOOPBACK_SUPPORTED {
        // Output devices are exposed as additional "loopback" sources so the user can
        // transcribe whatever is playing through the speakers. cpal records them by
        // building an input stream on the output device (a CoreAudio process tap on
        // macOS, WASAPI loopback on Windows).
        devices.extend(collect_loopback_devices());
    }
    devices
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
        config.input.device_host.as_deref(),
        config.input.device_id.as_deref(),
    ) {
        if let Some(real_host) = loopback_source_host(host) {
            if LOOPBACK_SUPPORTED && let Some(selected) = find_loopback_device(real_host, id)? {
                return Ok(selected);
            }
        } else if let Some(selected) = find_input_device(host, id)? {
            return Ok(selected);
        }
    }

    default_input_device()
}

/// Resolves an explicitly configured input endpoint without falling back to a
/// default device or another capture mode.
pub(crate) fn selected_input_device_strict(host: &str, id: &str) -> Result<InputDeviceSelection> {
    resolve_input_device_strict_with(
        host,
        id,
        LOOPBACK_SUPPORTED,
        find_input_device,
        find_loopback_device,
    )
}

fn resolve_input_device_strict_with<T, FindPhysical, FindLoopback>(
    host: &str,
    id: &str,
    loopback_supported: bool,
    mut find_physical: FindPhysical,
    mut find_loopback: FindLoopback,
) -> Result<T>
where
    FindPhysical: FnMut(&str, &str) -> Result<Option<T>>,
    FindLoopback: FnMut(&str, &str) -> Result<Option<T>>,
{
    let selected = if let Some(real_host) = loopback_source_host(host) {
        if !loopback_supported {
            bail!(
                "explicit input endpoint uses unsupported loopback capture: host={host}, id={id}"
            );
        }
        find_loopback(real_host, id)
    } else {
        find_physical(host, id)
    }
    .with_context(|| format!("failed to resolve explicit input endpoint: host={host}, id={id}"))?;

    selected.ok_or_else(|| {
        anyhow::anyhow!("explicit input endpoint was not found: host={host}, id={id}")
    })
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

/// Loopback (system-audio) capture is only wired up on platforms where cpal records an
/// output device by building an input stream on it: a `CoreAudio` process tap on macOS
/// (requires macOS 14.4+) and `WASAPI` loopback on Windows.
#[cfg(any(target_os = "macos", target_os = "windows"))]
const LOOPBACK_SUPPORTED: bool = true;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const LOOPBACK_SUPPORTED: bool = false;

/// Suffix appended to a host name to mark a device entry as a loopback capture source.
/// A real host name never contains it, so it round-trips unambiguously through the saved
/// `input_device_host`/`input_device_id` config and back into device resolution.
const LOOPBACK_HOST_SUFFIX: &str = " (Loopback)";

fn loopback_host_label(real_host: &str) -> String {
    format!("{real_host}{LOOPBACK_HOST_SUFFIX}")
}

/// Returns the underlying host name if `host_label` refers to a loopback source.
fn loopback_source_host(host_label: &str) -> Option<&str> {
    host_label.strip_suffix(LOOPBACK_HOST_SUFFIX)
}

fn collect_loopback_devices() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();
    for host_id in available_non_asio_hosts() {
        let Ok(host) = cpal::host_from_id(host_id) else {
            continue;
        };
        let Ok(output_devices) = host.output_devices() else {
            continue;
        };
        for device in output_devices {
            let Some(device_info) = loopback_device_info(host_id, &device) else {
                continue;
            };
            devices.push(device_info);
        }
    }

    devices
}

fn loopback_device_info(host_id: HostId, device: &Device) -> Option<DeviceInfo> {
    let default_config = device.default_output_config().ok()?;
    let stream_config = default_config.config();
    Some(DeviceInfo {
        id: device_id(device),
        host: loopback_host_label(&host_name_from_id(host_id)),
        display_name: device_name(device),
        channels: stream_config.channels,
        sample_rate: stream_config.sample_rate,
    })
}

fn find_loopback_device(host_name: &str, device_id: &str) -> Result<Option<InputDeviceSelection>> {
    for host_id in available_non_asio_hosts() {
        if host_name_from_id(host_id) != host_name {
            continue;
        }
        let host = cpal::host_from_id(host_id)?;
        for device in host.output_devices()? {
            let Some(device_info) = loopback_device_info(host_id, &device) else {
                continue;
            };
            if device_info.id == device_id {
                // A CoreAudio process tap records silence without the System Audio
                // Recording permission, so require it before starting capture.
                super::ensure_system_audio_permission()?;
                return Ok(Some(loopback_selection(device)?));
            }
        }
    }

    Ok(None)
}

/// Builds an input selection that captures an output device's audio via loopback.
///
/// The source device is the output device itself; cpal transparently creates the
/// process tap / loopback stream when an input stream is built on it. Because the
/// device has no input configuration, the stream is driven by its output format
/// (`default_output_config`).
fn loopback_selection(device: Device) -> Result<InputDeviceSelection> {
    let default_config = device.default_output_config()?;
    Ok(InputDeviceSelection {
        stream_config: default_config.config(),
        sample_format: default_config.sample_format(),
        device,
    })
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::resolve_input_device_strict_with;

    #[test]
    fn strict_resolver_returns_exact_physical_match_without_trying_loopback() {
        let physical_calls = Cell::new(0);
        let loopback_calls = Cell::new(0);

        let selected = resolve_input_device_strict_with(
            "WASAPI",
            "mic-1",
            true,
            |host, id| {
                physical_calls.set(physical_calls.get() + 1);
                assert_eq!((host, id), ("WASAPI", "mic-1"));
                Ok(Some("physical"))
            },
            |_host, _id| {
                loopback_calls.set(loopback_calls.get() + 1);
                Ok(Some("loopback"))
            },
        )
        .expect("exact physical endpoint should resolve");

        assert_eq!(selected, "physical");
        assert_eq!(physical_calls.get(), 1);
        assert_eq!(loopback_calls.get(), 0);
    }

    #[test]
    fn strict_resolver_reports_physical_miss_without_trying_another_resolver() {
        let physical_calls = Cell::new(0);
        let loopback_calls = Cell::new(0);

        let error = resolve_input_device_strict_with::<(), _, _>(
            "WASAPI",
            "missing-mic",
            true,
            |_host, _id| {
                physical_calls.set(physical_calls.get() + 1);
                Ok(None)
            },
            |_host, _id| {
                loopback_calls.set(loopback_calls.get() + 1);
                Ok(None)
            },
        )
        .expect_err("a missing explicit endpoint must fail");

        assert_eq!(
            error.to_string(),
            "explicit input endpoint was not found: host=WASAPI, id=missing-mic"
        );
        assert_eq!(physical_calls.get(), 1);
        assert_eq!(loopback_calls.get(), 0);
    }

    #[test]
    fn strict_resolver_returns_exact_loopback_match_without_trying_physical() {
        let physical_calls = Cell::new(0);
        let loopback_calls = Cell::new(0);

        let selected = resolve_input_device_strict_with(
            "WASAPI (Loopback)",
            "speakers-1",
            true,
            |_host, _id| {
                physical_calls.set(physical_calls.get() + 1);
                Ok(Some("physical"))
            },
            |host, id| {
                loopback_calls.set(loopback_calls.get() + 1);
                assert_eq!((host, id), ("WASAPI", "speakers-1"));
                Ok(Some("loopback"))
            },
        )
        .expect("exact loopback endpoint should resolve");

        assert_eq!(selected, "loopback");
        assert_eq!(physical_calls.get(), 0);
        assert_eq!(loopback_calls.get(), 1);
    }

    #[test]
    fn strict_resolver_reports_loopback_miss_without_trying_physical() {
        let physical_calls = Cell::new(0);
        let loopback_calls = Cell::new(0);

        let error = resolve_input_device_strict_with::<(), _, _>(
            "WASAPI (Loopback)",
            "missing-speakers",
            true,
            |_host, _id| {
                physical_calls.set(physical_calls.get() + 1);
                Ok(None)
            },
            |_host, _id| {
                loopback_calls.set(loopback_calls.get() + 1);
                Ok(None)
            },
        )
        .expect_err("a missing explicit loopback endpoint must fail");

        assert_eq!(
            error.to_string(),
            "explicit input endpoint was not found: host=WASAPI (Loopback), id=missing-speakers"
        );
        assert_eq!(physical_calls.get(), 0);
        assert_eq!(loopback_calls.get(), 1);
    }

    #[test]
    fn strict_resolver_rejects_unsupported_loopback_without_calling_resolvers() {
        let physical_calls = Cell::new(0);
        let loopback_calls = Cell::new(0);

        let error = resolve_input_device_strict_with::<(), _, _>(
            "ALSA (Loopback)",
            "speakers-1",
            false,
            |_host, _id| {
                physical_calls.set(physical_calls.get() + 1);
                Ok(None)
            },
            |_host, _id| {
                loopback_calls.set(loopback_calls.get() + 1);
                Ok(None)
            },
        )
        .expect_err("unsupported loopback must fail explicitly");

        assert_eq!(
            error.to_string(),
            "explicit input endpoint uses unsupported loopback capture: host=ALSA (Loopback), id=speakers-1"
        );
        assert_eq!(physical_calls.get(), 0);
        assert_eq!(loopback_calls.get(), 0);
    }
}
