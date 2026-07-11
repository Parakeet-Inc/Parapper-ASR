import type { AudioDeviceInfo } from "./types";

export type AudioDeviceOptionGroup = {
  group: string;
  items: { label: string; value: string }[];
};

const preferredHostOrder = ["Wasapi", "CoreAudio", "Null"];

// Kept in sync with `LOOPBACK_HOST_SUFFIX` in src-tauri/src/audio/device.rs: the
// backend tags system-audio (loopback) capture sources by appending this suffix to
// the host name, and resolves them back by stripping it.
const LOOPBACK_HOST_SUFFIX = " (Loopback)";

export const isLoopbackHost = (host: string): boolean =>
  host.endsWith(LOOPBACK_HOST_SUFFIX);

const deviceOption = (device: AudioDeviceInfo) => ({
  label: device.display_name,
  value: `${device.host}\u0000${device.id}`,
});

export const buildAudioDeviceOptions = (
  audioDevices: AudioDeviceInfo[],
  loopbackGroupLabel?: string,
): AudioDeviceOptionGroup[] => {
  const inputDevices = audioDevices.filter(
    (device) => !isLoopbackHost(device.host),
  );
  const loopbackDevices = audioDevices.filter((device) =>
    isLoopbackHost(device.host),
  );

  const knownHosts = new Set(preferredHostOrder);
  const hosts = [
    ...preferredHostOrder,
    ...Array.from(
      new Set(
        inputDevices
          .map((device) => device.host)
          .filter((host) => !knownHosts.has(host)),
      ),
    ).sort(),
  ];

  const groups = hosts
    .map((host) => ({
      group: host,
      items: inputDevices
        .filter((device) => device.host === host)
        .map(deviceOption),
    }))
    .filter((group) => group.items.length > 0);

  if (loopbackDevices.length > 0) {
    // All loopback sources are collapsed under a single, human-friendly group
    // regardless of their underlying host (there is at most one per platform).
    groups.push({
      group: loopbackGroupLabel ?? "Loopback",
      items: loopbackDevices.map(deviceOption),
    });
  }

  return groups;
};
