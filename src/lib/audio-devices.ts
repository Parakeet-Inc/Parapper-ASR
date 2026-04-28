import type { AudioDeviceInfo } from "./types";

export type AudioDeviceOptionGroup = {
  group: string;
  items: { label: string; value: string }[];
};

const preferredHostOrder = ["Wasapi", "CoreAudio", "Null"];

export const buildAudioDeviceOptions = (
  audioDevices: AudioDeviceInfo[],
): AudioDeviceOptionGroup[] => {
  const knownHosts = new Set(preferredHostOrder);
  const hosts = [
    ...preferredHostOrder,
    ...Array.from(
      new Set(
        audioDevices
          .map((device) => device.host)
          .filter((host) => !knownHosts.has(host)),
      ),
    ).sort(),
  ];

  return hosts
    .map((host) => ({
      group: host,
      items: audioDevices
        .filter((device) => device.host === host)
        .map((device) => ({
          label: device.display_name,
          value: `${device.host}\u0000${device.id}`,
        })),
    }))
    .filter((group) => group.items.length > 0);
};
