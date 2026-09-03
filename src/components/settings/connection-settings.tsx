import {
  ActionIcon,
  Checkbox,
  ColorSwatch,
  Group,
  Select,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  buildAudioDeviceOptions,
  isLoopbackHost,
} from "../../lib/audio-devices";
import { STT_PROFILE_DISPLAY_COLOR_CSS } from "../../lib/stt-profile-colors";
import {
  availableInputChannelForProfile,
  buildInputChannelRows,
  resolveSttProfileNameEdit,
  STT_PROFILE_DISPLAY_COLORS,
} from "../../lib/stt-profiles";
import type {
  AudioDeviceInfo,
  SttProfileConfig,
  SttProfileInputConfig,
} from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

import IconRefresh from "~icons/material-symbols/refresh";

type ConnectionSettingsProps = {
  input: SttProfileInputConfig;
  profiles: readonly SttProfileConfig[];
  selectedProfileId: string;
  profile: SttProfileConfig;
  onUpdateProfile: (
    profileId: string,
    update: (profile: SttProfileConfig) => SttProfileConfig,
  ) => void;
  inputAudioDevices: readonly AudioDeviceInfo[];
  runtimeLocked: boolean;
  localAudioDevicesAvailable: boolean;
  refreshingAudioDevices: boolean;
  onRefreshAudioDevices: () => void;
  onRequestLoopbackPermission: () => Promise<void>;
  onChange: (input: SttProfileInputConfig) => void;
};

export const ConnectionSettings: React.FC<ConnectionSettingsProps> = ({
  input,
  profiles,
  selectedProfileId,
  profile,
  onUpdateProfile,
  inputAudioDevices,
  runtimeLocked,
  localAudioDevicesAvailable,
  refreshingAudioDevices,
  onRefreshAudioDevices,
  onRequestLoopbackPermission,
  onChange,
}) => {
  const { t } = useTranslation();
  const displayColor = profile.display_color;
  const profileName =
    profile.name === profile.id
      ? t("sttProfiles.defaultName", {
          number: Number(profile.id.replace("stt-profile-", "")),
        })
      : profile.name;
  const [nameDraft, setNameDraft] = useState(profileName);
  useEffect(() => setNameDraft(profileName), [profileName]);
  const commitName = () => {
    const name = resolveSttProfileNameEdit(
      profiles,
      profile,
      nameDraft,
      (number) => t("sttProfiles.defaultName", { number }),
    );
    if (!name) {
      setNameDraft(profileName);
      return;
    }
    onUpdateProfile(profile.id, (current) => ({ ...current, name }));
  };
  const inputAudioDeviceOptions = useMemo(
    () =>
      buildAudioDeviceOptions(
        [...inputAudioDevices],
        t("settings.inputAudioDevice.loopbackGroup"),
      ),
    [inputAudioDevices, t],
  );
  const selectedDevice =
    input.device_host && input.device_id
      ? `${input.device_host}\u0000${input.device_id}`
      : null;
  const device = inputAudioDevices.find(
    (candidate) =>
      candidate.host === input.device_host && candidate.id === input.device_id,
  );
  const channelRows = device
    ? buildInputChannelRows(profiles, selectedProfileId, device)
    : [];

  return (
    <Stack gap="sm">
      <TextInput
        label={t("sttProfiles.name")}
        value={nameDraft}
        disabled={runtimeLocked}
        onChange={(event) => {
          setNameDraft(event.currentTarget.value);
        }}
        onBlur={commitName}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            commitName();
          }
        }}
      />
      <Stack gap={4}>
        <Text size="sm" fw={500}>
          {t("sttProfiles.color")}
        </Text>
        <Group gap="xs" role="radiogroup" aria-label={t("sttProfiles.color")}>
          {STT_PROFILE_DISPLAY_COLORS.map((color) => (
            <ActionIcon
              key={color}
              variant="subtle"
              color={color}
              aria-label={color}
              role="radio"
              aria-checked={displayColor === color}
              disabled={runtimeLocked}
              style={{
                border:
                  displayColor === color
                    ? "2px solid var(--mantine-color-text)"
                    : "2px solid transparent",
                backgroundColor: "transparent",
              }}
              onClick={() =>
                onUpdateProfile(profile.id, (current) => ({
                  ...current,
                  display_color: color,
                }))
              }
            >
              <ColorSwatch
                color={STT_PROFILE_DISPLAY_COLOR_CSS[color]}
                size={18}
                style={{
                  border: "1px solid var(--mantine-color-default-border)",
                }}
              />
            </ActionIcon>
          ))}
        </Group>
      </Stack>
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={t("tooltip.runtimeLocked")}
      >
        <Group align="end" gap="xs" wrap="nowrap">
          <Select
            label={settingLabel(
              t("settings.inputAudioDevice.label"),
              t("settings.inputAudioDevice.description"),
            )}
            placeholder={t("settings.inputAudioDevice.placeholder")}
            data={inputAudioDeviceOptions}
            value={selectedDevice}
            clearable={profiles.length === 1}
            searchable
            maxDropdownHeight={180}
            disabled={runtimeLocked || !localAudioDevicesAvailable}
            style={{ flex: 1 }}
            onChange={(value) => {
              if (!value) {
                if (profiles.length !== 1) return;
                onChange({
                  ...input,
                  device_host: null,
                  device_id: null,
                  device_name: null,
                  channel_index: 0,
                });
                return;
              }
              const [host, id] = value.split("\u0000");
              const nextDevice = inputAudioDevices.find(
                (candidate) => candidate.host === host && candidate.id === id,
              );
              if (!nextDevice) return;
              const channelIndex = availableInputChannelForProfile(
                profiles,
                selectedProfileId,
                nextDevice,
              );
              if (channelIndex === null) return;
              onChange({
                ...input,
                device_host: host,
                device_id: id,
                device_name: nextDevice.display_name,
                channel_index: channelIndex,
              });
              if (isLoopbackHost(host)) {
                void onRequestLoopbackPermission();
              }
            }}
          />
          <Tooltip label={t("settings.audioDevice.refreshTooltip")} withArrow>
            <span>
              <ActionIcon
                aria-label={t("settings.audioDevice.refreshAriaLabel")}
                variant="light"
                size="lg"
                loading={refreshingAudioDevices}
                disabled={runtimeLocked || !localAudioDevicesAvailable}
                onClick={onRefreshAudioDevices}
              >
                <IconRefresh />
              </ActionIcon>
            </span>
          </Tooltip>
        </Group>
      </DisabledReasonTooltip>
      {device && device.channels >= 2 ? (
        <Stack
          gap={4}
          role="radiogroup"
          aria-label={t("settings.inputAudioDevice.channelLabel")}
        >
          <Text size="sm" fw={500}>
            {t("settings.inputAudioDevice.channelLabel")}
          </Text>
          {channelRows.map((row, rowIndex) => (
            <ActionIcon.Group key={rowIndex}>
              {row.map(({ channelIndex, occupied }) => {
                const selected = channelIndex === input.channel_index;
                const disabled = runtimeLocked || occupied;

                return (
                  <ActionIcon
                    key={channelIndex}
                    role="radio"
                    aria-checked={selected}
                    aria-label={t("settings.inputAudioDevice.channel", {
                      number: channelIndex + 1,
                    })}
                    aria-disabled={disabled || undefined}
                    tabIndex={disabled ? -1 : selected ? 0 : -1}
                    variant={selected ? "filled" : "default"}
                    color={selected ? "blue" : undefined}
                    disabled={disabled}
                    onClick={() => {
                      if (!disabled) {
                        onChange({ ...input, channel_index: channelIndex });
                      }
                    }}
                  >
                    {channelIndex + 1}
                  </ActionIcon>
                );
              })}
            </ActionIcon.Group>
          ))}
        </Stack>
      ) : null}
      <Checkbox
        label={settingLabel(
          t("sttProfiles.neoHttpEnabled"),
          t("sttProfiles.neoHttpEnabledDescription"),
        )}
        checked={profile.neo_http_enabled}
        disabled={runtimeLocked}
        onChange={(event) =>
          onUpdateProfile(profile.id, (current) => ({
            ...current,
            neo_http_enabled: event.currentTarget.checked,
          }))
        }
      />
      <Checkbox
        label={settingLabel(
          t("sttProfiles.developerHttpEnabled"),
          t("sttProfiles.developerHttpEnabledDescription"),
        )}
        checked={profile.developer_http_enabled}
        disabled={runtimeLocked}
        onChange={(event) =>
          onUpdateProfile(profile.id, (current) => ({
            ...current,
            developer_http_enabled: event.currentTarget.checked,
          }))
        }
      />
    </Stack>
  );
};
