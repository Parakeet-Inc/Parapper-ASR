import {
  Button,
  Group,
  Modal,
  Progress,
  Select,
  Stack,
  Text,
} from "@mantine/core";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import type { OnboardingState } from "../hooks/use-app-state";
import type {
  ConfigPreset,
  ModelDownloadProgress,
  ParapperConfig,
} from "../lib/types";

type SelectOption<T extends string = string> = {
  label: string;
  value: T;
};

type OnboardingModalProps = {
  onboarding: OnboardingState;
  languageOptions: SelectOption[];
  currentLanguage: string;
  configPresets: ConfigPreset[];
  downloadingModels: boolean;
  modelDownloadProgress: ModelDownloadProgress | null;
  onClose: () => void;
  onBack: () => void;
  onNext: () => void;
  onLanguageChange: (language: string) => void;
  onApplyPresetAndDownload: (config: ParapperConfig) => Promise<unknown>;
};

export const OnboardingModal: React.FC<OnboardingModalProps> = ({
  onboarding,
  languageOptions,
  currentLanguage,
  configPresets,
  downloadingModels,
  modelDownloadProgress,
  onClose,
  onBack,
  onNext,
  onLanguageChange,
  onApplyPresetAndDownload,
}) => {
  const { t } = useTranslation();
  const [selectedPresetName, setSelectedPresetName] = useState<string | null>(
    null,
  );
  const selectedPreset = useMemo(
    () =>
      configPresets.find((preset) => preset.name === selectedPresetName) ??
      null,
    [configPresets, selectedPresetName],
  );
  const presetOptions = useMemo(
    () =>
      configPresets.map((preset) => ({
        value: preset.name,
        label: preset.built_in
          ? t("settings.configPresets.builtInLabel", { name: preset.name })
          : preset.name,
      })),
    [configPresets, t],
  );

  useEffect(() => {
    if (selectedPresetName || configPresets.length === 0) return;
    setSelectedPresetName(configPresets[0].name);
  }, [configPresets, selectedPresetName]);

  return (
    <Modal
      opened={onboarding.open}
      onClose={onClose}
      title={t("onboarding.title")}
      centered
      withCloseButton={false}
      closeOnClickOutside={false}
      closeOnEscape={false}
    >
      <Stack gap="md">
        {onboarding.step === 0 ? (
          <>
            <Text size="sm" fw={600}>
              {t("onboarding.languageStep")}
            </Text>
            <Select
              data={languageOptions}
              value={currentLanguage}
              allowDeselect={false}
              onChange={(value) => {
                if (value) onLanguageChange(value);
              }}
            />
            <Group justify="flex-end">
              <Button onClick={onNext}>{t("onboarding.next")}</Button>
            </Group>
          </>
        ) : (
          <>
            <Stack gap={4}>
              <Text size="sm" fw={600}>
                {t("onboarding.presetStep")}
              </Text>
              <Text size="sm" c="dimmed">
                {t("onboarding.presetDescription")}
              </Text>
            </Stack>
            <Select
              data={presetOptions}
              value={selectedPresetName}
              allowDeselect={false}
              searchable
              maxDropdownHeight={260}
              onChange={setSelectedPresetName}
            />
            {downloadingModels || modelDownloadProgress ? (
              <Stack gap={4}>
                <Progress
                  value={(modelDownloadProgress?.progress ?? 0) * 100}
                  color="primary"
                />
                {modelDownloadProgress ? (
                  <Text size="xs" c="dimmed">
                    {t("downloadProgress.label", {
                      file: modelDownloadProgress.file_name,
                      index: modelDownloadProgress.file_index,
                      total: modelDownloadProgress.total_files,
                    })}
                  </Text>
                ) : null}
              </Stack>
            ) : null}
            <Group justify="space-between">
              <Button variant="default" onClick={onBack}>
                {t("onboarding.back")}
              </Button>
              <Group gap="xs">
                <Button
                  loading={downloadingModels}
                  disabled={!selectedPreset}
                  onClick={() => {
                    if (!selectedPreset) return;
                    void onApplyPresetAndDownload(selectedPreset.config).then(
                      onClose,
                    );
                  }}
                >
                  {t("onboarding.downloadAndClose")}
                </Button>
              </Group>
            </Group>
          </>
        )}
      </Stack>
    </Modal>
  );
};
