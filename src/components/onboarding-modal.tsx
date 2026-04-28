import {
  Button,
  Group,
  Modal,
  Progress,
  Select,
  Stack,
  Text,
} from "@mantine/core";
import { useTranslation } from "react-i18next";

import type { OnboardingState } from "../hooks/use-app-state";
import type {
  AsrModel,
  ModelDownloadProgress,
  ParapperConfig,
} from "../lib/types";

type SelectOption<T extends string = string> = {
  label: string;
  value: T;
};

type OnboardingModalProps = {
  onboarding: OnboardingState;
  config: ParapperConfig;
  languageOptions: SelectOption[];
  currentLanguage: string;
  asrModelSelectOptions: SelectOption<AsrModel>[];
  downloadingModels: boolean;
  modelDownloadProgress: ModelDownloadProgress | null;
  onClose: () => void;
  onBack: () => void;
  onNext: () => void;
  onLanguageChange: (language: string) => void;
  onApplyAsrModel: (model: AsrModel) => void;
  onDownloadSelectedModels: () => Promise<unknown>;
};

export const OnboardingModal: React.FC<OnboardingModalProps> = ({
  onboarding,
  config,
  languageOptions,
  currentLanguage,
  asrModelSelectOptions,
  downloadingModels,
  modelDownloadProgress,
  onClose,
  onBack,
  onNext,
  onLanguageChange,
  onApplyAsrModel,
  onDownloadSelectedModels,
}) => {
  const { t } = useTranslation();

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
            <Text size="sm" fw={600}>
              {t("onboarding.modelStep")}
            </Text>
            <Select
              data={asrModelSelectOptions}
              value={config.asr_model}
              allowDeselect={false}
              onChange={(value) => {
                if (value) onApplyAsrModel(value as AsrModel);
              }}
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
                  onClick={() => {
                    void onDownloadSelectedModels().then(onClose);
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
