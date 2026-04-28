import { Anchor, Button, Modal, Stack, Text } from "@mantine/core";
import { lazy, Suspense, useState } from "react";
import { useTranslation } from "react-i18next";

import { notificationColor } from "../../lib/theme";

const RustLicenses = lazy(() => import("./rust-licenses"));

const modelLicenses = [
  {
    name: "ReazonSpeech K2 v2",
    license: "Apache-2.0",
    url: "https://huggingface.co/reazon-research/reazonspeech-k2-v2",
  },
  {
    name: "NeMo Parakeet TDT 0.6B v2 int8",
    license: "CC-BY-4.0",
    url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
  },
  {
    name: "NeMo Parakeet TDT 0.6B v3 int8",
    license: "CC-BY-4.0",
    url: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
  },
  {
    name: "Silero VAD",
    license: "MIT",
    url: "https://github.com/snakers4/silero-vad",
  },
];

const ExternalLink: React.FC<{
  href: string;
  children: React.ReactNode;
}> = ({ href, children }) => (
  <Anchor
    href={href}
    target="_blank"
    rel="noreferrer"
    c={notificationColor.info}
  >
    {children}
  </Anchor>
);

export const Licenses: React.FC = () => {
  const { t } = useTranslation();
  const [rustLicensesOpened, setRustLicensesOpened] = useState(false);

  return (
    <Stack gap="lg">
      <Stack gap="xs">
        <Text size="sm" fw={600}>
          {t("licenses.modelLicenses")}
        </Text>
        <Stack gap={4}>
          {modelLicenses.map((license) => (
            <Text key={license.name} size="sm">
              <ExternalLink href={license.url}>{license.name}</ExternalLink>:{" "}
              {license.license}
            </Text>
          ))}
        </Stack>
      </Stack>

      <Stack gap="xs">
        <Text size="sm" fw={600}>
          {t("licenses.rustLicenses")}
        </Text>
        <Button variant="default" onClick={() => setRustLicensesOpened(true)}>
          {t("licenses.openRustLicenses")}
        </Button>
      </Stack>

      <Modal
        opened={rustLicensesOpened}
        onClose={() => setRustLicensesOpened(false)}
        title={t("licenses.rustLicenses")}
        size="xl"
      >
        <Suspense fallback={<Text>{t("licenses.loadingRustLicenses")}</Text>}>
          {rustLicensesOpened ? <RustLicenses /> : null}
        </Suspense>
      </Modal>
    </Stack>
  );
};
