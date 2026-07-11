import { Button, Modal, Stack, Text } from "@mantine/core";
import { lazy, Suspense, useState } from "react";
import { useTranslation } from "react-i18next";

import { ExternalLink } from "./external-link";

const RustLicenses = lazy(() => import("./rust-licenses"));

const modelLicenses = [
  {
    name: "ReazonSpeech K2 v2",
    license: "Apache-2.0",
    url: "https://huggingface.co/reazon-research/reazonspeech-k2-v2",
  },
  {
    name: "NeMo Parakeet TDT CTC 0.6B Ja 35000 int8",
    license: "CC-BY-4.0",
    url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
  },
  {
    name: "NeMo Parakeet TDT 0.6B v2 int8",
    license: "CC-BY-4.0",
    url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
  },
  {
    name: "NeMo Parakeet TDT 0.6B v3 int8",
    license: "CC-BY-4.0",
    url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
  },
  {
    name: "Nemotron Speech Streaming 0.6B English",
    license: "openmdw-1.1",
    url: "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b",
  },
  {
    name: "Nemotron 3.5 ASR Streaming 0.6B",
    license: "openmdw-1.1",
    url: "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
  },
  {
    name: "Silero VAD",
    license: "MIT",
    url: "https://github.com/snakers4/silero-vad",
  },
  {
    name: "Namo Turn Detector v1 Japanese",
    license: "Apache-2.0",
    url: "https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Japanese",
  },
  {
    name: "Namo Turn Detector v1 English",
    license: "Apache-2.0",
    url: "https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-English",
  },
  {
    name: "Namo Turn Detector v1 Multilingual",
    license: "Apache-2.0",
    url: "https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual",
  },
  {
    name: "SpeechBrain ECAPA-TDNN VoxLingua107",
    license: "Apache-2.0",
    url: "https://huggingface.co/drakulavich/SpeechBrain-coreml",
  },
  {
    name: "LFM2-350M-ENJP-MT ONNX (ONNX Community conversion)",
    license: "LFM Open License v1.0",
    url: "https://huggingface.co/onnx-community/LFM2-350M-ENJP-MT-ONNX",
  },
  // CAT-Translateは配布を一時停止しているため、ライセンス一覧からも非表示にしています。
  // {
  //   name: "CAT-Translate-0.8b",
  //   license: "MIT",
  //   url: "https://huggingface.co/cyberagent/CAT-Translate-0.8b",
  // },
  {
    name: "Vibrato UniDic CWJ 3.1.1 dictionary",
    license: "See archive license files",
    url: "https://github.com/daac-tools/vibrato/releases/tag/v0.5.0",
  },
  {
    name: "piper-voices en_US Kristin medium",
    license: "MIT",
    url: "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/kristin/medium",
  },
  {
    name: "piper-voices en_US John medium",
    license: "MIT",
    url: "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/john/medium",
  },
  {
    name: "piper-voices en_US Norman medium",
    license: "MIT",
    url: "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/norman/medium",
  },
  {
    name: "espeak-ng-data",
    license: "GPL-3.0-or-later",
    url: "https://github.com/espeak-ng/espeak-ng/tree/master/espeak-ng-data",
  },
  {
    name: "Supertonic 2 ONNX",
    license: "OpenRAIL-M",
    url: "https://huggingface.co/Supertone/supertonic-2",
  },
  {
    name: "Supertonic 3 ONNX",
    license: "OpenRAIL-M",
    url: "https://huggingface.co/Supertone/supertonic-3",
  },
  {
    name: "UL-UNAS",
    license: "MIT",
    url: "https://github.com/Xiaobin-Rong/ul-unas",
  },
];

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
