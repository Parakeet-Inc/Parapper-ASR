import { Button, Code, Modal, ScrollArea, Stack, Text } from "@mantine/core";
import { lazy, Suspense, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { ExternalLink } from "./external-link";
import type { RustLicensesDocument } from "../../application/frontend-services";

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
    url: "https://huggingface.co/nadare/parakeet-tdt_ctc-0.6b-ja-onnx-dynamic-int8",
  },
  {
    name: "NVIDIA Parakeet TDT CTC 0.6B Ja (upstream)",
    license: "CC-BY-4.0",
    url: "https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja",
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
    name: "NVIDIA NeMo (ASR behavioral reference)",
    license: "Apache-2.0",
    url: "https://github.com/NVIDIA/NeMo",
  },
  {
    name: "sherpa-onnx (model distribution and compatibility reference)",
    license: "Apache-2.0",
    url: "https://github.com/k2-fsa/sherpa-onnx",
  },
  {
    name: "Nemotron Speech Streaming 0.6B English",
    license: "NVIDIA Open Model License",
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
    license:
      "LFM Open License v1.0 (commercial use above US$10M annual revenue requires a separate license)",
    url: "https://huggingface.co/onnx-community/LFM2-350M-ENJP-MT-ONNX",
  },
  {
    name: "static-embedding-japanese",
    license: "MIT",
    url: "https://huggingface.co/hotchpotch/static-embedding-japanese",
  },
  {
    name: "CAT-Translate 0.8B ONNX Q4 block16",
    license: "MIT",
    url: "https://huggingface.co/nadare/CAT-Translate-0.8b-onnx-q4-k-quant",
  },
  {
    name: "Vibrato UniDic CWJ 3.1.1 dictionary",
    license: "BSD-3-Clause",
    url: "https://clrd.ninjal.ac.jp/unidic_archive/cwj/3.1.1/",
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
    name: "Supertonic3 (quantized)",
    license: "OpenRAIL-M",
    url: "https://huggingface.co/nadare/supertonic-3-onnx-q4",
  },
  {
    name: "UL-UNAS",
    license: "MIT",
    url: "https://github.com/Xiaobin-Rong/ul-unas",
  },
  {
    name: "Microsoft ONNX Runtime",
    license: "MIT",
    url: "https://github.com/microsoft/onnxruntime",
  },
];

export const dictionaryNoticeUrls = [
  "/licenses/hotword-reading/NOTICE.md",
  "/licenses/hotword-reading/LICENSE-APACHE-2.0.txt",
  "/licenses/morph/NOTICE",
  "/licenses/morph/BSD",
  "/licenses/morph/AUTHORS",
] as const;

export const thirdPartyNoticeUrls = [
  "/licenses/THIRD_PARTY_NOTICES.md",
] as const;

const LicenseDocuments: React.FC<{
  descriptionKey: string;
  failedKey: string;
  loadingKey: string;
  urls: readonly string[];
}> = ({ descriptionKey, failedKey, loadingKey, urls }) => {
  const { t } = useTranslation();
  const [documents, setDocuments] = useState<string[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    void Promise.all(
      urls.map(async (url) => {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to load license notice: ${response.status}`);
        }
        return response.text();
      }),
    )
      .then((texts) => {
        if (!cancelled) {
          setDocuments(texts);
        }
      })
      .catch((error: unknown) => {
        console.error(error);
        if (!cancelled) {
          setDocuments([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [urls]);

  return (
    <Stack gap="md">
      <Text size="sm">{t(descriptionKey)}</Text>
      {!documents ? (
        <Text>{t(loadingKey)}</Text>
      ) : documents.length === 0 ? (
        <Text>{t(failedKey)}</Text>
      ) : (
        documents.map((document, index) => (
          <Code block key={index}>
            <ScrollArea h={index === 0 ? 360 : 480}>{document}</ScrollArea>
          </Code>
        ))
      )}
    </Stack>
  );
};

export const BuiltInDictionaryLicenses: React.FC = () => (
  <LicenseDocuments
    descriptionKey="licenses.hotwordReadingNoticesDescription"
    failedKey="licenses.failedToLoadHotwordNotices"
    loadingKey="licenses.loadingHotwordNotices"
    urls={dictionaryNoticeUrls}
  />
);

export const ThirdPartyNotices: React.FC = () => (
  <LicenseDocuments
    descriptionKey="licenses.thirdPartyNoticesDescription"
    failedKey="licenses.failedToLoadThirdPartyNotices"
    loadingKey="licenses.loadingThirdPartyNotices"
    urls={thirdPartyNoticeUrls}
  />
);

export const Licenses: React.FC<{
  onOpenExternalUrl: (url: string) => Promise<void>;
  onLoadRustLicenses: () => Promise<RustLicensesDocument>;
}> = ({ onOpenExternalUrl, onLoadRustLicenses }) => {
  const { t } = useTranslation();
  const [rustLicensesOpened, setRustLicensesOpened] = useState(false);
  const [dictionaryLicensesOpened, setDictionaryLicensesOpened] =
    useState(false);
  const [thirdPartyNoticesOpened, setThirdPartyNoticesOpened] = useState(false);

  return (
    <Stack gap="lg">
      <Stack gap="xs">
        <Text size="sm" fw={600}>
          {t("licenses.modelLicenses")}
        </Text>
        <Stack gap={4}>
          {modelLicenses.map((license) => (
            <Text key={license.name} size="sm">
              <ExternalLink href={license.url} onOpen={onOpenExternalUrl}>
                {license.name}
              </ExternalLink>
              : {license.license}
            </Text>
          ))}
        </Stack>
      </Stack>

      <Stack gap="xs">
        <Text size="sm" fw={600}>
          {t("licenses.thirdPartyNotices")}
        </Text>
        <Button
          variant="default"
          onClick={() => setThirdPartyNoticesOpened(true)}
        >
          {t("licenses.openThirdPartyNotices")}
        </Button>
      </Stack>

      <Stack gap="xs">
        <Text size="sm" fw={600}>
          {t("licenses.hotwordReadingNotices")}
        </Text>
        <Button
          variant="default"
          onClick={() => setDictionaryLicensesOpened(true)}
        >
          {t("licenses.openHotwordReadingNotices")}
        </Button>
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
          {rustLicensesOpened ? (
            <RustLicenses
              onOpenExternalUrl={onOpenExternalUrl}
              onLoadRustLicenses={onLoadRustLicenses}
            />
          ) : null}
        </Suspense>
      </Modal>

      <Modal
        opened={dictionaryLicensesOpened}
        onClose={() => setDictionaryLicensesOpened(false)}
        title={t("licenses.hotwordReadingNotices")}
        size="xl"
      >
        {dictionaryLicensesOpened ? <BuiltInDictionaryLicenses /> : null}
      </Modal>

      <Modal
        opened={thirdPartyNoticesOpened}
        onClose={() => setThirdPartyNoticesOpened(false)}
        title={t("licenses.thirdPartyNotices")}
        size="xl"
      >
        {thirdPartyNoticesOpened ? <ThirdPartyNotices /> : null}
      </Modal>
    </Stack>
  );
};
