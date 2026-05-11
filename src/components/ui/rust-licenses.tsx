import { Box, Code, ScrollArea, Stack, Text } from "@mantine/core";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { ExternalLink } from "./external-link";

type CargoAboutCrate = {
  name: string;
  version: string;
};

type CargoAboutUsedBy = {
  crate: CargoAboutCrate;
};

type CargoAboutLicense = {
  name: string;
  text: string;
  used_by: CargoAboutUsedBy[];
};

type CargoAboutOutput = {
  licenses: CargoAboutLicense[];
};

const RustLicenses: React.FC = () => {
  const { t } = useTranslation();
  const [rustLicenseData, setRustLicenseData] =
    useState<CargoAboutOutput | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/licenses/rust.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load Rust licenses: ${response.status}`);
        }
        return response.json() as Promise<CargoAboutOutput>;
      })
      .then((data) => {
        if (!cancelled) {
          setRustLicenseData(data);
        }
      })
      .catch((error: unknown) => {
        console.error(error);
        if (!cancelled) {
          setRustLicenseData({ licenses: [] });
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (!rustLicenseData) {
    return <Text>{t("licenses.loadingRustLicenses")}</Text>;
  }

  return (
    <Stack gap="md">
      {rustLicenseData.licenses.map((license) => (
        <Box key={license.name}>
          <Text size="sm" fw={600}>
            {license.name}
          </Text>
          <Text size="xs" c="dimmed">
            {t("licenses.usedBy")}
          </Text>
          <Stack gap={0} mb="xs">
            {license.used_by.map((usedBy) => (
              <Text key={`${license.name}-${usedBy.crate.name}`} size="xs">
                -{" "}
                <ExternalLink
                  href={`https://crates.io/crates/${usedBy.crate.name}`}
                >
                  {usedBy.crate.name}
                </ExternalLink>
              </Text>
            ))}
          </Stack>
          <Code block>
            <ScrollArea h={160}>{license.text}</ScrollArea>
          </Code>
        </Box>
      ))}
    </Stack>
  );
};

export default RustLicenses;
