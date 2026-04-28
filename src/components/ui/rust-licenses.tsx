import { Anchor, Box, Code, ScrollArea, Stack, Text } from "@mantine/core";
import { useTranslation } from "react-i18next";

import rustLicenses from "../../../licenses/rust.json";

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

const rustLicenseData = rustLicenses as CargoAboutOutput;

const ExternalLink: React.FC<{
  href: string;
  children: React.ReactNode;
}> = ({ href, children }) => (
  <Anchor href={href} target="_blank" rel="noreferrer">
    {children}
  </Anchor>
);

const RustLicenses: React.FC = () => {
  const { t } = useTranslation();

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
