import { MantineProvider } from "@mantine/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

import {
  BuiltInDictionaryLicenses,
  dictionaryNoticeUrls,
  Licenses,
  ThirdPartyNotices,
  thirdPartyNoticeUrls,
} from "./licenses";

describe("hotword reading dictionary license", () => {
  it("loads every built-in dictionary notice from the installed application", () => {
    expect(dictionaryNoticeUrls).toEqual([
      "/licenses/hotword-reading/NOTICE.md",
      "/licenses/hotword-reading/LICENSE-APACHE-2.0.txt",
      "/licenses/morph/NOTICE",
      "/licenses/morph/BSD",
      "/licenses/morph/AUTHORS",
    ]);
  });

  it("loads the upstream attribution notice from the installed application", () => {
    expect(thirdPartyNoticeUrls).toEqual(["/licenses/THIRD_PARTY_NOTICES.md"]);
  });

  it("shows the dictionary attribution description inside the modal content only", () => {
    const listMarkup = renderToStaticMarkup(
      createElement(
        MantineProvider,
        null,
        createElement(Licenses, {
          onOpenExternalUrl: vi.fn(),
          onLoadRustLicenses: vi.fn(),
        }),
      ),
    );
    const modalContentMarkup = renderToStaticMarkup(
      createElement(
        MantineProvider,
        null,
        createElement(BuiltInDictionaryLicenses),
      ),
    );
    const thirdPartyMarkup = renderToStaticMarkup(
      createElement(MantineProvider, null, createElement(ThirdPartyNotices)),
    );

    expect(listMarkup).toContain("licenses.hotwordReadingNotices");
    expect(listMarkup).toContain("licenses.thirdPartyNotices");
    expect(listMarkup).not.toContain(
      "licenses.hotwordReadingNoticesDescription",
    );
    expect(listMarkup).not.toContain("licenses.thirdPartyNoticesDescription");
    expect(modalContentMarkup).toContain(
      "licenses.hotwordReadingNoticesDescription",
    );
    expect(thirdPartyMarkup).toContain("licenses.thirdPartyNoticesDescription");
  });
});
