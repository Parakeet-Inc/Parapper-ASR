import type { RustLicensesDocument } from "../../application/frontend-services";
import { Licenses } from "../ui/licenses";

export const LicenseSettings: React.FC<{
  onOpenExternalUrl: (url: string) => Promise<void>;
  onLoadRustLicenses: () => Promise<RustLicensesDocument>;
}> = ({ onOpenExternalUrl, onLoadRustLicenses }) => (
  <Licenses
    onOpenExternalUrl={onOpenExternalUrl}
    onLoadRustLicenses={onLoadRustLicenses}
  />
);
