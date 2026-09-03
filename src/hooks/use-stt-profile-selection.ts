import { useEffect, useState } from "react";

import type { SttProfileConfig } from "../lib/types";

export const useSttProfileSelection = (
  profiles: readonly SttProfileConfig[],
) => {
  const [selectedProfileId, setSelectedProfileId] = useState(
    () => profiles[0]?.id ?? null,
  );

  useEffect(() => {
    if (profiles.some((profile) => profile.id === selectedProfileId)) return;
    setSelectedProfileId(profiles[0]?.id ?? null);
  }, [profiles, selectedProfileId]);

  return {
    selectedProfileId,
    selectProfile: setSelectedProfileId,
  };
};
