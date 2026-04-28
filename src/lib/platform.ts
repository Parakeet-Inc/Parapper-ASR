export const isMacOs = () => {
  if (typeof navigator === "undefined") return false;

  return (
    navigator.platform.toLowerCase().includes("mac") ||
    navigator.userAgent.toLowerCase().includes("mac os x")
  );
};
