import type { RefObject } from "react";
import { useEffect } from "react";

const syncLogRowHeights = (root: HTMLElement) => {
  const scrollContainers = Array.from(
    root.querySelectorAll<HTMLElement>("[data-log-scroll]"),
  ).map((element) => ({
    element,
    wasAtBottom:
      element.scrollHeight - element.scrollTop - element.clientHeight <= 24,
  }));
  const rows = Array.from(
    root.querySelectorAll<HTMLElement>("[data-log-row-id]"),
  );
  const rowsById = new Map<string, HTMLElement[]>();

  for (const row of rows) {
    row.style.minHeight = "";
    const id = row.dataset.logRowId;
    if (!id) continue;
    const groupedRows = rowsById.get(id) ?? [];
    groupedRows.push(row);
    rowsById.set(id, groupedRows);
  }

  for (const groupedRows of rowsById.values()) {
    if (groupedRows.length < 2) continue;
    const maxHeight = Math.max(...groupedRows.map((row) => row.offsetHeight));
    for (const row of groupedRows) {
      row.style.minHeight = `${maxHeight}px`;
    }
  }

  for (const { element, wasAtBottom } of scrollContainers) {
    if (wasAtBottom) {
      element.scrollTop = element.scrollHeight;
    }
  }
};

export const useSyncedLogRowHeights = (
  rootRef: RefObject<HTMLElement>,
  enabled: boolean,
) => {
  useEffect(() => {
    const root = rootRef.current;
    if (!root || !enabled) return;

    let animationFrame = 0;
    const scheduleSync = () => {
      cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(() => syncLogRowHeights(root));
    };

    scheduleSync();
    const resizeObserver = new ResizeObserver(scheduleSync);
    resizeObserver.observe(root);
    const mutationObserver = new MutationObserver(scheduleSync);
    root.querySelectorAll<HTMLElement>("[data-log-scroll]").forEach((element) =>
      mutationObserver.observe(element, {
        childList: true,
        characterData: true,
        subtree: true,
      }),
    );

    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
    };
  }, [enabled, rootRef]);
};
