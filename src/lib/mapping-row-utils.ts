export type MappingRow = {
  id: string;
};

export const updateById = <T extends MappingRow>(
  rows: T[],
  id: string,
  patch: Partial<T>,
) =>
  rows.map((row) => (row.id === id ? { ...row, ...patch } : row));

export const removeById = <T extends MappingRow>(rows: T[], id: string) =>
  rows.filter((row) => row.id !== id);

export const moveById = <T extends MappingRow>(
  rows: T[],
  id: string,
  direction: -1 | 1,
) => {
  const index = rows.findIndex((row) => row.id === id);
  const nextIndex = index + direction;
  if (index < 0 || nextIndex < 0 || nextIndex >= rows.length) {
    return rows;
  }
  const next = [...rows];
  [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
  return next;
};
