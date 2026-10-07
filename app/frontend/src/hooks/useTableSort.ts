import { useMemo, useState, useCallback } from 'react';

export type SortDirection = 'asc' | 'desc';

export interface SortConfig {
  key: string;
  direction: SortDirection;
}

interface UseTableSortReturn<T> {
  sortedData: T[];
  sortConfig: SortConfig;
  requestSort: (key: string) => void;
  getSortDirection: (key: string) => SortDirection | null;
}

export function useTableSort<T>(
  data: T[],
  defaultKey: string,
  defaultDirection: SortDirection = 'desc',
  accessor?: Record<string, (item: T) => number | string>
): UseTableSortReturn<T> {
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: defaultKey,
    direction: defaultDirection,
  });

  const requestSort = useCallback((key: string) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc',
    }));
  }, []);

  const getSortDirection = useCallback(
    (key: string): SortDirection | null => {
      return sortConfig.key === key ? sortConfig.direction : null;
    },
    [sortConfig]
  );

  const sortedData = useMemo(() => {
    const sorted = [...data];
    const { key, direction } = sortConfig;

    sorted.sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      if (accessor && accessor[key]) {
        aVal = accessor[key](a);
        bVal = accessor[key](b);
      } else {
        aVal = (a as Record<string, any>)[key];
        bVal = (b as Record<string, any>)[key];
      }

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return direction === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      const numA = typeof aVal === 'number' ? aVal : 0;
      const numB = typeof bVal === 'number' ? bVal : 0;
      return direction === 'asc' ? numA - numB : numB - numA;
    });

    return sorted;
  }, [data, sortConfig, accessor]);

  return { sortedData, sortConfig, requestSort, getSortDirection };
}
