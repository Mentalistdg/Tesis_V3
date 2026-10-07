import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import type { DailyLogEntry } from '../services/api';

export interface FilterState {
  dateStart: string;
  dateEnd: string;
  positions: Set<number>;       // {0, 1, 3}
  regimes: Set<string>;         // {'bull','bear','sideways','high_vol'}
  direction: 'all' | 'correct' | 'incorrect';
  changesOnly: boolean;
}

const ALL_POSITIONS = new Set([0, 1, 3]);
const ALL_REGIMES = new Set(['bull', 'bear', 'sideways', 'high_vol']);

function setsEqual<T>(a: Set<T>, b: Set<T>): boolean {
  if (a.size !== b.size) return false;
  for (const item of a) {
    if (!b.has(item)) return false;
  }
  return true;
}

function defaultState(): FilterState {
  return {
    dateStart: '',
    dateEnd: '',
    positions: new Set(ALL_POSITIONS),
    regimes: new Set(ALL_REGIMES),
    direction: 'all',
    changesOnly: false,
  };
}

export function useTableFilters(data: DailyLogEntry[]) {
  const [filters, setFilters] = useState<FilterState>(defaultState);

  // Reset filters when data source changes (e.g. model switch)
  const dataRef = useRef(data);
  useEffect(() => {
    if (data !== dataRef.current) {
      dataRef.current = data;
      setFilters(defaultState());
    }
  }, [data]);

  const isFiltered = useMemo(() => {
    return (
      filters.dateStart !== '' ||
      filters.dateEnd !== '' ||
      !setsEqual(filters.positions, ALL_POSITIONS) ||
      !setsEqual(filters.regimes, ALL_REGIMES) ||
      filters.direction !== 'all' ||
      filters.changesOnly
    );
  }, [filters]);

  const filteredData = useMemo(() => {
    if (!isFiltered) return data;

    return data.filter(day => {
      // Date range
      if (filters.dateStart && day.date < filters.dateStart) return false;
      if (filters.dateEnd && day.date > filters.dateEnd) return false;

      // Position
      if (!filters.positions.has(day.position_final)) return false;

      // Regime
      const regime = (!day.regime || day.regime === 'unknown') ? 'sideways' : day.regime;
      if (!filters.regimes.has(regime)) return false;

      // Direction
      if (filters.direction !== 'all') {
        const hasPosition = day.position_final > 0;
        const marketUp = day.market_return > 0;
        const isCorrect = (hasPosition && marketUp) || (!hasPosition && !marketUp);
        if (filters.direction === 'correct' && !isCorrect) return false;
        if (filters.direction === 'incorrect' && isCorrect) return false;
      }

      // Changes only
      if (filters.changesOnly && !day.position_changed) return false;

      return true;
    });
  }, [data, filters, isFiltered]);

  const togglePosition = useCallback((pos: number) => {
    setFilters(prev => {
      const next = new Set(prev.positions);
      if (next.has(pos)) {
        if (next.size > 1) next.delete(pos);
      } else {
        next.add(pos);
      }
      return { ...prev, positions: next };
    });
  }, []);

  const toggleRegime = useCallback((regime: string) => {
    setFilters(prev => {
      const next = new Set(prev.regimes);
      if (next.has(regime)) {
        if (next.size > 1) next.delete(regime);
      } else {
        next.add(regime);
      }
      return { ...prev, regimes: next };
    });
  }, []);

  const setDateStart = useCallback((v: string) => {
    setFilters(prev => ({ ...prev, dateStart: v }));
  }, []);

  const setDateEnd = useCallback((v: string) => {
    setFilters(prev => ({ ...prev, dateEnd: v }));
  }, []);

  const setDirection = useCallback((v: 'all' | 'correct' | 'incorrect') => {
    setFilters(prev => ({ ...prev, direction: v }));
  }, []);

  const toggleChangesOnly = useCallback(() => {
    setFilters(prev => ({ ...prev, changesOnly: !prev.changesOnly }));
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(defaultState());
  }, []);

  return {
    filters,
    filteredData,
    isFiltered,
    togglePosition,
    toggleRegime,
    setDateStart,
    setDateEnd,
    setDirection,
    toggleChangesOnly,
    resetFilters,
  };
}
