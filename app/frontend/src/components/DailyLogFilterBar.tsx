import { X } from 'lucide-react';
import { clsx } from 'clsx';
import type { FilterState } from '../hooks/useTableFilters';

interface DailyLogFilterBarProps {
  filters: FilterState;
  totalRows: number;
  filteredRows: number;
  isFiltered: boolean;
  onTogglePosition: (pos: number) => void;
  onToggleRegime: (regime: string) => void;
  onSetDateStart: (v: string) => void;
  onSetDateEnd: (v: string) => void;
  onSetDirection: (v: 'all' | 'correct' | 'incorrect') => void;
  onToggleChangesOnly: () => void;
  onReset: () => void;
}

const positionPills = [
  { value: 3, label: '3x', activeClass: 'bg-[#00c853]/20 text-[#00c853] border-[#00c853]/40' },
  { value: 1, label: '1x', activeClass: 'bg-[#22d3ee]/20 text-[#22d3ee] border-[#22d3ee]/40' },
  { value: 0, label: '0x', activeClass: 'bg-[#525252]/20 text-[#a3a3a3] border-[#525252]/40' },
];

const regimePills = [
  { value: 'bull', label: 'bull', activeClass: 'bg-[#00c853]/20 text-[#00c853] border-[#00c853]/40' },
  { value: 'bear', label: 'bear', activeClass: 'bg-[#c41e3a]/20 text-[#c41e3a] border-[#c41e3a]/40' },
  { value: 'sideways', label: 'side', activeClass: 'bg-[#525252]/20 text-[#a3a3a3] border-[#525252]/40' },
  { value: 'high_vol', label: 'hvol', activeClass: 'bg-[#f59e0b]/20 text-[#f59e0b] border-[#f59e0b]/40' },
];

const directionOptions: { value: 'all' | 'correct' | 'incorrect'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'correct', label: '\u2713' },
  { value: 'incorrect', label: '\u2717' },
];

const inactivePill = 'bg-transparent text-[#525252] border-[#333333]';

export default function DailyLogFilterBar({
  filters,
  totalRows,
  filteredRows,
  isFiltered,
  onTogglePosition,
  onToggleRegime,
  onSetDateStart,
  onSetDateEnd,
  onSetDirection,
  onToggleChangesOnly,
  onReset,
}: DailyLogFilterBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 py-2 px-1 text-xs">
      {/* Date Range */}
      <input
        type="date"
        value={filters.dateStart}
        onChange={(e) => onSetDateStart(e.target.value)}
        className="filter-date-input"
        title="Start date"
      />
      <span className="text-[#525252]">&mdash;</span>
      <input
        type="date"
        value={filters.dateEnd}
        onChange={(e) => onSetDateEnd(e.target.value)}
        className="filter-date-input"
        title="End date"
      />

      <div className="filter-divider" />

      {/* Position Pills */}
      {positionPills.map(p => (
        <button
          key={p.value}
          onClick={() => onTogglePosition(p.value)}
          className={clsx(
            'filter-pill',
            filters.positions.has(p.value) ? p.activeClass : inactivePill
          )}
        >
          {p.label}
        </button>
      ))}

      <div className="filter-divider" />

      {/* Regime Pills */}
      {regimePills.map(r => (
        <button
          key={r.value}
          onClick={() => onToggleRegime(r.value)}
          className={clsx(
            'filter-pill',
            filters.regimes.has(r.value) ? r.activeClass : inactivePill
          )}
        >
          {r.label}
        </button>
      ))}

      <div className="filter-divider" />

      {/* Direction */}
      {directionOptions.map(d => (
        <button
          key={d.value}
          onClick={() => onSetDirection(d.value)}
          className={clsx(
            'filter-pill',
            filters.direction === d.value
              ? 'bg-[#c41e3a]/20 text-[#c41e3a] border-[#c41e3a]/40'
              : inactivePill
          )}
        >
          {d.label}
        </button>
      ))}

      <div className="filter-divider" />

      {/* Changes Only */}
      <button
        onClick={onToggleChangesOnly}
        className={clsx(
          'filter-pill',
          filters.changesOnly
            ? 'bg-[#6366f1]/20 text-[#6366f1] border-[#6366f1]/40'
            : inactivePill
        )}
      >
        Changes
      </button>

      {/* Row Count */}
      <span className={clsx(
        "ml-auto font-mono tabular-nums",
        isFiltered ? "text-[#f59e0b]" : "text-[#525252]"
      )}>
        {isFiltered ? `${filteredRows} / ${totalRows}` : `${totalRows}`} rows
      </span>

      {/* Clear */}
      {isFiltered && (
        <button
          onClick={onReset}
          className="p-1 rounded hover:bg-[#222222] text-[#737373] hover:text-white transition-colors"
          title="Clear all filters"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}
