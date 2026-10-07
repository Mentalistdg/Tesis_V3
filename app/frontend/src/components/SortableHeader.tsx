import { ChevronUp, ChevronDown } from 'lucide-react';
import type { SortDirection } from '../hooks/useTableSort';

interface SortableHeaderProps {
  label: string;
  sortKey: string;
  activeDirection: SortDirection | null;
  onSort: (key: string) => void;
  className?: string;
}

export default function SortableHeader({
  label,
  sortKey,
  activeDirection,
  onSort,
  className = '',
}: SortableHeaderProps) {
  return (
    <th
      className={`cursor-pointer select-none group ${className}`}
      onClick={() => onSort(sortKey)}
    >
      <div className="flex items-center gap-0.5 justify-inherit">
        <span className="group-hover:text-[#f5f5f5] transition-colors">{label}</span>
        <span className="inline-flex flex-col -space-y-1 ml-0.5">
          <ChevronUp
            className={`w-3 h-3 transition-colors ${
              activeDirection === 'asc'
                ? 'text-[#c41e3a]'
                : 'text-transparent group-hover:text-[#525252]'
            }`}
          />
          <ChevronDown
            className={`w-3 h-3 transition-colors ${
              activeDirection === 'desc'
                ? 'text-[#c41e3a]'
                : 'text-transparent group-hover:text-[#525252]'
            }`}
          />
        </span>
      </div>
    </th>
  );
}
