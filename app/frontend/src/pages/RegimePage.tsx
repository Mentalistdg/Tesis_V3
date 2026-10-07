import { useEffect, useState, useMemo } from 'react';
import { getRegimes, getModels } from '../services/api';
import type { RegimeData, ModelsResponse } from '../types';
import { Thermometer, TrendingUp, TrendingDown, Activity, Cloud } from 'lucide-react';
import { clsx } from 'clsx';
import LoadingScreen from '../components/LoadingScreen';
import { useTableSort } from '../hooks/useTableSort';
import SortableHeader from '../components/SortableHeader';

const regimeConfig = {
  bull: { label: 'Bull Market', icon: TrendingUp, color: 'text-[#00c853]', bgColor: 'bg-[#00c853]/10' },
  bear: { label: 'Bear Market', icon: TrendingDown, color: 'text-[#c41e3a]', bgColor: 'bg-[#c41e3a]/10' },
  sideways: { label: 'Sideways', icon: Activity, color: 'text-[#737373]', bgColor: 'bg-[#737373]/10' },
  high_vol: { label: 'High Volatility', icon: Cloud, color: 'text-[#ff9800]', bgColor: 'bg-[#ff9800]/10' },
  unknown: { label: 'Unknown', icon: Activity, color: 'text-[#737373]', bgColor: 'bg-[#737373]/10' },
};

export default function RegimePage() {
  const [regimeData, setRegimeData] = useState<RegimeData | null>(null);
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedRegime, setSelectedRegime] = useState<string>('bull');

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      const [regimes, models] = await Promise.all([getRegimes(), getModels()]);
      setRegimeData(regimes);
      setModelsData(models);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  // Hooks must be above early returns
  interface RegimePerfRow {
    model: string;
    performance: { total_return: number; n_days: number };
  }

  const regimePerformanceRaw = useMemo<RegimePerfRow[]>(() => {
    if (!regimeData) return [];
    return Object.entries(regimeData.regime_performance)
      .map(([model, regimes]) => ({
        model,
        performance: regimes[selectedRegime] || { total_return: 0, n_days: 0 }
      }))
      .filter(p => p.performance.n_days > 0);
  }, [regimeData, selectedRegime]);

  const regimeAccessor = useMemo(() => ({
    model: (r: RegimePerfRow) => r.model,
    total_return: (r: RegimePerfRow) => r.performance.total_return,
    n_days: (r: RegimePerfRow) => r.performance.n_days,
  }), []);

  const {
    sortedData: regimePerformance,
    requestSort: requestRegimeSort,
    getSortDirection: getRegimeSortDir,
  } = useTableSort(regimePerformanceRaw, 'total_return', 'desc', regimeAccessor);

  if (loading) {
    return <LoadingScreen />;
  }

  if (!regimeData || !modelsData) {
    return <div className="text-center py-8 text-[#c41e3a]">Failed to load data</div>;
  }

  const currentRegime = regimeConfig[regimeData.current_regime as keyof typeof regimeConfig] || regimeConfig.unknown;
  const CurrentIcon = currentRegime.icon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Market Regime Analysis</h2>
      </div>

      {/* Current Regime */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Thermometer className="w-5 h-5 text-[#737373]" />
          <h3 className="text-lg font-medium">Current Market Regime</h3>
        </div>

        <div className={clsx("p-6 rounded-lg text-center", currentRegime.bgColor)}>
          <CurrentIcon className={clsx("w-12 h-12 mx-auto mb-3", currentRegime.color)} />
          <div className={clsx("text-2xl font-bold", currentRegime.color)}>
            {currentRegime.label}
          </div>
          <div className="text-sm text-[#737373] mt-2">
            Based on recent 60-day market performance and volatility
          </div>
        </div>
      </div>

      {/* Regime Distribution */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Regime Distribution</h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(regimeData.regime_counts)
            .filter(([regime]) => regime !== 'unknown')
            .map(([regime, count]) => {
              const config = regimeConfig[regime as keyof typeof regimeConfig];
              const Icon = config?.icon || Activity;
              const total = Object.values(regimeData.regime_counts).reduce((a, b) => a + b, 0);
              const percentage = ((count / total) * 100).toFixed(1);

              return (
                <div
                  key={regime}
                  className={clsx(
                    "p-4 rounded-lg cursor-pointer border-2 transition-colors",
                    selectedRegime === regime
                      ? "border-[#c41e3a]"
                      : "border-transparent",
                    config?.bgColor
                  )}
                  onClick={() => setSelectedRegime(regime)}
                >
                  <Icon className={clsx("w-6 h-6 mb-2", config?.color)} />
                  <div className={clsx("font-medium", config?.color)}>
                    {config?.label}
                  </div>
                  <div className="text-sm text-[#737373]">
                    {count} days ({percentage}%)
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Model Performance by Selected Regime */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">
          Model Performance in {regimeConfig[selectedRegime as keyof typeof regimeConfig]?.label}
        </h3>

        <div className="overflow-x-auto">
          <table className="table-dark">
            <thead>
              <tr>
                <th className="w-8">#</th>
                <SortableHeader label="Model" sortKey="model" activeDirection={getRegimeSortDir('model')} onSort={requestRegimeSort} className="w-40" />
                <SortableHeader label="Return" sortKey="total_return" activeDirection={getRegimeSortDir('total_return')} onSort={requestRegimeSort} className="w-24 text-right" />
                <SortableHeader label="Days" sortKey="n_days" activeDirection={getRegimeSortDir('n_days')} onSort={requestRegimeSort} className="w-20 text-right" />
              </tr>
            </thead>
            <tbody>
              {regimePerformance.map((item, idx) => (
                <tr key={item.model}>
                  <td className="text-[#737373]">{idx + 1}</td>
                  <td className="font-medium">{item.model}</td>
                  <td className={clsx(
                    "text-right font-mono",
                    item.performance.total_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                  )}>
                    {item.performance.total_return >= 0 ? '+' : ''}
                    {(item.performance.total_return * 100).toFixed(2)}%
                  </td>
                  <td className="text-right font-mono text-[#737373]">
                    {item.performance.n_days}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {regimePerformance.length === 0 && (
          <div className="text-center py-8 text-[#737373]">
            No performance data for this regime
          </div>
        )}
      </div>

      {/* Recommendation */}
      {regimePerformance.length > 0 && (
        <div className="card bg-[#c41e3a]/10 border-[#c41e3a]">
          <div className="text-lg font-medium mb-2">Recommendation</div>
          <div className="text-[#737373]">
            In {regimeConfig[selectedRegime as keyof typeof regimeConfig]?.label} conditions,{' '}
            <span className="text-[#f5f5f5] font-medium">{regimePerformance[0].model}</span>{' '}
            has historically performed best with a return of{' '}
            <span className={clsx(
              "font-medium",
              regimePerformance[0].performance.total_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
            )}>
              {regimePerformance[0].performance.total_return >= 0 ? '+' : ''}
              {(regimePerformance[0].performance.total_return * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
