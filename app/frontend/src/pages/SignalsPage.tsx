import { useEffect, useState } from 'react';
import { getSignals } from '../services/api';
import type { SignalsData } from '../types';
import { TrendingUp, Minus, ArrowUp, ArrowDown } from 'lucide-react';
import { clsx } from 'clsx';

export default function SignalsPage() {
  const [data, setData] = useState<SignalsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      const signals = await getSignals();
      setData(signals);
    } catch (err) {
      setError('Failed to load signals');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return <div className="text-center py-8 text-[#737373]">Loading signals...</div>;
  }

  if (error || !data) {
    return <div className="text-center py-8 text-[#c41e3a]">{error || 'No data'}</div>;
  }

  const { consensus, signals, strategy } = data;

  // Get counts (support both new LONG-ONLY and legacy format)
  const uproCount = consensus.upro_count ?? consensus.bullish_count ?? 0;
  const spyCount = consensus.spy_count ?? 0;
  const cashCount = consensus.cash_count ?? consensus.neutral_count ?? 0;
  const recommended = consensus.recommended ?? (uproCount > consensus.total_models / 2 ? 'UPRO' : 'CASH');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Current Signals</h2>
          {strategy && (
            <span className="text-xs text-[#00c853] bg-[#00c853]/10 px-2 py-0.5 rounded mt-1 inline-block">
              {strategy} Strategy
            </span>
          )}
        </div>
        <span className="text-sm text-[#737373]">
          Last update: {data.date}
        </span>
      </div>

      {/* Consensus View - LONG-ONLY */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Consensus View ({consensus.total_models} models)</h3>

        {/* Consensus Bar - LONG-ONLY: UPRO / SPY / CASH */}
        <div className="mb-6">
          <div className="flex h-10 rounded overflow-hidden">
            {uproCount > 0 && (
              <div
                className="bg-[#00c853] flex items-center justify-center text-sm font-bold"
                style={{ width: `${(uproCount / consensus.total_models) * 100}%` }}
              >
                {uproCount}
              </div>
            )}
            {spyCount > 0 && (
              <div
                className="bg-[#4ade80] flex items-center justify-center text-sm font-bold text-black"
                style={{ width: `${(spyCount / consensus.total_models) * 100}%` }}
              >
                {spyCount}
              </div>
            )}
            {cashCount > 0 && (
              <div
                className="bg-[#525252] flex items-center justify-center text-sm font-bold"
                style={{ width: `${(cashCount / consensus.total_models) * 100}%` }}
              >
                {cashCount}
              </div>
            )}
          </div>
          <div className="flex justify-between text-xs text-[#737373] mt-2">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-[#00c853]"></span>
              UPRO +3x ({uproCount})
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-[#4ade80]"></span>
              SPY +1x ({spyCount})
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-[#525252]"></span>
              Cash ({cashCount})
            </span>
          </div>
        </div>

        {/* Recommended Position */}
        <div className="text-center py-4 bg-[#1a1a1a] rounded-lg">
          <div className="text-sm text-[#737373] uppercase tracking-wider">Recommended Position</div>
          <div className={clsx(
            "text-4xl font-bold mt-2",
            recommended === 'UPRO' ? "text-[#00c853]" :
            recommended === 'SPY' ? "text-[#4ade80]" : "text-[#737373]"
          )}>
            {recommended === 'UPRO' ? '+3x UPRO' : recommended === 'SPY' ? '+1x SPY' : 'CASH'}
          </div>
          <div className="text-sm text-[#737373] mt-1">
            Avg Position: {consensus.position.toFixed(2)}x |
            Bullish Models: {((uproCount + spyCount) / consensus.total_models * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Individual Signals - LONG-ONLY */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Individual Model Signals</h3>

        <table className="table-dark">
          <thead>
            <tr>
              <th className="w-40">Model</th>
              <th className="w-32">Category</th>
              <th className="w-20 text-right">Position</th>
              <th className="w-28">Instrument</th>
              <th className="w-24 text-right">Percentile</th>
              <th className="w-20 text-right">Change</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((signal) => {
              // Determine instrument and styling based on position
              const isUPRO = signal.position === 3 || signal.signal === 'UPRO_3X';
              const isSPY = signal.position === 1 || signal.signal === 'SPY_1X';
              const isCash = signal.position === 0 || signal.signal === 'CASH';

              return (
                <tr key={signal.model}>
                  <td className="font-medium">{signal.model}</td>
                  <td className="text-[#737373]">{signal.category}</td>
                  <td className={clsx(
                    "text-right font-mono font-bold",
                    isUPRO ? "text-[#00c853]" :
                    isSPY ? "text-[#4ade80]" : "text-[#737373]"
                  )}>
                    {isUPRO ? '+3x' : isSPY ? '+1x' : '0'}
                  </td>
                  <td>
                    <span className={clsx(
                      "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
                      isUPRO && "bg-[#00c853]/20 text-[#00c853]",
                      isSPY && "bg-[#4ade80]/20 text-[#4ade80]",
                      isCash && "bg-[#525252]/20 text-[#737373]"
                    )}>
                      {isUPRO && <TrendingUp className="w-3 h-3" />}
                      {isSPY && <TrendingUp className="w-3 h-3" />}
                      {isCash && <Minus className="w-3 h-3" />}
                      {isUPRO ? 'UPRO' : isSPY ? 'SPY' : 'CASH'}
                    </span>
                  </td>
                  <td className="text-right font-mono text-[#a3a3a3]">
                    {signal.percentile !== undefined ? `${signal.percentile.toFixed(0)}%` : '-'}
                  </td>
                  <td className={clsx(
                    "text-right font-mono",
                    signal.change > 0 ? "text-[#00c853]" :
                    signal.change < 0 ? "text-[#737373]" : "text-[#525252]"
                  )}>
                    <span className="inline-flex items-center gap-1">
                      {signal.change > 0 && <ArrowUp className="w-3 h-3" />}
                      {signal.change < 0 && <ArrowDown className="w-3 h-3" />}
                      {signal.change !== 0 ? `${signal.change > 0 ? '+' : ''}${signal.change.toFixed(0)}` : '-'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Strategy Info */}
      <div className="card bg-[#0a0a0a]">
        <h4 className="text-sm font-medium text-[#737373] mb-2">LONG-ONLY Strategy</h4>
        <div className="grid grid-cols-3 gap-4 text-xs">
          <div className="text-center p-2 bg-[#111111] rounded border border-[#00c853]/20">
            <div className="text-[#00c853] font-bold">+3x UPRO</div>
            <div className="text-[#525252]">Top 10% predictions</div>
          </div>
          <div className="text-center p-2 bg-[#111111] rounded border border-[#4ade80]/20">
            <div className="text-[#4ade80] font-bold">+1x SPY</div>
            <div className="text-[#525252]">Top 30% predictions</div>
          </div>
          <div className="text-center p-2 bg-[#111111] rounded border border-[#525252]/20">
            <div className="text-[#737373] font-bold">CASH</div>
            <div className="text-[#525252]">Not confident</div>
          </div>
        </div>
      </div>
    </div>
  );
}
