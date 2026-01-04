import { useEffect, useState } from 'react';
import { getSignals } from '../services/api';
import type { SignalsData } from '../types';
import { TrendingUp, TrendingDown, Minus, ArrowUp, ArrowDown } from 'lucide-react';
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

  const { consensus, signals } = data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Current Signals</h2>
        <span className="text-sm text-[#737373]">
          Last update: {data.date}
        </span>
      </div>

      {/* Consensus View */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Consensus View ({consensus.total_models} models)</h3>

        {/* Consensus Bar */}
        <div className="mb-6">
          <div className="flex h-8 rounded overflow-hidden">
            <div
              className="bg-[#00c853] flex items-center justify-center text-sm font-medium"
              style={{ width: `${(consensus.bullish_count / consensus.total_models) * 100}%` }}
            >
              {consensus.bullish_count}
            </div>
            <div
              className="bg-[#737373] flex items-center justify-center text-sm font-medium"
              style={{ width: `${(consensus.neutral_count / consensus.total_models) * 100}%` }}
            >
              {consensus.neutral_count}
            </div>
            <div
              className="bg-[#c41e3a] flex items-center justify-center text-sm font-medium"
              style={{ width: `${(consensus.bearish_count / consensus.total_models) * 100}%` }}
            >
              {consensus.bearish_count}
            </div>
          </div>
          <div className="flex justify-between text-xs text-[#737373] mt-1">
            <span>Bullish (≥1.5x)</span>
            <span>Neutral</span>
            <span>Bearish (&lt;0.5x)</span>
          </div>
        </div>

        {/* Consensus Position */}
        <div className="text-center py-4 bg-[#1a1a1a] rounded-lg">
          <div className="text-sm text-[#737373] uppercase tracking-wider">Consensus Position</div>
          <div className={clsx(
            "text-4xl font-bold mt-2",
            consensus.position >= 1.5 ? "text-[#00c853]" :
            consensus.position < 0.5 ? "text-[#c41e3a]" : "text-[#737373]"
          )}>
            {consensus.position.toFixed(1)}x LONG
          </div>
          <div className="text-sm text-[#737373] mt-1">
            Confidence: {((consensus.bullish_count / consensus.total_models) * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Individual Signals */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Individual Model Signals</h3>

        <table className="table-dark">
          <thead>
            <tr>
              <th className="w-40">Model</th>
              <th className="w-32">Category</th>
              <th className="w-20 text-right">Position</th>
              <th className="w-24">Signal</th>
              <th className="w-20 text-right">Change</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((signal) => (
              <tr key={signal.model}>
                <td className="font-medium">{signal.model}</td>
                <td className="text-[#737373]">{signal.category}</td>
                <td className={clsx(
                  "text-right font-mono",
                  signal.position >= 1.5 ? "text-[#00c853]" :
                  signal.position < 0.5 ? "text-[#c41e3a]" : "text-[#737373]"
                )}>
                  {signal.position.toFixed(1)}x
                </td>
                <td>
                  <span className={clsx(
                    "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
                    signal.signal === 'STRONG_LONG' && "bg-[#00c853]/20 text-[#00c853]",
                    signal.signal === 'LONG' && "bg-[#00c853]/10 text-[#00c853]",
                    signal.signal === 'WEAK_LONG' && "bg-[#737373]/20 text-[#737373]",
                    signal.signal === 'CASH' && "bg-[#c41e3a]/10 text-[#c41e3a]"
                  )}>
                    {signal.signal === 'STRONG_LONG' && <TrendingUp className="w-3 h-3" />}
                    {signal.signal === 'LONG' && <TrendingUp className="w-3 h-3" />}
                    {signal.signal === 'WEAK_LONG' && <Minus className="w-3 h-3" />}
                    {signal.signal === 'CASH' && <TrendingDown className="w-3 h-3" />}
                    {signal.signal.replace('_', ' ')}
                  </span>
                </td>
                <td className={clsx(
                  "text-right font-mono",
                  signal.change > 0 ? "text-[#00c853]" :
                  signal.change < 0 ? "text-[#c41e3a]" : "text-[#737373]"
                )}>
                  <span className="inline-flex items-center gap-1">
                    {signal.change > 0 && <ArrowUp className="w-3 h-3" />}
                    {signal.change < 0 && <ArrowDown className="w-3 h-3" />}
                    {signal.change !== 0 ? `${signal.change > 0 ? '+' : ''}${signal.change.toFixed(1)}x` : '-'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
