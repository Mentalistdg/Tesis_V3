import { useEffect, useState } from 'react';
import { getModels, getModelDetail } from '../services/api';
import type { ModelsResponse, ModelData } from '../types';
import { DollarSign, Calculator, TrendingDown, Info } from 'lucide-react';
import { clsx } from 'clsx';
import {
  calculateSummaryFromTrades,
  INITIAL_CAPITAL,
  TRANSACTION_COST_BREAKDOWN
} from '../utils/transactionCosts';

export default function CostsPage() {
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [modelDetails, setModelDetails] = useState<Record<string, ModelData>>({});
  const [loading, setLoading] = useState(true);

  // Additional cost parameters (default 0 to match other pages)
  const [commission, setCommission] = useState(0);
  const [slippageBps, setSlippageBps] = useState(0);
  const [spreadBps, setSpreadBps] = useState(0);
  const [marginInterest, setMarginInterest] = useState(0);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      const data = await getModels();
      setModelsData(data);

      // Load all model details for trade-based calculations
      const details: Record<string, ModelData> = {};
      await Promise.all(
        data.models.map(async (m) => {
          try {
            const detail = await getModelDetail(m.model);
            details[m.model] = detail;
          } catch (e) {
            console.error(`Failed to load ${m.model}:`, e);
          }
        })
      );
      setModelDetails(details);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  // Get trade-based summary for a model (consistent with Trades page)
  function getTradeSummary(modelName: string) {
    const detail = modelDetails[modelName];
    if (!detail || !detail.trades || detail.trades.length === 0) {
      return { finalCapital: INITIAL_CAPITAL, totalTxCosts: 0, netReturn: 0, grossReturn: 0 };
    }
    return calculateSummaryFromTrades(detail.trades, INITIAL_CAPITAL);
  }

  function calculateNetReturns(model: any) {
    const nTrades = model.n_trades || 0;
    const days = modelsData?.test_period.n_days || 252;
    const initialCapital = modelsData?.config.initial_capital || INITIAL_CAPITAL;

    // Get trade-based summary (consistent with Trades page)
    const tradeSummary = getTradeSummary(model.model);

    // ETF costs from trade-based calculation
    const etfCostDollars = tradeSummary.totalTxCosts;
    const grossReturn = tradeSummary.grossReturn;
    const netReturnAfterEtf = tradeSummary.netReturn;

    // ETF cost as % of initial capital (consistent with additional costs which are also simple %)
    const etfCostPct = etfCostDollars / initialCapital;

    // Additional broker costs (user-adjustable)
    const slippageCost = nTrades * (slippageBps / 10000);
    const spreadCost = nTrades * (spreadBps / 10000);
    const commissionCost = (nTrades * commission) / initialCapital;

    // Margin cost: based on time in leveraged positions (3x UPRO = 2x borrowed portion)
    // When using 3x leverage, you're borrowing 2x (3x - 1x = 2x borrowed)
    const pct3x = (model.pct_3x_long ?? model.pct_3x ?? 0) / 100;
    const marginCost = pct3x * 2 * (marginInterest / 100) * (days / 252);

    const additionalCosts = slippageCost + spreadCost + commissionCost + marginCost;
    const totalCostPct = etfCostPct + additionalCosts;
    const finalNetReturn = netReturnAfterEtf - additionalCosts;

    return {
      grossReturn,
      netReturnAfterEtf,
      finalNetReturn,
      finalCapital: tradeSummary.finalCapital - (additionalCosts * initialCapital),
      totalCostPct,
      etfCostPct,
      etfCostDollars,
      additionalCosts,
      breakdown: {
        etf: etfCostPct,
        slippage: slippageCost,
        spread: spreadCost,
        commission: commissionCost,
        margin: marginCost
      }
    };
  }

  if (loading) {
    return <div className="text-center py-8 text-[#737373]">Loading...</div>;
  }

  if (!modelsData) {
    return <div className="text-center py-8 text-[#c41e3a]">Failed to load data</div>;
  }

  const modelCosts = modelsData.models.map(model => ({
    model: model.model,
    category: model.category,
    nTrades: model.n_trades,
    avgLeverage: model.mean_position,
    ...calculateNetReturns(model)
  })).sort((a, b) => b.finalNetReturn - a.finalNetReturn);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Transaction Cost Analysis</h2>
      </div>

      {/* ETF Bid-Ask Spreads - Base Costs (LONG-ONLY) */}
      <div className="card border-l-4 border-[#f59e0b]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Info className="w-5 h-5 text-[#f59e0b]" />
            <h3 className="text-lg font-medium">Base Transaction Costs (Long-Only ETF Bid-Ask Spreads)</h3>
          </div>
          <span className="text-xs bg-[#f59e0b]/20 text-[#f59e0b] px-2 py-1 rounded">Long-Only Strategy</span>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(TRANSACTION_COST_BREAKDOWN)
            .filter(([pos]) => ['+3x (UPRO)', '+1x (SPY)', '0 (Cash)'].includes(pos))
            .map(([position, data]) => (
            <div key={position} className="bg-[#1a1a1a] rounded p-3 text-center">
              <div className="text-xs text-[#737373] uppercase mb-1">{position}</div>
              <div className="text-lg font-semibold text-[#f59e0b]">{data.label}</div>
              <div className="text-xs text-[#525252] mt-1">round-trip</div>
            </div>
          ))}
        </div>
        <div className="mt-3 text-xs text-[#525252]">
          Long-Only strategy uses only UPRO (3x), SPY (1x), and Cash. Each trade incurs entry (buy) + exit (sell) costs = 2× bid-ask spread.
        </div>
      </div>

      {/* Additional Cost Parameters */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Calculator className="w-5 h-5 text-[#737373]" />
            <h3 className="text-lg font-medium">Additional Broker Costs (Optional)</h3>
          </div>
          <button
            onClick={() => {
              setCommission(0);
              setSlippageBps(0);
              setSpreadBps(0);
              setMarginInterest(0);
            }}
            className="text-xs text-[#737373] hover:text-white px-2 py-1 rounded border border-[#333] hover:border-[#525252] transition-colors"
          >
            Reset to 0
          </button>
        </div>
        <p className="text-sm text-[#525252] mb-4">
          Adjust these parameters to simulate additional broker costs beyond ETF spreads. Set all to 0 to match other pages.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-[#737373] mb-1">Commission per Trade ($)</label>
            <input
              type="number"
              value={commission}
              onChange={(e) => setCommission(parseFloat(e.target.value) || 0)}
              className="w-full bg-[#1a1a1a] border border-[#222222] rounded px-3 py-2 text-sm"
              step="0.01"
              min="0"
            />
          </div>
          <div>
            <label className="block text-sm text-[#737373] mb-1">Slippage (bps)</label>
            <input
              type="number"
              value={slippageBps}
              onChange={(e) => setSlippageBps(parseFloat(e.target.value) || 0)}
              className="w-full bg-[#1a1a1a] border border-[#222222] rounded px-3 py-2 text-sm"
              step="1"
              min="0"
            />
          </div>
          <div>
            <label className="block text-sm text-[#737373] mb-1">Spread (bps)</label>
            <input
              type="number"
              value={spreadBps}
              onChange={(e) => setSpreadBps(parseFloat(e.target.value) || 0)}
              className="w-full bg-[#1a1a1a] border border-[#222222] rounded px-3 py-2 text-sm"
              step="1"
              min="0"
            />
          </div>
          <div>
            <label className="block text-sm text-[#737373] mb-1">Margin Interest (% annual)</label>
            <input
              type="number"
              value={marginInterest}
              onChange={(e) => setMarginInterest(parseFloat(e.target.value) || 0)}
              className="w-full bg-[#1a1a1a] border border-[#222222] rounded px-3 py-2 text-sm"
              step="0.1"
              min="0"
            />
          </div>
        </div>
      </div>


      {/* Impact Summary */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <TrendingDown className="w-5 h-5 text-[#c41e3a]" />
          <h3 className="text-lg font-medium">Cost Impact by Model</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="table-dark">
            <thead>
              <tr>
                <th className="w-8">#</th>
                <th className="w-36">Model</th>
                <th className="w-16 text-right">Trades</th>
                <th className="w-24 text-right">Gross Cap</th>
                <th className="w-20 text-right">ETF Costs</th>
                <th className="w-20 text-right">Add. Costs</th>
                <th className="w-24 text-right">Total Costs</th>
                <th className="w-24 text-right">Net Cap</th>
                <th className="w-20 text-right">Net Ret</th>
              </tr>
            </thead>
            <tbody>
              {modelCosts.map((model, idx) => {
                const additionalCostsDollars = model.additionalCosts * INITIAL_CAPITAL;
                const totalCostsDollars = model.etfCostDollars + additionalCostsDollars;
                // Gross Capital = Net Capital + Total Costs (so Gross - Costs = Net)
                const grossCapital = model.finalCapital + totalCostsDollars;
                return (
                  <tr key={model.model}>
                    <td className="text-[#737373]">{idx + 1}</td>
                    <td className="font-medium">{model.model}</td>
                    <td className="text-right font-mono text-[#737373]">{model.nTrades}</td>
                    <td className={clsx(
                      "text-right font-mono",
                      model.grossReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                    )}>
                      ${grossCapital.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right font-mono text-[#f59e0b]">
                      -${model.etfCostDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right font-mono text-[#c41e3a]">
                      {additionalCostsDollars > 0 ? `-$${additionalCostsDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '$0'}
                    </td>
                    <td className="text-right font-mono text-[#f59e0b] font-medium">
                      -${totalCostsDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className={clsx(
                      "text-right font-mono font-medium",
                      model.finalCapital >= INITIAL_CAPITAL ? "text-[#00c853]" : "text-[#c41e3a]"
                    )}>
                      ${model.finalCapital.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className={clsx(
                      "text-right font-mono",
                      model.finalNetReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                    )}>
                      {(model.finalNetReturn * 100).toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cost Breakdown for Top Model */}
      {modelCosts.length > 0 && (() => {
        const topModel = modelCosts[0];
        const addCostsDollars = topModel.additionalCosts * INITIAL_CAPITAL;
        const totalCostsDollars = topModel.etfCostDollars + addCostsDollars;
        const slippageDollars = topModel.breakdown.slippage * INITIAL_CAPITAL;
        const spreadDollars = topModel.breakdown.spread * INITIAL_CAPITAL;
        const commissionDollars = topModel.breakdown.commission * INITIAL_CAPITAL;
        const marginDollars = topModel.breakdown.margin * INITIAL_CAPITAL;

        return (
          <div className="card">
            <div className="flex items-center gap-2 mb-4">
              <DollarSign className="w-5 h-5 text-[#737373]" />
              <h3 className="text-lg font-medium">
                Cost Breakdown: {topModel.model}
              </h3>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="bg-[#1a1a1a] rounded p-4 border-l-2 border-[#f59e0b]">
                <div className="text-xs text-[#737373] uppercase">ETF Bid-Ask</div>
                <div className="text-xl font-semibold text-[#f59e0b] mt-1">
                  -${topModel.etfCostDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                </div>
                <div className="text-xs text-[#525252] mt-1">
                  {topModel.nTrades} trades
                </div>
              </div>
              <div className="bg-[#1a1a1a] rounded p-4">
                <div className="text-xs text-[#737373] uppercase">Slippage</div>
                <div className="text-xl font-semibold text-[#c41e3a] mt-1">
                  {slippageDollars > 0 ? `-$${slippageDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '$0'}
                </div>
              </div>
              <div className="bg-[#1a1a1a] rounded p-4">
                <div className="text-xs text-[#737373] uppercase">Spread</div>
                <div className="text-xl font-semibold text-[#c41e3a] mt-1">
                  {spreadDollars > 0 ? `-$${spreadDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '$0'}
                </div>
              </div>
              <div className="bg-[#1a1a1a] rounded p-4">
                <div className="text-xs text-[#737373] uppercase">Commission</div>
                <div className="text-xl font-semibold text-[#c41e3a] mt-1">
                  {commissionDollars > 0 ? `-$${commissionDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '$0'}
                </div>
              </div>
              <div className="bg-[#1a1a1a] rounded p-4">
                <div className="text-xs text-[#737373] uppercase">Margin Interest</div>
                <div className="text-xl font-semibold text-[#c41e3a] mt-1">
                  {marginDollars > 0 ? `-$${marginDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '$0'}
                </div>
              </div>
            </div>

            <div className="mt-4 p-3 bg-[#1a1a1a] rounded text-sm text-[#737373]">
              With {topModel.nTrades} trades, total costs: <span className="text-[#f59e0b] font-medium">-${totalCostsDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span>
              {' '}(ETF: ${topModel.etfCostDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })} + Additional: ${addCostsDollars.toLocaleString('en-US', { maximumFractionDigits: 0 })})
            </div>
          </div>
        );
      })()}
    </div>
  );
}
