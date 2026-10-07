import { useEffect, useState, useMemo } from 'react';
import { getModels, getModelDetail } from '../services/api';
import type { ModelsResponse, ModelData, Trade } from '../types';
import { Info } from 'lucide-react';
import { clsx } from 'clsx';
import {
  calculateSummaryFromTrades,
  INITIAL_CAPITAL,
  type CostBreakdown,
} from '../utils/transactionCosts';
import LoadingScreen from '../components/LoadingScreen';
import { useTableSort } from '../hooks/useTableSort';
import SortableHeader from '../components/SortableHeader';

interface ModelCostRow {
  model: string;
  category: string;
  nTrades: number;
  grossReturn: number;
  netReturn: number;
  finalCapital: number;
  costBreakdown: CostBreakdown;
}

export default function CostsPage() {
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [modelDetails, setModelDetails] = useState<Record<string, ModelData>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      const data = await getModels();
      setModelsData(data);

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

  // Build cost rows for all models (hooks must be above early returns)
  const modelCostsRaw: ModelCostRow[] = useMemo(() => {
    if (!modelsData) return [];
    return modelsData.models.map((model) => {
      const detail = modelDetails[model.model];
      if (!detail || !detail.trades || detail.trades.length === 0) {
        return {
          model: model.model,
          category: model.category,
          nTrades: model.n_trades,
          grossReturn: 0,
          netReturn: 0,
          finalCapital: INITIAL_CAPITAL,
          costBreakdown: { expense: 0, trading: 0, volDrag: 0, total: 0 },
        };
      }
      const activeTrades = detail.trades.filter((t: Trade) => Math.abs(t.entry_position) >= 0.5);
      const summary = calculateSummaryFromTrades(detail.trades, INITIAL_CAPITAL);
      return {
        model: model.model,
        category: model.category,
        nTrades: activeTrades.length,
        grossReturn: summary.grossReturn,
        netReturn: summary.netReturn,
        finalCapital: summary.finalCapital,
        costBreakdown: summary.costBreakdown,
      };
    });
  }, [modelsData, modelDetails]);

  const costsAccessor = useMemo(() => ({
    model: (r: ModelCostRow) => r.model,
    nTrades: (r: ModelCostRow) => r.nTrades,
    grossReturn: (r: ModelCostRow) => r.grossReturn,
    expense: (r: ModelCostRow) => r.costBreakdown.expense,
    trading: (r: ModelCostRow) => r.costBreakdown.trading,
    volDrag: (r: ModelCostRow) => r.costBreakdown.volDrag,
    totalCosts: (r: ModelCostRow) => r.costBreakdown.total,
    netReturn: (r: ModelCostRow) => r.netReturn,
    costImpact: (r: ModelCostRow) => (r.grossReturn - r.netReturn) * 10000,
  }), []);

  const { sortedData: modelCosts, requestSort: requestCostSort, getSortDirection: getCostSortDir } =
    useTableSort(modelCostsRaw, 'netReturn', 'desc', costsAccessor);

  if (loading) {
    return <LoadingScreen />;
  }

  if (!modelsData) {
    return <div className="text-center py-8 text-[#c41e3a]">Failed to load data</div>;
  }

  // Aggregate KPIs
  const n = modelCostsRaw.length;
  const avgTotalCost = n > 0 ? modelCostsRaw.reduce((s, m) => s + m.costBreakdown.total, 0) / n : 0;
  const avgExpense = n > 0 ? modelCostsRaw.reduce((s, m) => s + m.costBreakdown.expense, 0) / n : 0;
  const avgTrading = n > 0 ? modelCostsRaw.reduce((s, m) => s + m.costBreakdown.trading, 0) / n : 0;
  const avgVolDrag = n > 0 ? modelCostsRaw.reduce((s, m) => s + m.costBreakdown.volDrag, 0) / n : 0;

  const testPeriod = modelsData.test_period;
  const startDate = testPeriod?.start ?? '';
  const endDate = testPeriod?.end ?? '';

  // Top 7 models for the visualization
  const topModels = modelCosts.slice(0, 7);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold">Transaction Cost Analysis</h2>
        <p className="text-sm text-[#737373] mt-1">
          {startDate} to {endDate} &middot; Initial capital: ${INITIAL_CAPITAL.toLocaleString()}
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          label="Avg Total Cost"
          dollars={avgTotalCost}
          initial={INITIAL_CAPITAL}
        />
        <KpiCard
          label="Avg Expense Ratio"
          dollars={avgExpense}
          initial={INITIAL_CAPITAL}
          sub="Annual ETF fees"
        />
        <KpiCard
          label="Avg Bid-Ask Spread"
          dollars={avgTrading}
          initial={INITIAL_CAPITAL}
          sub="Entry + exit spreads"
        />
        <KpiCard
          label="Avg Volatility Drag"
          dollars={avgVolDrag}
          initial={INITIAL_CAPITAL}
          sub="3x leverage rebalancing"
        />
      </div>

      {/* Cost Type Explanation */}
      <div className="card border-l-4 border-[#f59e0b]">
        <div className="flex items-center gap-2 mb-4">
          <Info className="w-5 h-5 text-[#f59e0b]" />
          <h3 className="text-base font-medium">Three Sources of Transaction Costs</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <CostExplanation
            title="Expense Ratio"
            formula="annual_fee / 252 trading days"
            details={[
              'UPRO (3x): 0.91% / year',
              'SPY (1x): 0.09% / year',
              'Cash: 0%',
            ]}
          />
          <CostExplanation
            title="Bid-Ask Spread"
            formula="(entry + exit) spread per trade"
            details={[
              'UPRO: 0.05% × 2 = 0.10%',
              'SPY: 0.02% × 2 = 0.04%',
              'Applied on position changes',
            ]}
          />
          <CostExplanation
            title="Volatility Drag"
            formula="daily drag from 3x leverage rebalancing"
            details={[
              'Only applies to UPRO days',
              'Compounds over holding period',
              'Largest cost for leveraged positions',
            ]}
          />
        </div>
      </div>

      {/* Cost Impact Table */}
      <div className="card">
        <h3 className="text-base font-medium mb-4">Cost Impact by Model</h3>
        <div className="overflow-x-auto">
          <table className="table-dark">
            <thead>
              <tr>
                <th className="w-8">#</th>
                <SortableHeader label="Model" sortKey="model" activeDirection={getCostSortDir('model')} onSort={requestCostSort} />
                <SortableHeader label="Trades" sortKey="nTrades" activeDirection={getCostSortDir('nTrades')} onSort={requestCostSort} className="text-right" />
                <SortableHeader label="Gross Return" sortKey="grossReturn" activeDirection={getCostSortDir('grossReturn')} onSort={requestCostSort} className="text-right" />
                <SortableHeader label="Expense" sortKey="expense" activeDirection={getCostSortDir('expense')} onSort={requestCostSort} className="text-right" />
                <SortableHeader label="Trading" sortKey="trading" activeDirection={getCostSortDir('trading')} onSort={requestCostSort} className="text-right" />
                <SortableHeader label="Vol Drag" sortKey="volDrag" activeDirection={getCostSortDir('volDrag')} onSort={requestCostSort} className="text-right" />
                <SortableHeader label="Total Costs" sortKey="totalCosts" activeDirection={getCostSortDir('totalCosts')} onSort={requestCostSort} className="text-right" />
                <SortableHeader label="Net Return" sortKey="netReturn" activeDirection={getCostSortDir('netReturn')} onSort={requestCostSort} className="text-right" />
                <SortableHeader label="Cost Impact" sortKey="costImpact" activeDirection={getCostSortDir('costImpact')} onSort={requestCostSort} className="text-right" />
              </tr>
            </thead>
            <tbody>
              {modelCosts.map((row, idx) => {
                const costImpactBps = (row.grossReturn - row.netReturn) * 10000;
                const grossCapital = row.finalCapital + row.costBreakdown.total;
                return (
                  <tr key={row.model}>
                    <td className="text-[#525252]">{idx + 1}</td>
                    <td className="font-medium">{row.model}</td>
                    <td className="text-right font-mono text-[#737373]">{row.nTrades}</td>
                    <td className={clsx(
                      'text-right font-mono',
                      row.grossReturn >= 0 ? 'text-[#00c853]' : 'text-[#c41e3a]'
                    )}>
                      ${grossCapital.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right font-mono text-[#f59e0b]">
                      -${row.costBreakdown.expense.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right font-mono text-[#f59e0b]">
                      -${row.costBreakdown.trading.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right font-mono text-[#f59e0b]">
                      -${row.costBreakdown.volDrag.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right font-mono text-[#f59e0b] font-medium">
                      -${row.costBreakdown.total.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className={clsx(
                      'text-right font-mono font-medium',
                      row.netReturn >= 0 ? 'text-[#00c853]' : 'text-[#c41e3a]'
                    )}>
                      {row.netReturn >= 0 ? '+' : ''}{(row.netReturn * 100).toFixed(1)}%
                    </td>
                    <td className="text-right font-mono text-[#737373]">
                      -{costImpactBps.toFixed(0)} bps
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cost Breakdown Visualization */}
      <div className="card">
        <h3 className="text-base font-medium mb-4">Cost Composition — Top Models</h3>
        <p className="text-xs text-[#525252] mb-4">Proportion of each cost type relative to total costs</p>
        <div className="space-y-3">
          {topModels.map((row) => {
            const total = row.costBreakdown.total || 1;
            const expPct = (row.costBreakdown.expense / total) * 100;
            const tradPct = (row.costBreakdown.trading / total) * 100;
            const volPct = (row.costBreakdown.volDrag / total) * 100;
            return (
              <div key={row.model} className="flex items-center gap-3">
                <div className="w-32 text-xs font-medium truncate text-right">{row.model}</div>
                <div className="flex-1 h-5 bg-[#0a0a0a] rounded overflow-hidden flex">
                  {expPct > 0 && (
                    <div
                      className="h-full bg-[#f59e0b]"
                      style={{ width: `${expPct}%` }}
                      title={`Expense: ${expPct.toFixed(1)}%`}
                    />
                  )}
                  {tradPct > 0 && (
                    <div
                      className="h-full bg-[#3b82f6]"
                      style={{ width: `${tradPct}%` }}
                      title={`Trading: ${tradPct.toFixed(1)}%`}
                    />
                  )}
                  {volPct > 0 && (
                    <div
                      className="h-full bg-[#c41e3a]"
                      style={{ width: `${volPct}%` }}
                      title={`Vol Drag: ${volPct.toFixed(1)}%`}
                    />
                  )}
                </div>
                <div className="w-24 text-right text-xs font-mono text-[#737373]">
                  ${row.costBreakdown.total.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                </div>
              </div>
            );
          })}
        </div>
        <div className="flex items-center gap-5 mt-4 text-xs text-[#737373]">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-sm bg-[#f59e0b]" /> Expense Ratio
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-sm bg-[#3b82f6]" /> Bid-Ask Spread
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-sm bg-[#c41e3a]" /> Volatility Drag
          </span>
        </div>
      </div>
    </div>
  );
}

/* ---------- Sub-components ---------- */

function KpiCard({ label, dollars, initial, sub }: {
  label: string;
  dollars: number;
  initial: number;
  sub?: string;
}) {
  const pct = (dollars / initial) * 100;
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="text-xl font-semibold mt-2 text-[#f59e0b]" style={{ fontVariantNumeric: 'tabular-nums' }}>
        ${dollars.toLocaleString('en-US', { maximumFractionDigits: 0 })}
      </div>
      <div className="text-xs text-[#525252] mt-1">
        {pct.toFixed(2)}% of capital{sub ? ` · ${sub}` : ''}
      </div>
    </div>
  );
}

function CostExplanation({ title, formula, details }: {
  title: string;
  formula: string;
  details: string[];
}) {
  return (
    <div className="bg-[#0d0d0d] rounded p-4">
      <div className="text-sm font-medium text-[#f5f5f5] mb-1">{title}</div>
      <div className="text-xs text-[#f59e0b] font-mono mb-2">{formula}</div>
      <ul className="space-y-1">
        {details.map((d, i) => (
          <li key={i} className="text-xs text-[#737373]">{d}</li>
        ))}
      </ul>
    </div>
  );
}
