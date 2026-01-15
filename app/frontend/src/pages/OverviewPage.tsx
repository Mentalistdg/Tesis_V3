import { useEffect, useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getModels, getMarketData, getModelDetail } from '../services/api';
import type { ModelsResponse, MarketData, ModelSummary, ModelData } from '../types';
import { createChart, IChartApi } from 'lightweight-charts';
import { TrendingUp, Award, Shield, BarChart2, DollarSign } from 'lucide-react';
import { clsx } from 'clsx';
import { calculateSummaryFromTrades, INITIAL_CAPITAL } from '../utils/transactionCosts';

// Chart colors for top 5 models
const CHART_COLORS = ['#c41e3a', '#00c853', '#ff6b35', '#4ecdc4', '#a855f7'];

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);
}

// Chart data type
interface ChartModelData {
  name: string;
  finalCapital: number;
  dates: string[];
  equityCurve: number[];
}

export default function OverviewPage() {
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [modelDetails, setModelDetails] = useState<Record<string, ModelData>>({});
  const [chartData, setChartData] = useState<ChartModelData[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<keyof ModelSummary>('total_return');
  const [sortDesc, setSortDesc] = useState(true);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const navigate = useNavigate();

  // Load all data on mount
  useEffect(() => {
    async function loadAllData() {
      try {
        setLoading(true);

        // Step 1: Load models list and market data
        const [models, market] = await Promise.all([getModels(), getMarketData()]);
        setModelsData(models);
        setMarketData(market);

        // Step 2: Load all model details
        const details: Record<string, ModelData> = {};
        await Promise.all(
          models.models.map(async (m) => {
            try {
              details[m.model] = await getModelDetail(m.model);
            } catch (e) {
              console.error(`Failed to load ${m.model}:`, e);
            }
          })
        );
        setModelDetails(details);

        // Step 3: Get top 5 models for chart (same order as table - by total_return from JSON)
        const sortedModels = [...models.models].sort((a, b) => b.total_return - a.total_return);
        const top5 = sortedModels
          .slice(0, 5)
          .map(m => {
            const detail = details[m.model];
            if (!detail?.trades?.length || !detail?.dates?.length || !detail?.equity_curve?.length) {
              return null;
            }
            const summary = calculateSummaryFromTrades(detail.trades, INITIAL_CAPITAL);
            return {
              name: m.model,
              finalCapital: summary.finalCapital,
              dates: detail.dates,
              equityCurve: detail.equity_curve,
            };
          })
          .filter((m): m is ChartModelData => m !== null);

        setChartData(top5);

      } catch (err) {
        console.error('Failed to load data:', err);
      } finally {
        setLoading(false);
      }
    }

    loadAllData();
  }, []);

  // Build chart when chartData and marketData are ready
  const buildChart = useCallback(() => {
    if (!chartContainerRef.current || !marketData || chartData.length === 0) {
      return;
    }

    // Cleanup previous chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 350,
      layout: {
        background: { color: '#111111' },
        textColor: '#737373',
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: '#1a1a1a' },
        horzLines: { color: '#1a1a1a' },
      },
      rightPriceScale: { borderColor: '#222222' },
      timeScale: { borderColor: '#222222', timeVisible: true },
      crosshair: { mode: 1 },
      localization: {
        priceFormatter: (price: number) => '$' + price.toLocaleString('en-US', { maximumFractionDigits: 0 }),
      },
    });

    chartRef.current = chart;

    // Add each model line
    chartData.forEach((model, idx) => {
      const grossFinal = model.equityCurve[model.equityCurve.length - 1] * INITIAL_CAPITAL;
      const scaleFactor = model.finalCapital / grossFinal;

      const series = chart.addLineSeries({
        color: CHART_COLORS[idx],
        lineWidth: 2,
        title: model.name,
      });

      const lineData = model.dates.map((date, i) => {
        const progress = i / (model.dates.length - 1);
        const factor = 1 + (scaleFactor - 1) * progress;
        return { time: date, value: model.equityCurve[i] * INITIAL_CAPITAL * factor };
      });

      series.setData(lineData as any);
    });

    // Add Buy & Hold benchmark
    const marketSeries = chart.addLineSeries({
      color: '#525252',
      lineWidth: 2,
      lineStyle: 2,
      title: 'Buy & Hold',
    });

    const marketLineData = marketData.dates.map((date, i) => ({
      time: date,
      value: marketData.equity_curve[i] * INITIAL_CAPITAL,
    }));
    marketSeries.setData(marketLineData as any);

    chart.timeScale().fitContent();
  }, [chartData, marketData]);

  // Effect to build chart
  useEffect(() => {
    buildChart();

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [buildChart]);

  // Calculate trade-based summary for a model
  function getModelSummary(modelName: string) {
    const detail = modelDetails[modelName];
    if (!detail?.trades?.length) {
      return {
        finalCapital: INITIAL_CAPITAL,
        totalTxCosts: 0,
        netReturn: 0,
        grossReturn: 0,
        costBreakdown: { expense: 0, trading: 0, volDrag: 0, total: 0 }
      };
    }
    return calculateSummaryFromTrades(detail.trades, INITIAL_CAPITAL);
  }

  function handleSort(key: keyof ModelSummary) {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  }

  function handleRowDoubleClick(modelName: string) {
    navigate(`/detail/${modelName}`);
  }

  if (loading) {
    return <div className="text-center py-8 text-[#737373]">Loading...</div>;
  }

  if (!modelsData || !marketData) {
    return <div className="text-center py-8 text-[#c41e3a]">Failed to load data</div>;
  }

  // Create Buy & Hold as a virtual model for comparison
  const spyFinalValue = modelsData.benchmark.spy_final_value ?? modelsData.benchmark.final_value ?? modelsData.config.initial_capital;
  const spyTotalReturn = modelsData.benchmark.spy_total_return ?? modelsData.benchmark.total_return ?? 0;
  const buyAndHoldModel: ModelSummary = {
    model: 'Buy & Hold (SPY)',
    category: 'Benchmark',
    final_equity: spyFinalValue,
    profit_loss: spyFinalValue - modelsData.config.initial_capital,
    total_return: spyTotalReturn,
    annual_return: marketData.metrics.annual_return,
    market_return: spyTotalReturn,
    excess_return: 0,
    sharpe: marketData.metrics.sharpe,
    sharpe_ci_lower: 0,
    sharpe_ci_upper: 0,
    prob_sharpe_positive: 0,
    sortino: 0,
    calmar: 0,
    max_drawdown: marketData.metrics.max_drawdown,
    win_rate: 0,
    mean_position: 1,
    n_trades: 0,
    pct_3x_long: 0,
    pct_long: 100,
    pct_cash: 0,
    pct_short: 0,
    pct_3x_short: 0,
    transaction_costs: 0,
    optimal_params: { q_3x: 0, q_1x: 0 },
    return_improvement: 0,
    sharpe_improvement: 0,
  };

  const allModels = [...modelsData.models, buyAndHoldModel];

  // Get net return for sorting (what investors actually earn after costs)
  const getNetReturn = (model: ModelSummary): number => {
    if (model.category === 'Benchmark') {
      return model.total_return; // SPY has no tx costs
    }
    const summary = getModelSummary(model.model);
    return summary?.netReturn ?? model.total_return;
  };

  const sortedModels = allModels.sort((a, b) => {
    // When sorting by 'total_return', use net return (actual investor return)
    if (sortKey === 'total_return') {
      const aVal = getNetReturn(a);
      const bVal = getNetReturn(b);
      return sortDesc ? bVal - aVal : aVal - bVal;
    }
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDesc ? bVal - aVal : aVal - bVal;
    }
    return 0;
  });

  // Derive best model from models array (first model in list, or null if empty)
  const bestModel = modelsData.models.length > 0 ? modelsData.models[0] : null;
  const bestSharpe = modelsData.models.length > 0
    ? modelsData.models.reduce((best, m) => m.sharpe > best.sharpe ? m : best)
    : null;
  const lowestDD = modelsData.models.length > 0
    ? modelsData.models.reduce((best, m) => m.max_drawdown > best.max_drawdown ? m : best)
    : null;
  const beatingMarket = modelsData.models_beating_spy ?? modelsData.models_beating_benchmark ?? 0;

  // Calculate trade-based summary for best model (consistent with all pages)
  const bestModelSummary = bestModel ? getModelSummary(bestModel.model) : null;

  // Get top 5 model names for legend (from chartData state)
  const top5Models = chartData.map(m => m.name);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Optimized Strategy Performance</h2>
          <p className="text-sm text-[#737373] mt-1">
            Initial Capital: {formatCurrency(modelsData.config.initial_capital)}{modelsData.config.n_bootstrap ? ` | Bootstrap: ${modelsData.config.n_bootstrap.toLocaleString()} samples` : ''} | Strategy: {modelsData.strategy ?? 'Optimized'}
          </p>
        </div>
        <span className="text-sm text-[#737373]">
          Period: {modelsData.test_period.start} to {modelsData.test_period.end} ({modelsData.test_period.n_days} days)
        </span>
      </div>

      {/* Best Model Highlight - LONG-ONLY */}
      <div className="card bg-gradient-to-r from-[#111111] to-[#1a1a1a] border-l-4 border-[#00c853]">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-2 text-[#00c853] mb-1">
              <Award className="w-5 h-5" />
              <span className="font-medium">Best Model: {bestModel?.model ?? 'N/A'}</span>
              <span className="text-xs bg-[#00c853]/20 px-2 py-0.5 rounded">LONG-ONLY</span>
            </div>
            <div className="text-3xl font-bold text-[#00c853]">
              {bestModelSummary ? formatCurrency(bestModelSummary.finalCapital) : 'N/A'}
            </div>
            <div className="text-sm text-[#737373] mt-1">
              {bestModelSummary && (
                <>P&L: +{formatCurrency(bestModelSummary.finalCapital - INITIAL_CAPITAL)} ({(bestModelSummary.netReturn * 100).toFixed(0)}%)</>
              )}
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="text-sm text-[#737373]">Sharpe</div>
              <div className="text-xl font-semibold text-white">{bestModel?.sharpe?.toFixed(2) ?? 'N/A'}</div>
            </div>
            <div>
              <div className="text-sm text-[#737373]">Return</div>
              <div className="text-xl font-semibold text-[#00c853]">
                +{((bestModel?.total_return ?? 0) * 100).toFixed(0)}%
              </div>
            </div>
            <div>
              <div className="text-sm text-[#737373]">3x UPRO</div>
              <div className="text-xl font-semibold text-white">{(bestModel?.pct_3x_long ?? 0).toFixed(0)}%</div>
            </div>
            <div>
              <div className="text-sm text-[#737373]">Cash</div>
              <div className="text-xl font-semibold text-white">{(bestModel?.pct_cash ?? 0).toFixed(0)}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="flex items-center gap-2 text-[#737373]">
            <TrendingUp className="w-4 h-4" />
            <span className="metric-label">Best Sharpe</span>
          </div>
          <div className="metric-value text-[#c41e3a]">
            {bestSharpe?.sharpe.toFixed(2) ?? 'N/A'}
          </div>
          <div className="text-sm text-[#737373] mt-1">{bestSharpe?.model ?? '-'}</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center gap-2 text-[#737373]">
            <Shield className="w-4 h-4" />
            <span className="metric-label">Lowest Drawdown</span>
          </div>
          <div className="metric-value text-[#00c853]">
            {lowestDD ? (lowestDD.max_drawdown * 100).toFixed(1) : 'N/A'}%
          </div>
          <div className="text-sm text-[#737373] mt-1">{lowestDD?.model ?? '-'}</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center gap-2 text-[#737373]">
            <BarChart2 className="w-4 h-4" />
            <span className="metric-label">Beating SPY</span>
          </div>
          <div className="metric-value text-[#00c853]">
            {beatingMarket}/{modelsData.total_models}
          </div>
          <div className="text-sm text-[#737373] mt-1">
            {((beatingMarket / modelsData.total_models) * 100).toFixed(0)}% of models
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center gap-2 text-[#737373]">
            <DollarSign className="w-4 h-4" />
            <span className="metric-label">SPY Benchmark</span>
          </div>
          <div className="metric-value text-[#525252]">
            {formatCurrency(spyFinalValue)}
          </div>
          <div className="text-sm text-[#737373] mt-1">
            +{(spyTotalReturn * 100).toFixed(0)}% return
          </div>
        </div>
      </div>

      {/* Equity Curves Chart */}
      <div className="card">
        <h3 className="text-lg font-medium mb-2 text-white">Equity Curves - Top 5 Models vs Buy & Hold (Starting: $10,000)</h3>
        <div className="flex flex-wrap gap-3 mb-4 text-xs">
          {top5Models.map((name, idx) => (
            <div key={name} className="flex items-center gap-1">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: CHART_COLORS[idx] }}
              />
              <span className="text-[#a3a3a3]">{name}</span>
            </div>
          ))}
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-[#525252]" style={{ borderStyle: 'dashed' }} />
            <span className="text-[#525252]">Buy & Hold</span>
          </div>
        </div>
        <div ref={chartContainerRef} className="w-full" />
      </div>

      {/* Models Table */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4 text-white">Model Rankings</h3>
        <p className="text-sm text-[#737373] mb-4">Double-click a row to see model details</p>

        <div className="overflow-x-auto">
          <table className="table-dark">
            <thead>
              <tr>
                <th className="w-8">#</th>
                <th
                  className="w-32 cursor-pointer hover:text-white"
                  onClick={() => handleSort('model')}
                >
                  Model
                </th>
                <th
                  className="w-24 text-right cursor-pointer hover:text-white"
                  onClick={() => handleSort('final_equity')}
                >
                  Final Capital {sortKey === 'final_equity' && (sortDesc ? '↓' : '↑')}
                </th>
                <th className="w-16 text-right">Gross Ret</th>
                <th
                  className="w-20 text-right cursor-pointer hover:text-white"
                  onClick={() => handleSort('total_return')}
                >
                  Net Ret {sortKey === 'total_return' && (sortDesc ? '↓' : '↑')}
                </th>
                <th className="w-16 text-right">Annual</th>
                <th className="w-16 text-right">vs SPY</th>
                <th
                  className="w-16 text-right cursor-pointer hover:text-white"
                  onClick={() => handleSort('sharpe')}
                >
                  Sharpe {sortKey === 'sharpe' && (sortDesc ? '↓' : '↑')}
                </th>
                <th
                  className="w-20 text-right cursor-pointer hover:text-white"
                  onClick={() => handleSort('prob_sharpe_positive')}
                >
                  P(S&gt;0) {sortKey === 'prob_sharpe_positive' && (sortDesc ? '↓' : '↑')}
                </th>
                <th
                  className="w-16 text-right cursor-pointer hover:text-white"
                  onClick={() => handleSort('max_drawdown')}
                >
                  Max DD {sortKey === 'max_drawdown' && (sortDesc ? '↓' : '↑')}
                </th>
                <th
                  className="w-14 text-right cursor-pointer hover:text-white"
                  onClick={() => handleSort('n_trades')}
                >
                  Trades {sortKey === 'n_trades' && (sortDesc ? '↓' : '↑')}
                </th>
                <th className="w-20 text-right">Tx Costs</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.map((model, idx) => {
                const isBenchmark = model.category === 'Benchmark';
                // Use trade-based calculation for consistency with Trades page
                const summary = !isBenchmark ? getModelSummary(model.model) : null;
                const grossReturn = summary?.grossReturn ?? model.total_return;
                const netReturn = summary?.netReturn ?? model.total_return;
                const excessReturn = netReturn - spyTotalReturn;
                // Annualize: (1 + total)^(252/days) - 1
                const years = modelsData.test_period.n_days / 252;
                const annualReturn = Math.pow(1 + netReturn, 1 / years) - 1;

                return (
                  <tr
                    key={model.model}
                    className={clsx(
                      isBenchmark ? "bg-[#1a1a1a] border-y border-[#333333]" : "cursor-pointer"
                    )}
                    onDoubleClick={() => !isBenchmark && handleRowDoubleClick(model.model)}
                  >
                    <td className="text-[#525252]">{idx + 1}</td>
                    <td className="font-medium">
                      <div className={isBenchmark ? "text-[#525252]" : "text-white"}>{model.model}</div>
                      <div className="text-xs text-[#525252]">{model.category}</div>
                    </td>
                    <td className={clsx(
                      "text-right font-mono",
                      isBenchmark ? "text-[#525252]" : "text-[#00c853]"
                    )}>
                      {formatCurrency(summary ? summary.finalCapital : (model.final_equity ?? INITIAL_CAPITAL))}
                    </td>
                    <td className={clsx(
                      "text-right font-mono",
                      isBenchmark ? "text-[#525252]" :
                      grossReturn >= 0 ? "text-[#a3a3a3]" : "text-[#737373]"
                    )}>
                      {isBenchmark ? `${(grossReturn * 100).toFixed(0)}%` : `${(grossReturn * 100).toFixed(0)}%`}
                    </td>
                    <td className={clsx(
                      "text-right font-mono font-semibold",
                      isBenchmark ? "text-[#00c853]" :
                      netReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                    )}>
                      {isBenchmark ? `${(netReturn * 100).toFixed(0)}%` : `${(netReturn * 100).toFixed(0)}%`}
                    </td>
                    <td className={clsx(
                      "text-right font-mono",
                      isBenchmark ? "text-[#525252]" :
                      annualReturn >= 0 ? "text-[#a3a3a3]" : "text-[#737373]"
                    )}>
                      {(annualReturn * 100).toFixed(1)}%
                    </td>
                    <td className={clsx(
                      "text-right font-mono font-semibold",
                      isBenchmark ? "text-[#525252]" :
                      excessReturn > 0 ? "text-[#00c853]" :
                      excessReturn < 0 ? "text-[#c41e3a]" : "text-[#737373]"
                    )}>
                      {isBenchmark ? "-" : `${excessReturn >= 0 ? '+' : ''}${(excessReturn * 100).toFixed(0)}%`}
                    </td>
                    <td className="text-right font-mono">
                      <span className={clsx(
                        isBenchmark ? "text-[#525252]" : "font-semibold text-white"
                      )}>
                        {model.sharpe.toFixed(2)}
                      </span>
                      {!isBenchmark && model.sharpe_ci_lower !== undefined && model.sharpe_ci_upper !== undefined && (
                        <div className="text-xs text-[#525252]">
                          [{model.sharpe_ci_lower.toFixed(1)}, {model.sharpe_ci_upper.toFixed(1)}]
                        </div>
                      )}
                    </td>
                    <td className={clsx(
                      "text-right font-mono",
                      isBenchmark ? "text-[#525252]" :
                      (model.prob_sharpe_positive ?? 0) >= 0.99 ? "text-[#00c853]" :
                      (model.prob_sharpe_positive ?? 0) >= 0.95 ? "text-[#a3a3a3]" : "text-[#525252]"
                    )}>
                      {isBenchmark ? "-" : model.prob_sharpe_positive !== undefined ? `${(model.prob_sharpe_positive * 100).toFixed(0)}%` : "-"}
                    </td>
                    <td className={clsx(
                      "text-right font-mono",
                      isBenchmark ? "text-[#525252]" : "text-[#c41e3a]"
                    )}>
                      {(model.max_drawdown * 100).toFixed(1)}%
                    </td>
                    <td className="text-right font-mono text-[#a3a3a3]">
                      {isBenchmark ? "-" : model.n_trades}
                    </td>
                    <td className="text-right font-mono text-[#f59e0b]">
                      {isBenchmark ? "-" : `-$${(summary?.totalTxCosts ?? 0).toFixed(0)}`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Position Distribution & Cost Breakdown - LONG-ONLY */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4 text-white">Position Distribution & Cost Breakdown (LONG-ONLY)</h3>
        <div className="overflow-x-auto">
          <table className="table-dark">
            <thead>
              <tr>
                <th className="w-32">Model</th>
                <th className="w-24 text-center">+3x UPRO</th>
                <th className="w-24 text-center">+1x SPY</th>
                <th className="w-24 text-center">Cash</th>
                <th className="w-20 text-right">Trading</th>
                <th className="w-20 text-right">Expense</th>
                <th className="w-20 text-right">VolDrag</th>
                <th className="w-20 text-right">Total</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.filter(m => m.category !== 'Benchmark').map((model) => {
                const summary = getModelSummary(model.model);
                const costs = summary?.costBreakdown;
                // For LONG-ONLY: pct_3x = UPRO, pct_1x = SPY, pct_cash = Cash
                const pctUpro = model.pct_3x ?? model.pct_3x_long ?? 0;
                const pctSpy = model.pct_1x ?? (model.pct_long - (model.pct_3x ?? model.pct_3x_long ?? 0));
                const pctCash = model.pct_cash ?? 0;
                return (
                  <tr key={model.model}>
                    <td className="font-medium text-white">{model.model}</td>
                    <td className="text-center">
                      <div className="inline-block w-12 h-3 rounded" style={{
                        background: `linear-gradient(to right, #00c853 ${pctUpro}%, #1a1a1a ${pctUpro}%)`,
                      }} />
                      <span className="ml-1 text-xs text-[#a3a3a3]">{pctUpro.toFixed(0)}%</span>
                    </td>
                    <td className="text-center">
                      <div className="inline-block w-12 h-3 rounded" style={{
                        background: `linear-gradient(to right, #22d3ee ${pctSpy}%, #1a1a1a ${pctSpy}%)`,
                      }} />
                      <span className="ml-1 text-xs text-[#a3a3a3]">{pctSpy.toFixed(0)}%</span>
                    </td>
                    <td className="text-center">
                      <div className="inline-block w-12 h-3 rounded" style={{
                        background: `linear-gradient(to right, #525252 ${pctCash}%, #1a1a1a ${pctCash}%)`,
                      }} />
                      <span className="ml-1 text-xs text-[#a3a3a3]">{pctCash.toFixed(0)}%</span>
                    </td>
                    <td className="text-right font-mono text-[#a3a3a3]">
                      ${(costs?.trading ?? 0).toFixed(0)}
                    </td>
                    <td className="text-right font-mono text-[#a3a3a3]">
                      ${(costs?.expense ?? 0).toFixed(0)}
                    </td>
                    <td className="text-right font-mono text-[#a3a3a3]">
                      ${(costs?.volDrag ?? 0).toFixed(0)}
                    </td>
                    <td className="text-right font-mono text-[#f59e0b] font-semibold">
                      -${(costs?.total ?? 0).toFixed(0)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
