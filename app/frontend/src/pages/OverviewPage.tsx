import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { getModels, getMarketData, getModelDetail } from '../services/api';
import type { ModelsResponse, MarketData, ModelSummary, ModelData, Trade } from '../types';
import { createChart, IChartApi } from 'lightweight-charts';
import { TrendingUp, Award, Shield, BarChart2, DollarSign } from 'lucide-react';
import { clsx } from 'clsx';
import { calculateSummaryFromTrades, INITIAL_CAPITAL } from '../utils/transactionCosts';
import LoadingScreen from '../components/LoadingScreen';
import { useTableSort } from '../hooks/useTableSort';
import SortableHeader from '../components/SortableHeader';

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
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const navigate = useNavigate();

  // Load all data on mount, with minimum 4s splash for branding
  useEffect(() => {
    async function loadAllData() {
      const splashMin = new Promise(resolve => setTimeout(resolve, 4000));

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

        // Step 3: Get top 5 models for chart
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

        // Wait for splash minimum before hiding
        await splashMin;

      } catch (err) {
        console.error('Failed to load data:', err);
        await splashMin;
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

  function handleRowDoubleClick(modelName: string) {
    navigate(`/detail/${modelName}`);
  }

  // Hooks must be above early returns
  const allModels = useMemo(() => {
    if (!modelsData || !marketData) return [];
    const spyFinalValue = modelsData.benchmark.spy_final_value ?? modelsData.benchmark.final_value ?? modelsData.config.initial_capital;
    const spyTotalReturn = modelsData.benchmark.total_return ?? 0;
    const spyReturns = marketData.returns ?? [];
    const spyYears = spyReturns.length / 252;
    const spyAnnualReturn = spyYears > 0 ? Math.pow(1 + spyTotalReturn, 1 / spyYears) - 1 : 0;
    const spySharpe = modelsData.benchmark.sharpe ?? 0;
    const spyMaxDD = modelsData.benchmark.max_drawdown ?? 0;
    const buyAndHoldModel: ModelSummary = {
      model: 'Buy & Hold (SPY)',
      category: 'Benchmark',
      final_equity: spyFinalValue,
      profit_loss: spyFinalValue - modelsData.config.initial_capital,
      total_return: spyTotalReturn,
      annual_return: spyAnnualReturn,
      market_return: spyTotalReturn,
      excess_return: 0,
      sharpe: spySharpe,
      sortino: 0,
      calmar: 0,
      max_drawdown: spyMaxDD,
      n_trades: 0,
      pct_long: 100,
      pct_cash: 0,
    };
    return [...modelsData.models, buyAndHoldModel];
  }, [modelsData, marketData]);

  const overviewAccessor = useMemo(() => ({
    model: (m: ModelSummary) => m.model,
    final_equity: (m: ModelSummary) => m.final_equity ?? 0,
    total_return: (m: ModelSummary) => {
      if (m.category === 'Benchmark') return m.total_return;
      const summary = getModelSummary(m.model);
      return summary?.netReturn ?? m.total_return;
    },
    sharpe: (m: ModelSummary) => m.sharpe,
    max_drawdown: (m: ModelSummary) => m.max_drawdown,
    n_trades: (m: ModelSummary) => {
      if (m.category === 'Benchmark') return 0;
      const detail = modelDetails[m.model];
      if (!detail?.trades?.length) return m.n_trades;
      return detail.trades.filter((t: Trade) => Math.abs(t.entry_position) >= 0.5).length;
    },
  }), [modelDetails]);

  const {
    sortedData: sortedModels, requestSort: requestOverviewSort,
    getSortDirection: getOverviewSortDir,
  } = useTableSort(allModels, 'total_return', 'desc', overviewAccessor);

  if (loading) {
    return <LoadingScreen />;
  }

  if (!modelsData || !marketData) {
    return <div className="text-center py-8 text-[#c41e3a]">Failed to load data</div>;
  }

  // Derive values from loaded data
  const spyFinalValue = modelsData.benchmark.spy_final_value ?? modelsData.benchmark.final_value ?? modelsData.config.initial_capital;
  const spyTotalReturn = modelsData.benchmark.total_return ?? 0;

  // Derive best model
  const bestModel = modelsData.models.length > 0 ? modelsData.models[0] : null;
  const bestSharpe = modelsData.models.length > 0
    ? modelsData.models.reduce((best, m) => m.sharpe > best.sharpe ? m : best)
    : null;
  const lowestDD = modelsData.models.length > 0
    ? modelsData.models.reduce((best, m) => m.max_drawdown > best.max_drawdown ? m : best)
    : null;
  const beatingMarket = modelsData.models_beating_spy ?? 0;

  const bestModelSummary = bestModel ? getModelSummary(bestModel.model) : null;

  // Get top 5 model names for legend
  const top5Models = chartData.map(m => m.name);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Optimized Strategy Performance</h2>
          <p className="text-sm text-[#737373] mt-1">
            Initial Capital: {formatCurrency(modelsData.config.initial_capital)} | Strategy: {modelsData.strategy ?? 'LONG-ONLY'}
          </p>
        </div>
        <span className="text-sm text-[#737373]">
          Period: {modelsData.test_period.start} to {modelsData.test_period.end} ({modelsData.test_period.n_days} days)
        </span>
      </div>

      {/* Best Model Highlight */}
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
                <>P&L: +{formatCurrency(bestModelSummary.finalCapital - INITIAL_CAPITAL)} ({(bestModelSummary.netReturn * 100).toFixed(1)}%)</>
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
                +{((bestModel?.total_return ?? 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-sm text-[#737373]">3x UPRO</div>
              <div className="text-xl font-semibold text-white">{(bestModel?.pct_3x ?? 0).toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-sm text-[#737373]">Cash</div>
              <div className="text-xl font-semibold text-white">{(bestModel?.pct_cash ?? 0).toFixed(1)}%</div>
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
            {((beatingMarket / modelsData.total_models) * 100).toFixed(1)}% of models
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
            +{(spyTotalReturn * 100).toFixed(1)}% return
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

      {/* Models Table - Simplified */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4 text-white">Model Rankings</h3>
        <p className="text-xs text-[#525252] mb-4">Double-click a row to see model details</p>

        <div className="overflow-x-auto">
          <table className="table-dark">
            <thead>
              <tr>
                <th className="w-8">#</th>
                <SortableHeader label="Model" sortKey="model" activeDirection={getOverviewSortDir('model')} onSort={requestOverviewSort} />
                <SortableHeader label="Final Capital" sortKey="final_equity" activeDirection={getOverviewSortDir('final_equity')} onSort={requestOverviewSort} className="text-right" />
                <SortableHeader label="Net Return" sortKey="total_return" activeDirection={getOverviewSortDir('total_return')} onSort={requestOverviewSort} className="text-right" />
                <th className="text-right">vs SPY</th>
                <SortableHeader label="Sharpe" sortKey="sharpe" activeDirection={getOverviewSortDir('sharpe')} onSort={requestOverviewSort} className="text-right" />
                <SortableHeader label="Max DD" sortKey="max_drawdown" activeDirection={getOverviewSortDir('max_drawdown')} onSort={requestOverviewSort} className="text-right" />
                <SortableHeader label="Trades" sortKey="n_trades" activeDirection={getOverviewSortDir('n_trades')} onSort={requestOverviewSort} className="text-right" />
              </tr>
            </thead>
            <tbody>
              {sortedModels.map((model, idx) => {
                const isBenchmark = model.category === 'Benchmark';
                const summary = !isBenchmark ? getModelSummary(model.model) : null;
                const netReturn = summary?.netReturn ?? model.total_return;
                const excessReturn = netReturn - spyTotalReturn;

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
                      "text-right font-mono font-semibold",
                      isBenchmark ? "text-[#525252]" :
                      netReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                    )}>
                      {netReturn >= 0 ? '+' : ''}{(netReturn * 100).toFixed(1)}%
                    </td>
                    <td className={clsx(
                      "text-right font-mono font-semibold",
                      isBenchmark ? "text-[#525252]" :
                      excessReturn > 0 ? "text-[#00c853]" :
                      excessReturn < 0 ? "text-[#c41e3a]" : "text-[#737373]"
                    )}>
                      {isBenchmark ? "-" : `${excessReturn >= 0 ? '+' : ''}${(excessReturn * 100).toFixed(1)}%`}
                    </td>
                    <td className="text-right font-mono">
                      <span className={clsx(
                        isBenchmark ? "text-[#525252]" : "font-semibold text-white"
                      )}>
                        {model.sharpe.toFixed(2)}
                      </span>
                    </td>
                    <td className={clsx(
                      "text-right font-mono",
                      isBenchmark ? "text-[#525252]" : "text-[#c41e3a]"
                    )}>
                      {(model.max_drawdown * 100).toFixed(1)}%
                    </td>
                    <td className="text-right font-mono text-[#a3a3a3]">
                      {isBenchmark ? "-" : (() => {
                        const detail = modelDetails[model.model];
                        if (!detail?.trades?.length) return model.n_trades;
                        return detail.trades.filter((t: Trade) => Math.abs(t.entry_position) >= 0.5).length;
                      })()}
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
