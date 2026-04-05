import { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getModels, getModelDetail, getModelMetrics } from '../services/api';
import type { ModelData, ModelsResponse } from '../types';
import { createChart, IChartApi, ITimeScaleApi } from 'lightweight-charts';
import { ChevronLeft, ChevronRight, Calendar, Settings, TrendingUp, AlertTriangle, DollarSign, BarChart2 } from 'lucide-react';
import { clsx } from 'clsx';
import { calculateSummaryFromTrades, buildNetEquityCurve, INITIAL_CAPITAL } from '../utils/transactionCosts';
import LoadingScreen from '../components/LoadingScreen';

export default function DetailPage() {
  const { modelName } = useParams();
  const navigate = useNavigate();
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [filteredMetrics, setFilteredMetrics] = useState<any>(null);

  const equityChartRef = useRef<HTMLDivElement>(null);
  const positionChartRef = useRef<HTMLDivElement>(null);
  const drawdownChartRef = useRef<HTMLDivElement>(null);
  const equityChartInstance = useRef<IChartApi | null>(null);
  const positionChartInstance = useRef<IChartApi | null>(null);
  const drawdownChartInstance = useRef<IChartApi | null>(null);
  const isSyncing = useRef(false);

  useEffect(() => {
    loadModelsData();
  }, []);

  useEffect(() => {
    if (modelName) {
      loadModelData(modelName);
    }
  }, [modelName]);

  useEffect(() => {
    if (modelData) {
      initCharts();
    }
    return () => {
      if (equityChartInstance.current) {
        equityChartInstance.current.remove();
        equityChartInstance.current = null;
      }
      if (positionChartInstance.current) {
        positionChartInstance.current.remove();
        positionChartInstance.current = null;
      }
      if (drawdownChartInstance.current) {
        drawdownChartInstance.current.remove();
        drawdownChartInstance.current = null;
      }
    };
  }, [modelData]);

  async function loadModelsData() {
    try {
      const data = await getModels();
      setModelsData(data);
      if (!modelName && data.models.length > 0) {
        navigate(`/detail/${data.models[0].model}`);
      }
    } catch (err) {
      console.error(err);
    }
  }

  async function loadModelData(name: string) {
    try {
      setLoading(true);
      const data = await getModelDetail(name);
      setModelData(data);
      if (data.dates.length > 0) {
        setDateRange({
          start: data.dates[0],
          end: data.dates[data.dates.length - 1]
        });
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function applyDateFilter() {
    if (!modelName || !dateRange.start || !dateRange.end) return;
    try {
      const data = await getModelMetrics(modelName, dateRange.start, dateRange.end);
      setFilteredMetrics(data);
    } catch (err) {
      console.error(err);
    }
  }

  function syncTimeScales(sourceTimeScale: ITimeScaleApi<any>, sourceChart: IChartApi) {
    if (isSyncing.current) return;
    const visibleRange = sourceTimeScale.getVisibleLogicalRange();
    if (!visibleRange) return;
    isSyncing.current = true;
    const charts = [equityChartInstance.current, positionChartInstance.current, drawdownChartInstance.current];
    charts.forEach(chart => {
      if (chart && chart !== sourceChart) {
        chart.timeScale().setVisibleLogicalRange(visibleRange);
      }
    });
    setTimeout(() => {
      isSyncing.current = false;
    }, 10);
  }

  function initCharts() {
    if (!modelData) return;

    const chartOptions = {
      layout: {
        background: { color: '#111111' },
        textColor: '#737373',
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: '#1a1a1a' },
        horzLines: { color: '#1a1a1a' },
      },
      rightPriceScale: {
        borderColor: '#222222',
      },
      timeScale: {
        borderColor: '#222222',
        timeVisible: true,
      },
      crosshair: {
        mode: 1,
      },
    };

    // Build net equity curve with transaction costs applied at correct trade dates
    const netEquity = buildNetEquityCurve(modelData.dates, modelData.equity_curve, modelData.trades, INITIAL_CAPITAL);
    const hasValidNetEquity = netEquity && netEquity.length > 0;

    // Equity Chart
    if (equityChartRef.current) {
      if (equityChartInstance.current) {
        equityChartInstance.current.remove();
      }

      const chart = createChart(equityChartRef.current, {
        ...chartOptions,
        width: equityChartRef.current.clientWidth,
        height: 350,
        localization: {
          priceFormatter: (price: number) => '$' + price.toLocaleString('en-US', { maximumFractionDigits: 0 }),
        },
      });

      equityChartInstance.current = chart;

      const strategySeries = chart.addLineSeries({
        color: '#c41e3a',
        lineWidth: 2,
        title: 'Strategy',
      });

      // Use net equity curve with transaction costs applied at correct dates
      const strategyData = modelData.dates.map((date, i) => ({
        time: date,
        value: hasValidNetEquity ? (netEquity[i] || 0) : modelData.equity_curve[i] * INITIAL_CAPITAL,
      }));
      strategySeries.setData(strategyData as any);

      const marketSeries = chart.addLineSeries({
        color: '#525252',
        lineWidth: 1,
        lineStyle: 2,
        title: 'Buy & Hold',
      });

      // Convert market equity to actual dollar values
      const marketData = modelData.dates.map((date, i) => ({
        time: date,
        value: modelData.market_equity[i] * INITIAL_CAPITAL,
      }));
      marketSeries.setData(marketData as any);

      chart.timeScale().fitContent();
      chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        syncTimeScales(chart.timeScale(), chart);
      });
    }

    // Position Chart
    if (positionChartRef.current) {
      if (positionChartInstance.current) {
        positionChartInstance.current.remove();
      }

      const chart = createChart(positionChartRef.current, {
        ...chartOptions,
        width: positionChartRef.current.clientWidth,
        height: 180,
      });

      positionChartInstance.current = chart;

      const positionSeries = chart.addHistogramSeries({
        color: '#c41e3a',
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => `${price.toFixed(0)}x`,
        },
      });

      const positionData = modelData.dates.map((date, i) => ({
        time: date,
        value: modelData.positions[i],
        color: modelData.positions[i] >= 1 ? '#00c853' :
               modelData.positions[i] <= -1 ? '#c41e3a' : '#525252',
      }));
      positionSeries.setData(positionData as any);

      chart.applyOptions({
        rightPriceScale: {
          scaleMargins: { top: 0.1, bottom: 0.1 },
        },
      });

      chart.timeScale().fitContent();
      chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        syncTimeScales(chart.timeScale(), chart);
      });
    }

    // Drawdown Chart
    if (drawdownChartRef.current) {
      if (drawdownChartInstance.current) {
        drawdownChartInstance.current.remove();
      }

      const chart = createChart(drawdownChartRef.current, {
        ...chartOptions,
        width: drawdownChartRef.current.clientWidth,
        height: 150,
      });

      drawdownChartInstance.current = chart;

      const drawdownSeries = chart.addAreaSeries({
        lineColor: '#c41e3a',
        topColor: 'rgba(196, 30, 58, 0.4)',
        bottomColor: 'rgba(196, 30, 58, 0.0)',
        lineWidth: 1,
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => `${(price * 100).toFixed(1)}%`,
        },
      });

      const drawdownData = modelData.dates.map((date, i) => ({
        time: date,
        value: modelData.drawdown[i],
      }));
      drawdownSeries.setData(drawdownData as any);

      chart.timeScale().fitContent();
      chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        syncTimeScales(chart.timeScale(), chart);
      });
    }

    const handleResize = () => {
      if (equityChartRef.current && equityChartInstance.current) {
        equityChartInstance.current.applyOptions({ width: equityChartRef.current.clientWidth });
      }
      if (positionChartRef.current && positionChartInstance.current) {
        positionChartInstance.current.applyOptions({ width: positionChartRef.current.clientWidth });
      }
      if (drawdownChartRef.current && drawdownChartInstance.current) {
        drawdownChartInstance.current.applyOptions({ width: drawdownChartRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }

  function navigateToModel(direction: 'prev' | 'next') {
    if (!modelsData || !modelName) return;
    const currentIdx = modelsData.models.findIndex(m => m.model === modelName);
    const newIdx = direction === 'prev'
      ? (currentIdx - 1 + modelsData.models.length) % modelsData.models.length
      : (currentIdx + 1) % modelsData.models.length;
    navigate(`/detail/${modelsData.models[newIdx].model}`);
  }

  if (loading) {
    return <LoadingScreen />;
  }

  if (!modelData) {
    return <div className="text-center py-8 text-[#c41e3a]">Model not found</div>;
  }

  const baseMetrics = modelData.metrics;
  const filteredM = filteredMetrics?.metrics;
  const metrics = filteredM ? {
    ...baseMetrics,
    ...filteredM,
    isFiltered: true,
  } : {
    ...baseMetrics,
    isFiltered: false,
  };
  // Backend LONG-ONLY uses 'params' field, not 'optimal_params'
  const params = (modelData as any).params ?? modelData.optimal_params ?? { q_3x: 10, q_1x: 30 };

  // Calculate summary from trades (consistent with Trades page)
  const tradeSummary = modelData.trades && modelData.trades.length > 0
    ? calculateSummaryFromTrades(modelData.trades, INITIAL_CAPITAL)
    : { finalCapital: INITIAL_CAPITAL, totalTxCosts: 0, netReturn: 0, grossReturn: 0 };

  return (
    <div className="space-y-6">
      {/* Header with Navigation */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigateToModel('prev')}
            className="p-2 rounded hover:bg-[#1a1a1a] text-[#737373] hover:text-white"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <div>
            <h2 className="text-xl font-semibold text-white">{modelName}</h2>
            <span className="text-sm text-[#525252]">{modelData.category}</span>
          </div>
          <button
            onClick={() => navigateToModel('next')}
            className="p-2 rounded hover:bg-[#1a1a1a] text-[#737373] hover:text-white"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
        <div className="flex items-center gap-4">
          <select
            value={modelName || ''}
            onChange={(e) => navigate(`/detail/${e.target.value}`)}
            className="bg-[#1a1a1a] border border-[#222222] rounded px-3 py-2 text-sm text-white"
          >
            {modelsData?.models.map(m => (
              <option key={m.model} value={m.model}>{m.model}</option>
            ))}
          </select>
          <button
            onClick={() => navigate('/overview')}
            className="text-sm text-[#737373] hover:text-white"
          >
            Back to Overview
          </button>
        </div>
      </div>

      {/* Optimal Parameters Card - LONG-ONLY */}
      <div className="card bg-gradient-to-r from-[#111111] to-[#1a1a1a] border-l-4 border-[#c41e3a]">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-[#c41e3a]" />
          <h3 className="font-medium text-white">Optimal Position Parameters (Long-Only)</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-[#0a0a0a] rounded">
            <div className="text-xs text-[#525252] mb-1">3x Long (UPRO)</div>
            <div className="text-2xl font-bold text-[#00c853]">Top {params.q_3x ?? 10}%</div>
          </div>
          <div className="text-center p-3 bg-[#0a0a0a] rounded">
            <div className="text-xs text-[#525252] mb-1">1x Long (SPY)</div>
            <div className="text-2xl font-bold text-[#22c55e]">Top {params.q_1x ?? 30}%</div>
          </div>
          <div className="text-center p-3 bg-[#0a0a0a] rounded">
            <div className="text-xs text-[#525252] mb-1">Cash</div>
            <div className="text-2xl font-bold text-[#525252]">Remaining</div>
          </div>
        </div>
      </div>

      {/* Date Range Filter */}
      <div className="card">
        <div className="flex items-center gap-4 flex-wrap">
          <Calendar className="w-5 h-5 text-[#737373]" />
          <div className="flex items-center gap-2">
            <label className="text-sm text-[#737373]">From:</label>
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
              className="bg-[#1a1a1a] border border-[#222222] rounded px-3 py-1.5 text-sm text-white"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-[#737373]">To:</label>
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
              className="bg-[#1a1a1a] border border-[#222222] rounded px-3 py-1.5 text-sm text-white"
            />
          </div>
          <button
            onClick={applyDateFilter}
            className="px-4 py-1.5 bg-[#c41e3a] rounded text-sm font-medium hover:bg-[#9a1830] text-white"
          >
            Apply
          </button>
          <button
            onClick={() => {
              setDateRange({
                start: modelData.dates[0],
                end: modelData.dates[modelData.dates.length - 1]
              });
              setFilteredMetrics(null);
            }}
            className="px-4 py-1.5 bg-[#1a1a1a] border border-[#222222] rounded text-sm font-medium hover:bg-[#222222] text-white"
          >
            Reset
          </button>

          <div className="flex gap-2 ml-4">
            {['1M', '3M', '6M', '1Y', 'ALL'].map(period => (
              <button
                key={period}
                className="px-3 py-1 text-xs bg-[#1a1a1a] rounded hover:bg-[#222222] text-[#a3a3a3]"
                onClick={() => {
                  const end = modelData.dates[modelData.dates.length - 1];
                  let start = end;
                  const endDate = new Date(end);
                  if (period === '1M') {
                    endDate.setMonth(endDate.getMonth() - 1);
                    start = endDate.toISOString().split('T')[0];
                  } else if (period === '3M') {
                    endDate.setMonth(endDate.getMonth() - 3);
                    start = endDate.toISOString().split('T')[0];
                  } else if (period === '6M') {
                    endDate.setMonth(endDate.getMonth() - 6);
                    start = endDate.toISOString().split('T')[0];
                  } else if (period === '1Y') {
                    endDate.setFullYear(endDate.getFullYear() - 1);
                    start = endDate.toISOString().split('T')[0];
                  } else {
                    start = modelData.dates[0];
                  }
                  setDateRange({ start, end });
                }}
              >
                {period}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Filtered Period Indicator */}
      {metrics.isFiltered && (
        <div className="bg-[#c41e3a]/20 border border-[#c41e3a]/50 rounded-lg p-3 text-sm">
          <span className="text-[#c41e3a] font-medium">Filtered Period:</span>
          <span className="ml-2 text-white">{dateRange.start} to {dateRange.end}</span>
          <span className="ml-4 text-[#737373]">(Bootstrap CI and position distribution show full period values)</span>
        </div>
      )}

      {/* Primary Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div className="metric-card">
          <div className="flex items-center gap-2 text-[#737373] mb-1">
            <DollarSign className="w-4 h-4" />
            <span className="metric-label">Net Return</span>
          </div>
          <div className={clsx(
            "metric-value",
            tradeSummary.netReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
          )}>
            {(tradeSummary.netReturn * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-[#f59e0b] mt-1">
            Tx: -${(tradeSummary.totalTxCosts ?? 0).toFixed(0)}
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center gap-2 text-[#737373] mb-1">
            <TrendingUp className="w-4 h-4" />
            <span className="metric-label">Sharpe Ratio</span>
          </div>
          <div className="metric-value text-white">{(metrics.sharpe ?? 0).toFixed(2)}</div>
          {metrics.sharpe_ci_lower !== undefined && metrics.sharpe_ci_upper !== undefined && (
            <div className="text-xs text-[#525252] mt-1">
              [{metrics.sharpe_ci_lower.toFixed(2)}, {metrics.sharpe_ci_upper.toFixed(2)}]
            </div>
          )}
        </div>

        {metrics.prob_sharpe_positive !== undefined && (
          <div className="metric-card">
            <div className="metric-label">P(Sharpe &gt; 0)</div>
            <div className={clsx(
              "metric-value",
              metrics.prob_sharpe_positive >= 0.99 ? "text-[#00c853]" :
              metrics.prob_sharpe_positive >= 0.95 ? "text-white" : "text-[#737373]"
            )}>
              {(metrics.prob_sharpe_positive * 100).toFixed(1)}%
            </div>
          </div>
        )}

        <div className="metric-card">
          <div className="metric-label">Sortino</div>
          <div className="metric-value text-white">{(metrics.sortino ?? 0).toFixed(2)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Calmar</div>
          <div className="metric-value text-white">{(metrics.calmar ?? 0).toFixed(2)}</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center gap-2 text-[#737373] mb-1">
            <AlertTriangle className="w-4 h-4" />
            <span className="metric-label">Max Drawdown</span>
          </div>
          <div className="metric-value text-[#c41e3a]">
            {((metrics.max_drawdown ?? 0) * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        {metrics.win_rate !== undefined && (
          <div className="metric-card">
            <div className="metric-label">Win Rate</div>
            <div className="metric-value text-[#a3a3a3]">
              {(metrics.win_rate * 100).toFixed(1)}%
            </div>
          </div>
        )}

        {metrics.dir_accuracy !== undefined && (
          <div className="metric-card">
            <div className="metric-label">Dir. Accuracy</div>
            <div className="metric-value text-[#a3a3a3]">
              {(metrics.dir_accuracy * 100).toFixed(1)}%
            </div>
          </div>
        )}

        <div className="metric-card">
          <div className="metric-label">Active Trades</div>
          <div className="metric-value text-white">
            {modelData?.trades?.length
              ? modelData.trades.filter(t => Math.abs(t.entry_position) >= 0.5).length
              : metrics.n_trades}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Trans. Costs</div>
          <div className="metric-value text-[#f59e0b]">
            -${(tradeSummary.totalTxCosts ?? 0).toFixed(0)}
          </div>
        </div>

      </div>

      {/* Position Distribution - LONG-ONLY */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <BarChart2 className="w-5 h-5 text-[#737373]" />
          <h3 className="font-medium text-white">Position Distribution (Long-Only)</h3>
        </div>
        <div className="flex flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[#00c853]" />
            <span className="text-sm text-[#a3a3a3]">3x UPRO: {(metrics.pct_3x ?? 0).toFixed(1)}%</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[#22d3ee]" />
            <span className="text-sm text-[#a3a3a3]">1x SPY: {(metrics.pct_1x ?? 0).toFixed(1)}%</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[#525252]" />
            <span className="text-sm text-[#a3a3a3]">Cash: {(metrics.pct_cash ?? 0).toFixed(1)}%</span>
          </div>
        </div>
        <div className="mt-4 h-8 rounded overflow-hidden flex">
          <div style={{ width: `${metrics.pct_3x ?? 0}%` }} className="bg-[#00c853]" title="3x UPRO" />
          <div style={{ width: `${metrics.pct_1x ?? 0}%` }} className="bg-[#22d3ee]" title="1x SPY" />
          <div style={{ width: `${metrics.pct_cash ?? 0}%` }} className="bg-[#525252]" title="Cash" />
        </div>
      </div>

      {/* Synchronized Charts Section */}
      <div className="card">
        <h3 className="text-lg font-medium mb-2 text-white">Synchronized Charts</h3>
        <p className="text-sm text-[#525252] mb-4">
          Zoom or scroll on any chart to synchronize all three. Drag to pan, scroll to zoom.
        </p>

        {/* Equity Chart */}
        <div className="mb-2">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">Equity Curve: Strategy vs Buy & Hold</span>
            <div className="flex gap-4 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-[#c41e3a]" />
                <span className="text-[#a3a3a3]">Strategy</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-0.5 bg-[#525252]" style={{ borderStyle: 'dashed' }} />
                <span className="text-[#525252]">Buy & Hold</span>
              </div>
            </div>
          </div>
          <div ref={equityChartRef} className="w-full" />
        </div>

        {/* Position Chart - LONG-ONLY */}
        <div className="mb-2">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">Position Size (Long-Only: 0, +1, +3)</span>
            <div className="flex gap-4 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-[#00c853]" />
                <span className="text-[#a3a3a3]">Long (+1x, +3x)</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-[#525252]" />
                <span className="text-[#a3a3a3]">Cash (0)</span>
              </div>
            </div>
          </div>
          <div ref={positionChartRef} className="w-full" />
        </div>

        {/* Drawdown Chart */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">Drawdown (Underwater)</span>
            <div className="flex gap-4 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-[#c41e3a]" />
                <span className="text-[#a3a3a3]">Drawdown %</span>
              </div>
            </div>
          </div>
          <div ref={drawdownChartRef} className="w-full" />
        </div>
      </div>
    </div>
  );
}
