import { useEffect, useState, useRef, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getModels, getModelTrades, getModelDetail, getModelDailyLog, DailyLogEntry } from '../services/api';
import type { ModelsResponse, Trade, ModelData } from '../types';
import { createChart, IChartApi } from 'lightweight-charts';
import { Download, CheckCircle, XCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { INITIAL_CAPITAL, calculateSummaryFromTrades, formatCurrency } from '../utils/transactionCosts';
import LoadingScreen from '../components/LoadingScreen';
import { useTableSort } from '../hooks/useTableSort';
import { useTableFilters } from '../hooks/useTableFilters';
import SortableHeader from '../components/SortableHeader';
import DailyLogFilterBar from '../components/DailyLogFilterBar';

interface HistogramBin {
  min: number;
  max: number;
  count: number;
  midpoint: number;
}

export default function TradesPage() {
  const { modelName } = useParams();
  const navigate = useNavigate();
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [, setModelData] = useState<ModelData | null>(null);
  const [dailyLog, setDailyLog] = useState<DailyLogEntry[]>([]);
  const [warmupUsed, setWarmupUsed] = useState<boolean>(false);
  const [loading, setLoading] = useState(true);
  const [loadingDailyLog, setLoadingDailyLog] = useState(false);
  const [hoveredBin, setHoveredBin] = useState<number | null>(null);

  const pnlChartRef = useRef<HTMLDivElement>(null);
  const pnlChartInstance = useRef<IChartApi | null>(null);
  const [hoveredTrade, setHoveredTrade] = useState<{date: string; value: number; entry: string; duration: number; position: number} | null>(null);

  useEffect(() => {
    loadModelsData();
  }, []);

  useEffect(() => {
    if (modelName) {
      loadTrades(modelName);
      loadModelData(modelName);
      loadDailyLog(modelName);
    }
  }, [modelName]);

  // Filter out cash positions - only trades with actual market exposure
  // A trade with abs(entry_position) < 0.5 means the model was in cash (position = 0)
  // P&L = position * market_return (already included in total_return)
  const activeTrades = trades.filter(trade => Math.abs(trade.entry_position) >= 0.5);

  useEffect(() => {
    if (activeTrades.length > 0 && pnlChartRef.current) {
      initPnLChart();
    }
    return () => {
      if (pnlChartInstance.current) {
        pnlChartInstance.current.remove();
        pnlChartInstance.current = null;
      }
    };
  }, [activeTrades]);

  function initPnLChart() {
    if (!pnlChartRef.current || activeTrades.length === 0) return;

    if (pnlChartInstance.current) {
      pnlChartInstance.current.remove();
      pnlChartInstance.current = null;
    }

    const chart = createChart(pnlChartRef.current, {
      width: pnlChartRef.current.clientWidth,
      height: 300,
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
        timeVisible: false,
      },
      crosshair: {
        mode: 1,
        vertLine: {
          labelVisible: false, // Disable X-axis crosshair label to avoid confusion
        },
      },
    });

    pnlChartInstance.current = chart;

    const pnlSeries = chart.addHistogramSeries({
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => `${price >= 0 ? '+' : ''}${price.toFixed(2)}%`,
      },
    });

    // Only show trades with actual market exposure (position != 0)
    // P&L = position * market_return (already calculated in total_return)
    const pnlData = activeTrades.map(trade => ({
      time: trade.exit_date,
      value: trade.total_return * 100,
      color: trade.total_return >= 0 ? '#00c853' : '#c41e3a',
    }));

    // Create a map for quick lookup of trade data by date (with full trade info)
    const tradeDataMap = new Map<string, {value: number; entry: string; duration: number; position: number}>();
    activeTrades.forEach(t => {
      tradeDataMap.set(t.exit_date, {
        value: t.total_return * 100,
        entry: t.entry_date,
        duration: t.duration,
        position: t.entry_position
      });
    });

    pnlData.sort((a, b) => a.time.localeCompare(b.time));
    pnlSeries.setData(pnlData as any);

    const zeroLine = chart.addLineSeries({
      color: '#525252',
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    if (pnlData.length > 0) {
      zeroLine.setData([
        { time: pnlData[0].time, value: 0 },
        { time: pnlData[pnlData.length - 1].time, value: 0 },
      ] as any);
    }

    // Subscribe to crosshair move to show correct trade date
    chart.subscribeCrosshairMove((param) => {
      if (param.time) {
        const timeStr = param.time as string;
        const tradeInfo = tradeDataMap.get(timeStr);
        if (tradeInfo !== undefined) {
          setHoveredTrade({
            date: timeStr,
            value: tradeInfo.value,
            entry: tradeInfo.entry,
            duration: tradeInfo.duration,
            position: tradeInfo.position
          });
        } else {
          setHoveredTrade(null);
        }
      } else {
        setHoveredTrade(null);
      }
    });

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (pnlChartRef.current && pnlChartInstance.current) {
        pnlChartInstance.current.applyOptions({ width: pnlChartRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }

  async function loadModelsData() {
    try {
      const data = await getModels();
      setModelsData(data);
      if (!modelName && data.models.length > 0) {
        navigate(`/trades/${data.models[0].model}`);
      }
    } catch (err) {
      console.error(err);
    }
  }

  async function loadTrades(name: string) {
    try {
      setLoading(true);
      const data = await getModelTrades(name);
      setTrades(data.trades || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function loadModelData(name: string) {
    try {
      const data = await getModelDetail(name);
      setModelData(data);
    } catch (err) {
      console.error(err);
    }
  }

  async function loadDailyLog(name: string) {
    try {
      setLoadingDailyLog(true);
      const response = await getModelDailyLog(name);
      setDailyLog(response.daily_log || []);
      setWarmupUsed(response.warmup_used);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingDailyLog(false);
    }
  }

  function formatPercent(value: number, decimals: number = 2): string {
    return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(decimals)}%`;
  }

  function formatPercentile(value: number): string {
    return `${value.toFixed(1)}%`;
  }

  function formatPrediction(value: number): string {
    const pct = value * 100;
    return `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
  }

  function getPercentileColor(percentile: number): string {
    if (percentile >= 90) return 'text-[#00c853]';
    if (percentile >= 70) return 'text-[#4caf50]';
    if (percentile <= 10) return 'text-[#c41e3a]';
    if (percentile <= 30) return 'text-[#ef5350]';
    return 'text-[#737373]';
  }

  function getPredictionColor(prediction: number): string {
    const pct = prediction * 100;
    if (pct >= 0.5) return 'text-[#00c853]';
    if (pct >= 0.2) return 'text-[#4caf50]';
    if (pct <= -0.5) return 'text-[#c41e3a]';
    if (pct <= -0.2) return 'text-[#ef5350]';
    return 'text-[#a3a3a3]';
  }

  // Filter & Sort hooks for daily log
  const {
    filters, filteredData: filteredDailyLog, isFiltered,
    togglePosition, toggleRegime, setDateStart, setDateEnd,
    setDirection, toggleChangesOnly, resetFilters,
  } = useTableFilters(dailyLog);

  const dailyLogAccessor = useMemo(() => ({
    day_num: (d: DailyLogEntry) => d.day_num,
    date: (d: DailyLogEntry) => d.date,
    prediction: (d: DailyLogEntry) => d.prediction,
    percentile: (d: DailyLogEntry) => d.percentile,
    position_final: (d: DailyLogEntry) => d.position_final,
    market_return: (d: DailyLogEntry) => d.market_return,
    strategy_return: (d: DailyLogEntry) => d.strategy_return,
    trading_cost: (d: DailyLogEntry) => d.trading_cost,
    equity: (d: DailyLogEntry) => d.equity,
    cumReturn: (d: DailyLogEntry) => d.equity - 1,
    drawdown: (d: DailyLogEntry) => d.drawdown,
    regime: (d: DailyLogEntry) => d.regime || 'unknown',
  }), []);

  const {
    sortedData: sortedDailyLog, requestSort: requestDailySort,
    getSortDirection: getDailySortDir,
  } = useTableSort(filteredDailyLog, 'day_num', 'asc', dailyLogAccessor);

  function downloadDailyLogCSV() {
    const exportData = isFiltered ? sortedDailyLog : dailyLog;
    if (exportData.length === 0 || !modelName) return;
    const headers = ['Day','Date','Prediction_BP','Percentile','Position_Base','Position_Final','Market_Return','Strategy_Return','Equity','HWM','Drawdown','Trading_Cost','Regime','Position_Changed','Direction_Correct'];
    const rows = exportData.map((day) => [
      day.day_num, day.date, (day.prediction * 10000).toFixed(2), day.percentile.toFixed(1),
      day.position_base, day.position_final, (day.market_return * 100).toFixed(4),
      (day.strategy_return * 100).toFixed(4), (day.equity * INITIAL_CAPITAL).toFixed(2),
      (day.high_water_mark * INITIAL_CAPITAL).toFixed(2), (day.drawdown * 100).toFixed(2),
      (day.trading_cost * 100).toFixed(4), day.regime, day.position_changed ? 'YES' : 'NO',
      day.direction_correct ? 'YES' : 'NO'
    ]);
    const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.setAttribute('href', URL.createObjectURL(blob));
    link.setAttribute('download', `${modelName}_daily_log${isFiltered ? '_filtered' : ''}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // Daily log statistics
  const dailyStats = useMemo(() => {
    if (dailyLog.length === 0) return null;
    const totalDays = dailyLog.length;
    const positionChanges = dailyLog.filter(d => d.position_changed).length;
    const correctDirections = dailyLog.filter(d => {
      const hasPosition = d.position_final > 0;
      const marketUp = d.market_return > 0;
      return (hasPosition && marketUp) || (!hasPosition && !marketUp);
    }).length;
    const days3x = dailyLog.filter(d => d.position_final === 3).length;
    const days1x = dailyLog.filter(d => d.position_final === 1).length;
    const cashDays = dailyLog.filter(d => d.position_final === 0).length;
    const winningDays = dailyLog.filter(d => d.strategy_return > 0).length;
    const finalEquity = dailyLog[dailyLog.length - 1].equity * INITIAL_CAPITAL;
    const totalReturn = (dailyLog[dailyLog.length - 1].equity - 1) * 100;
    const maxDrawdown = Math.min(...dailyLog.map(d => d.drawdown)) * 100;
    return {
      totalDays, positionChanges, correctDirections,
      dirAccuracy: totalDays > 0 ? (correctDirections / totalDays) * 100 : 0,
      days3x, days1x, cashDays, winningDays,
      winRate: (winningDays / totalDays) * 100,
      finalEquity, totalReturn, maxDrawdown
    };
  }, [dailyLog]);

  function createHistogramBins(): HistogramBin[] {
    if (activeTrades.length === 0) return [];

    const returns = activeTrades.map(t => t.total_return * 100);
    const minReturn = Math.min(...returns);
    const maxReturn = Math.max(...returns);

    const binCount = 20;
    const range = maxReturn - minReturn;
    const binWidth = range > 0 ? (range * 1.0001) / binCount : 1;

    const bins: HistogramBin[] = [];
    for (let i = 0; i < binCount; i++) {
      bins.push({
        min: minReturn + i * binWidth,
        max: minReturn + (i + 1) * binWidth,
        midpoint: minReturn + (i + 0.5) * binWidth,
        count: 0
      });
    }

    returns.forEach(r => {
      let binIdx = Math.floor((r - minReturn) / binWidth);
      binIdx = Math.max(0, Math.min(binIdx, binCount - 1));
      bins[binIdx].count++;
    });

    return bins;
  }

  // Statistics based on active trades only (excludes cash positions)
  const winningTrades = activeTrades.filter(t => t.total_return > 0);
  const losingTrades = activeTrades.filter(t => t.total_return < 0);
  const avgWin = winningTrades.length > 0
    ? winningTrades.reduce((sum, t) => sum + t.total_return, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length > 0
    ? losingTrades.reduce((sum, t) => sum + t.total_return, 0) / losingTrades.length
    : 0;
  const profitFactor = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0;
  const avgDuration = activeTrades.length > 0
    ? activeTrades.reduce((sum, t) => sum + t.duration, 0) / activeTrades.length
    : 0;

  const histogramBins = createHistogramBins();
  const maxBinCount = Math.max(...histogramBins.map(b => b.count), 1);
  const zeroIndex = histogramBins.findIndex(b => b.min <= 0 && b.max >= 0);

  if (loading) {
    return <LoadingScreen />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Trade Analysis</h2>
          <span className="text-sm text-[#525252]">{modelName}</span>
        </div>

        <select
          value={modelName || ''}
          onChange={(e) => navigate(`/trades/${e.target.value}`)}
          className="bg-[#1a1a1a] border border-[#222222] rounded px-3 py-2 text-sm text-white"
        >
          {modelsData?.models.map(m => (
            <option key={m.model} value={m.model}>{m.model}</option>
          ))}
        </select>
      </div>

      {/* Cost Summary KPIs */}
      {trades.length > 0 && (() => {
        const summary = calculateSummaryFromTrades(trades, INITIAL_CAPITAL);
        const costImpactBps = (summary.grossReturn - summary.netReturn) * 10000;
        return (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="metric-card border-[#00c853]/20">
              <div className="metric-label">Net Capital</div>
              <div className={clsx(
                "metric-value",
                summary.finalCapital >= INITIAL_CAPITAL ? "text-[#00c853]" : "text-[#c41e3a]"
              )}>
                {formatCurrency(summary.finalCapital)}
              </div>
            </div>
            <div className="metric-card border-[#00c853]/20">
              <div className="metric-label">Net Return</div>
              <div className={clsx(
                "metric-value",
                summary.netReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
              )}>
                {summary.netReturn >= 0 ? '+' : ''}{(summary.netReturn * 100).toFixed(1)}%
              </div>
            </div>
            <div className="metric-card border-[#f59e0b]/20">
              <div className="metric-label">Total Costs</div>
              <div className="metric-value text-[#f59e0b]">
                {formatCurrency(summary.costBreakdown.total)}
              </div>
              <div className="text-xs text-[#525252] mt-1">
                Exp {formatCurrency(summary.costBreakdown.expense)} / Trad {formatCurrency(summary.costBreakdown.trading)} / Vol {formatCurrency(summary.costBreakdown.volDrag)}
              </div>
            </div>
            <div className="metric-card border-[#f59e0b]/20">
              <div className="metric-label">Cost Impact</div>
              <div className="metric-value text-[#f59e0b]">
                {costImpactBps.toFixed(0)} bps
              </div>
              <div className="text-xs text-[#525252] mt-1">
                {(summary.grossReturn * 100).toFixed(1)}% gross → {(summary.netReturn * 100).toFixed(1)}% net
              </div>
            </div>
          </div>
        );
      })()}

      {/* Statistics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
        <div className="metric-card">
          <div className="metric-label">Active Trades</div>
          <div className="metric-value text-white">{activeTrades.length}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Winning</div>
          <div className="metric-value text-[#00c853]">
            {winningTrades.length}
            <span className="text-sm text-[#525252] ml-1">
              ({activeTrades.length > 0 ? ((winningTrades.length / activeTrades.length) * 100).toFixed(1) : '0.0'}%)
            </span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Losing</div>
          <div className="metric-value text-[#c41e3a]">
            {losingTrades.length}
            <span className="text-sm text-[#525252] ml-1">
              ({activeTrades.length > 0 ? ((losingTrades.length / activeTrades.length) * 100).toFixed(1) : '0.0'}%)
            </span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Avg Win</div>
          <div className="metric-value text-[#00c853]">
            +{(avgWin * 100).toFixed(2)}%
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Avg Loss</div>
          <div className="metric-value text-[#c41e3a]">
            {(avgLoss * 100).toFixed(2)}%
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Profit Factor</div>
          <div className="metric-value text-white">{profitFactor.toFixed(2)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Avg Duration</div>
          <div className="metric-value text-[#a3a3a3]">{avgDuration.toFixed(1)}d</div>
        </div>
      </div>

      {/* P&L Timeline Chart */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-white">P&L Timeline ({activeTrades.length} trades)</h3>
          <div className="flex gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-[#00c853]" />
              <span className="text-[#737373]">Profit</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-[#c41e3a]" />
              <span className="text-[#737373]">Loss</span>
            </div>
          </div>
        </div>
        <p className="text-sm text-[#525252] mb-3">
          Each bar represents a trade's P&L (%) at its exit date. Only trades with market exposure (position ≠ 0). Scroll to zoom, drag to pan.
        </p>
        <div className="relative">
          {/* Custom Tooltip */}
          {hoveredTrade && (
            <div className="absolute top-2 left-2 bg-[#222222] border border-[#333] px-3 py-2 rounded shadow-lg z-10 text-sm">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[#737373] text-xs">Entry:</span>
                <span className="text-white font-mono text-xs">{hoveredTrade.entry}</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[#737373] text-xs">Exit:</span>
                <span className="text-white font-mono text-xs font-medium">{hoveredTrade.date}</span>
              </div>
              <div className="flex items-center gap-3 mt-2 pt-2 border-t border-[#333]">
                <div>
                  <span className="text-[#737373] text-xs">Pos: </span>
                  <span className={clsx(
                    "font-mono text-xs",
                    hoveredTrade.position > 0 ? "text-[#00c853]" : hoveredTrade.position < 0 ? "text-[#c41e3a]" : "text-[#737373]"
                  )}>
                    {hoveredTrade.position > 0 ? '+' : ''}{hoveredTrade.position.toFixed(0)}x
                  </span>
                </div>
                <div>
                  <span className="text-[#737373] text-xs">Dur: </span>
                  <span className="font-mono text-xs text-[#a3a3a3]">{hoveredTrade.duration}d</span>
                </div>
              </div>
              <div className={clsx(
                "font-mono font-bold text-lg mt-1 text-center",
                hoveredTrade.value >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
              )}>
                {hoveredTrade.value >= 0 ? '+' : ''}{hoveredTrade.value.toFixed(2)}%
              </div>
            </div>
          )}
          <div ref={pnlChartRef} className="w-full" />
        </div>
      </div>

      {/* P&L Distribution */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-white">P&L Distribution ({activeTrades.length} trades)</h3>
          <div className="flex gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-[#00c853]" />
              <span className="text-[#737373]">Profit ({winningTrades.length})</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-[#c41e3a]" />
              <span className="text-[#737373]">Loss ({losingTrades.length})</span>
            </div>
          </div>
        </div>

        {activeTrades.length > 0 ? (
          <div className="relative">
            {hoveredBin !== null && histogramBins[hoveredBin] && (
              <div className="absolute top-0 left-1/2 transform -translate-x-1/2 bg-[#222222] px-3 py-2 rounded shadow-lg z-10 text-sm whitespace-nowrap">
                <div className="font-medium text-white">
                  {histogramBins[hoveredBin].min.toFixed(1)}% to {histogramBins[hoveredBin].max.toFixed(1)}%
                </div>
                <div className="text-[#737373]">
                  {histogramBins[hoveredBin].count} trades ({((histogramBins[hoveredBin].count / activeTrades.length) * 100).toFixed(1)}%)
                </div>
              </div>
            )}

            <div className="flex">
              <div className="flex flex-col justify-between h-[180px] pr-2 text-xs text-[#525252] w-8">
                <span>{maxBinCount}</span>
                <span>{Math.round(maxBinCount / 2)}</span>
                <span>0</span>
              </div>

              <div className="flex-1">
                <div className="text-xs text-[#525252] mb-1 text-center">Number of Trades</div>

                <div className="h-[180px] flex items-end gap-0.5 border-l border-b border-[#222222] relative">
                  <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
                    <div className="border-b border-[#1a1a1a] border-dashed" />
                    <div className="border-b border-[#1a1a1a] border-dashed" />
                    <div />
                  </div>

                  {zeroIndex >= 0 && (
                    <div
                      className="absolute bottom-0 top-0 w-0.5 bg-[#525252] opacity-50"
                      style={{ left: `${((zeroIndex + 0.5) / histogramBins.length) * 100}%` }}
                    />
                  )}

                  {histogramBins.map((bin, idx) => {
                    const barHeight = maxBinCount > 0 ? (bin.count / maxBinCount) * 100 : 0;
                    const isProfit = bin.midpoint >= 0;
                    const isHovered = hoveredBin === idx;

                    return (
                      <div
                        key={idx}
                        className="flex-1 flex flex-col items-center justify-end h-full relative cursor-pointer"
                        onMouseEnter={() => setHoveredBin(idx)}
                        onMouseLeave={() => setHoveredBin(null)}
                      >
                        {isHovered && bin.count > 0 && (
                          <span className="absolute -top-5 text-xs font-medium text-white">
                            {bin.count}
                          </span>
                        )}

                        <div
                          className={clsx(
                            "w-full rounded-t transition-all duration-150",
                            isProfit ? "bg-[#00c853]" : "bg-[#c41e3a]",
                            isHovered && "opacity-80 scale-105"
                          )}
                          style={{
                            height: `${barHeight}%`,
                            minHeight: bin.count > 0 ? '2px' : '0'
                          }}
                        />
                      </div>
                    );
                  })}
                </div>

                <div className="flex justify-between mt-2 text-xs text-[#525252] px-1">
                  <span>{(histogramBins[0]?.min ?? 0).toFixed(1)}%</span>
                  <span className="font-medium text-white">0%</span>
                  <span>{(histogramBins[histogramBins.length - 1]?.max ?? 0).toFixed(1)}%</span>
                </div>

                <div className="text-xs text-[#525252] mt-1 text-center">Return per Trade (%)</div>
              </div>
            </div>

            <div className="flex justify-center gap-8 mt-4 pt-4 border-t border-[#1a1a1a]">
              <div className="text-center">
                <div className="text-xs text-[#525252]">Min Return</div>
                <div className={clsx(
                  "font-mono font-medium",
                  histogramBins[0]?.min >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                )}>
                  {(histogramBins[0]?.min ?? 0).toFixed(2)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-[#525252]">Median Return</div>
                <div className={clsx(
                  "font-mono font-medium",
                  activeTrades.length > 0 && [...activeTrades].sort((a, b) => a.total_return - b.total_return)[Math.floor(activeTrades.length / 2)]?.total_return >= 0
                    ? "text-[#00c853]"
                    : "text-[#c41e3a]"
                )}>
                  {activeTrades.length > 0
                    ? (([...activeTrades].sort((a, b) => a.total_return - b.total_return)[Math.floor(activeTrades.length / 2)]?.total_return ?? 0) * 100).toFixed(2)
                    : 0}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-[#525252]">Max Return</div>
                <div className={clsx(
                  "font-mono font-medium",
                  histogramBins[histogramBins.length - 1]?.max >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                )}>
                  {(histogramBins[histogramBins.length - 1]?.max ?? 0).toFixed(2)}%
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-[200px] flex items-center justify-center text-[#525252]">
            No trades to display
          </div>
        )}
      </div>

      {/* Position Distribution - LONG-ONLY */}
      {dailyStats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="metric-card border-[#00c853]/30">
            <div className="metric-label text-[#00c853]">3x UPRO Days</div>
            <div className="metric-value text-[#00c853]">
              {dailyStats.days3x}
              <span className="text-sm text-[#525252] ml-1">
                ({((dailyStats.days3x / dailyStats.totalDays) * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
          <div className="metric-card border-[#22d3ee]/30">
            <div className="metric-label text-[#22d3ee]">1x SPY Days</div>
            <div className="metric-value text-[#22d3ee]">
              {dailyStats.days1x}
              <span className="text-sm text-[#525252] ml-1">
                ({((dailyStats.days1x / dailyStats.totalDays) * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Cash Days</div>
            <div className="metric-value text-[#737373]">
              {dailyStats.cashDays}
              <span className="text-sm text-[#525252] ml-1">
                ({((dailyStats.cashDays / dailyStats.totalDays) * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
          <div className="metric-card border-[#6366f1]/30">
            <div className="metric-label text-[#6366f1]">Position Changes</div>
            <div className="metric-value text-[#6366f1]">
              {dailyStats.positionChanges}
              <span className="text-sm text-[#525252] ml-1">
                ({((dailyStats.positionChanges / dailyStats.totalDays) * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Daily Trading Log */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-medium text-white">Daily Trading Log</h3>
            <div className="flex items-center gap-4 text-sm text-[#737373] mt-1">
              <span>{dailyLog.length} trading days</span>
              {warmupUsed && (
                <span className="px-2 py-0.5 bg-[#6366f1]/20 text-[#6366f1] rounded text-xs">
                  Warmup: 63d training data
                </span>
              )}
            </div>
          </div>
          {dailyLog.length > 0 && (
            <button
              onClick={downloadDailyLogCSV}
              className="flex items-center gap-2 px-4 py-2 bg-[#00c853] text-black rounded font-medium hover:bg-[#00c853]/90 transition-colors text-sm"
            >
              <Download className="w-4 h-4" />
              CSV{isFiltered ? ` (${sortedDailyLog.length})` : ''}
            </button>
          )}
        </div>

        {loadingDailyLog ? (
          <div className="text-center py-8 text-[#737373]">Loading daily data...</div>
        ) : dailyLog.length > 0 ? (
          <>
            <DailyLogFilterBar
              filters={filters}
              totalRows={dailyLog.length}
              filteredRows={sortedDailyLog.length}
              isFiltered={isFiltered}
              onTogglePosition={togglePosition}
              onToggleRegime={toggleRegime}
              onSetDateStart={setDateStart}
              onSetDateEnd={setDateEnd}
              onSetDirection={setDirection}
              onToggleChangesOnly={toggleChangesOnly}
              onReset={resetFilters}
            />
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
              <table className="table-dark">
                <thead className="sticky top-0 bg-[#111111] z-10">
                  <tr>
                    <SortableHeader label="#" sortKey="day_num" activeDirection={getDailySortDir('day_num')} onSort={requestDailySort} className="text-center px-2" />
                    <SortableHeader label="Fecha" sortKey="date" activeDirection={getDailySortDir('date')} onSort={requestDailySort} className="text-center px-2" />
                    <SortableHeader label="Pred." sortKey="prediction" activeDirection={getDailySortDir('prediction')} onSort={requestDailySort} className="text-center px-2 bg-[#6366f1]/10" />
                    <SortableHeader label="%ile" sortKey="percentile" activeDirection={getDailySortDir('percentile')} onSort={requestDailySort} className="text-center px-2 bg-[#6366f1]/10" />
                    <SortableHeader label="Pos" sortKey="position_final" activeDirection={getDailySortDir('position_final')} onSort={requestDailySort} className="text-center px-2" />
                    <SortableHeader label="Mkt" sortKey="market_return" activeDirection={getDailySortDir('market_return')} onSort={requestDailySort} className="text-center px-2" />
                    <SortableHeader label="Strat" sortKey="strategy_return" activeDirection={getDailySortDir('strategy_return')} onSort={requestDailySort} className="text-center px-2" />
                    <SortableHeader label="Cost" sortKey="trading_cost" activeDirection={getDailySortDir('trading_cost')} onSort={requestDailySort} className="text-center px-2 bg-[#f59e0b]/5" />
                    <SortableHeader label="Capital" sortKey="equity" activeDirection={getDailySortDir('equity')} onSort={requestDailySort} className="text-center px-2 bg-[#00c853]/5" />
                    <SortableHeader label="Acum." sortKey="cumReturn" activeDirection={getDailySortDir('cumReturn')} onSort={requestDailySort} className="text-center px-2 bg-[#00c853]/5" />
                    <SortableHeader label="DD" sortKey="drawdown" activeDirection={getDailySortDir('drawdown')} onSort={requestDailySort} className="text-center px-2 bg-[#c41e3a]/5" />
                    <th className="text-center px-1">✓/✗</th>
                    <SortableHeader label="Rég." sortKey="regime" activeDirection={getDailySortDir('regime')} onSort={requestDailySort} className="text-center px-2" />
                  </tr>
                </thead>
                <tbody>
                  {sortedDailyLog.map((day, idx) => {
                    const equity = day.equity * INITIAL_CAPITAL;
                    const cumReturn = (day.equity - 1) * 100;
                    const position = day.position_final;

                    return (
                      <tr
                        key={idx}
                        className={clsx(day.position_changed && "bg-[#6366f1]/5")}
                      >
                        <td className="text-center text-[#737373] font-mono text-xs px-2">{day.day_num}</td>
                        <td className="text-center font-mono text-xs px-2">{day.date}</td>
                        <td className={clsx(
                          "text-center font-mono text-xs px-2 bg-[#6366f1]/5",
                          getPredictionColor(day.prediction)
                        )}>
                          {formatPrediction(day.prediction)}
                        </td>
                        <td className={clsx(
                          "text-center font-mono text-xs font-semibold px-2 bg-[#6366f1]/5",
                          getPercentileColor(day.percentile)
                        )}>
                          {formatPercentile(day.percentile)}
                        </td>
                        <td className={clsx(
                          "text-center font-mono text-xs font-bold px-2",
                          position === 3 && "text-[#00c853]",
                          position === 1 && "text-[#22d3ee]",
                          position === 0 && "text-[#737373]"
                        )}>
                          {position === 3 ? '3x' : position === 1 ? '1x' : '0x'}
                        </td>
                        <td className={clsx(
                          "text-center font-mono text-xs px-2",
                          day.market_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          {formatPercent(day.market_return, 2)}
                        </td>
                        <td className={clsx(
                          "text-center font-mono text-xs font-semibold px-2",
                          day.strategy_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          {formatPercent(day.strategy_return, 2)}
                        </td>
                        <td className="text-center font-mono text-xs px-2 bg-[#f59e0b]/5 text-[#f59e0b]">
                          {day.trading_cost > 0.000001 ? (day.trading_cost * 100).toFixed(2) + '%' : '-'}
                        </td>
                        <td className={clsx(
                          "text-center font-mono text-xs px-2 bg-[#00c853]/5",
                          equity >= INITIAL_CAPITAL ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          ${equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className={clsx(
                          "text-center font-mono text-xs px-2 bg-[#00c853]/5 font-semibold",
                          cumReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          {cumReturn >= 0 ? '+' : ''}{cumReturn.toFixed(2)}%
                        </td>
                        <td className={clsx(
                          "text-center font-mono text-xs px-2 bg-[#c41e3a]/5",
                          day.drawdown < -0.10 ? "text-[#c41e3a] font-semibold" :
                          day.drawdown < -0.05 ? "text-[#f59e0b]" : "text-[#737373]"
                        )}>
                          {day.drawdown !== 0 ? `${(day.drawdown * 100).toFixed(1)}%` : '-'}
                        </td>
                        {(() => {
                          const hasPosition = position > 0;
                          const marketUp = day.market_return > 0;
                          const isCorrect = (hasPosition && marketUp) || (!hasPosition && !marketUp);
                          return (
                            <td className="text-center px-1">
                              {isCorrect ? (
                                <CheckCircle className="w-3 h-3 text-[#00c853] inline" />
                              ) : (
                                <XCircle className="w-3 h-3 text-[#c41e3a] inline" />
                              )}
                            </td>
                          );
                        })()}
                        <td className={clsx(
                          "text-center font-mono text-xs px-2",
                          day.regime === 'bull' && "text-[#00c853]",
                          day.regime === 'bear' && "text-[#c41e3a]",
                          day.regime === 'high_vol' && "text-[#f59e0b]",
                          day.regime === 'sideways' && "text-[#737373]",
                          day.regime === 'unknown' && "text-[#525252]"
                        )}>
                          {(!day.regime || day.regime === 'unknown') ? '-' : day.regime}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Summary */}
            <div className="mt-4 pt-4 border-t border-[#222222]">
              <div className="text-sm text-[#737373]">
                Showing {isFiltered ? `${sortedDailyLog.length} of ` : ''}{dailyLog.length} trading days.
                <span className="text-[#6366f1] ml-2">
                  {(isFiltered ? sortedDailyLog : dailyLog).filter(d => d.position_changed).length} position changes.
                </span>
                <span className="text-[#f59e0b] ml-2">
                  {(isFiltered ? sortedDailyLog : dailyLog).filter(d => d.position_base !== d.position_final).length} days with risk management reduction.
                </span>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-8 text-[#737373]">Select a model to view daily data</div>
        )}
      </div>

      {/* Column Definitions */}
      <div className="card">
        <h3 className="text-lg font-medium mb-3">Definición de Columnas</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[#333333]">
                <th className="text-left py-2 px-2 w-20">Columna</th>
                <th className="text-left py-2 px-2">Descripción</th>
              </tr>
            </thead>
            <tbody className="text-[#a3a3a3]">
              <tr className="border-b border-[#222222]">
                <td className="py-1 px-2 font-mono text-white">#</td>
                <td className="py-1 px-2">Número de día de trading (1 = primer día del período out-of-sample)</td>
              </tr>
              <tr className="border-b border-[#222222]">
                <td className="py-1 px-2 font-mono text-white">Fecha</td>
                <td className="py-1 px-2">Fecha del día de trading (YYYY-MM-DD)</td>
              </tr>
              <tr className="border-b border-[#222222] bg-[#6366f1]/5">
                <td className="py-1 px-2 font-mono text-[#6366f1]">Pred.</td>
                <td className="py-1 px-2">
                  <strong className="text-white">Predicción del modelo</strong>: Retorno esperado para el día siguiente (%).
                </td>
              </tr>
              <tr className="border-b border-[#222222] bg-[#6366f1]/5">
                <td className="py-1 px-2 font-mono text-[#6366f1]">%ile</td>
                <td className="py-1 px-2">
                  <strong className="text-white">Percentil (63 días)</strong>: Ranking vs últimas 63 predicciones. 92% = más optimista que 92% de predicciones recientes.
                </td>
              </tr>
              <tr className="border-b border-[#222222]">
                <td className="py-1 px-2 font-mono text-white">Pos</td>
                <td className="py-1 px-2">
                  <strong className="text-white">Posición</strong>:
                  <span className="text-[#00c853]"> 3x</span>=UPRO,
                  <span className="text-[#22d3ee]"> 1x</span>=SPY,
                  <span className="text-[#737373]"> 0x</span>=Cash
                </td>
              </tr>
              <tr className="border-b border-[#222222]">
                <td className="py-1 px-2 font-mono text-white">Mkt</td>
                <td className="py-1 px-2"><strong className="text-white">Retorno del mercado</strong>: Retorno real del S&P 500 ese día.</td>
              </tr>
              <tr className="border-b border-[#222222]">
                <td className="py-1 px-2 font-mono text-white">Strat</td>
                <td className="py-1 px-2">
                  <strong className="text-white">Retorno estrategia</strong>: <code className="bg-[#0a0a0a] px-1 rounded text-[10px]">r_f + pos × (r_mkt - r_f) - costos</code>
                </td>
              </tr>
              <tr className="border-b border-[#222222] bg-[#f59e0b]/5">
                <td className="py-1 px-2 font-mono text-[#f59e0b]">Cost</td>
                <td className="py-1 px-2">
                  <strong className="text-white">Costo de transacción</strong>: Bid-ask spread al cambiar de posición (%). Solo aparece en días con cambio de posición.
                </td>
              </tr>
              <tr className="border-b border-[#222222] bg-[#00c853]/5">
                <td className="py-1 px-2 font-mono text-[#00c853]">Capital</td>
                <td className="py-1 px-2"><strong className="text-white">Capital acumulado</strong>: Valor del portafolio en USD (inicio: $10,000).</td>
              </tr>
              <tr className="border-b border-[#222222] bg-[#00c853]/5">
                <td className="py-1 px-2 font-mono text-[#00c853]">Acum.</td>
                <td className="py-1 px-2"><strong className="text-white">Retorno acumulado</strong>: Ganancia/pérdida total desde el inicio (%).</td>
              </tr>
              <tr className="border-b border-[#222222] bg-[#c41e3a]/5">
                <td className="py-1 px-2 font-mono text-[#c41e3a]">DD</td>
                <td className="py-1 px-2"><strong className="text-white">Drawdown</strong>: Caída desde el máximo histórico (High Water Mark).</td>
              </tr>
              <tr className="border-b border-[#222222]">
                <td className="py-1 px-2 font-mono text-white">✓/✗</td>
                <td className="py-1 px-2">
                  <strong className="text-white">Acierto</strong>:
                  <span className="text-[#00c853]"> ✓</span> (pos+sube ó cash+baja),
                  <span className="text-[#c41e3a]"> ✗</span> (pos+baja ó cash+sube)
                </td>
              </tr>
              <tr>
                <td className="py-1 px-2 font-mono text-white">Rég.</td>
                <td className="py-1 px-2">
                  <strong className="text-white">Régimen</strong>:
                  <span className="text-[#00c853]"> bull</span>,
                  <span className="text-[#c41e3a]"> bear</span>,
                  <span className="text-[#f59e0b]"> high_vol</span>,
                  <span className="text-[#737373]"> sideways</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Percentile System Explanation */}
      <div className="card border-[#6366f1]/30">
        <h3 className="text-lg font-medium mb-4">
          Sistema de Percentiles: Conversión de Predicciones a Posiciones
        </h3>

        <div className="text-sm text-[#a3a3a3] space-y-6">
          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">¿Por qué usar Percentiles en lugar de Predicciones Brutas?</h4>
            <div className="space-y-3">
              <p>
                Los modelos de machine learning generan predicciones de retorno (ej: +0.15%, -0.08%) que varían
                significativamente en magnitud según las condiciones del mercado. Utilizar umbrales fijos
                (ej: &quot;si predicción &gt; 0.1%, ir long&quot;) presenta problemas fundamentales:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="p-3 bg-[#0a0a0a] rounded border-l-4 border-[#c41e3a]">
                  <div className="text-[#c41e3a] font-medium mb-2">Problema con Umbrales Fijos</div>
                  <ul className="text-xs space-y-1">
                    <li>En mercados volátiles, predicciones de ±0.5% son comunes</li>
                    <li>En mercados tranquilos, predicciones de ±0.05% son significativas</li>
                    <li>Un umbral fijo estaría siempre long en volatilidad, siempre cash en calma</li>
                  </ul>
                </div>
                <div className="p-3 bg-[#0a0a0a] rounded border-l-4 border-[#00c853]">
                  <div className="text-[#00c853] font-medium mb-2">Solución: Percentiles Adaptativos</div>
                  <ul className="text-xs space-y-1">
                    <li>Compara cada predicción con las últimas 63 predicciones</li>
                    <li>Pregunta: &quot;¿Esta predicción es alta o baja <em>para el contexto actual</em>?&quot;</li>
                    <li>Se adapta automáticamente a diferentes regímenes de mercado</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">Ventana Móvil de 63 Días</h4>
            <div className="space-y-3">
              <p>
                El sistema utiliza una <strong className="text-white">ventana móvil de 63 días de trading</strong> (aproximadamente
                3 meses calendario) para calcular el percentil de cada predicción:
              </p>
              <div className="bg-[#0a0a0a] p-4 rounded font-mono text-xs">
                <div className="text-[#737373] mb-2">Ejemplo de cálculo para el día t:</div>
                <div className="space-y-1">
                  <div>Predicción actual (día t): <span className="text-[#00c853]">+0.12%</span></div>
                  <div>Predicciones en ventana [t-63, t-1]:</div>
                  <div className="pl-4 text-[#737373]">
                    -0.25%, -0.18%, -0.15%, ..., +0.08%, +0.10%, +0.15%, +0.22%, +0.31%
                  </div>
                  <div className="mt-2">De las 63 predicciones anteriores:</div>
                  <div className="pl-4">58 son menores que +0.12%</div>
                  <div className="pl-4">5 son mayores que +0.12%</div>
                  <div className="mt-2">
                    Percentil = (58 / 63) × 100 = <span className="text-[#00c853] font-bold">92%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">Mapeo de Percentiles a Posiciones (LONG-ONLY)</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#333333]">
                    <th className="text-left py-2 px-3">Rango de Percentil</th>
                    <th className="text-center py-2 px-3">Posición</th>
                    <th className="text-center py-2 px-3">Instrumento</th>
                    <th className="text-left py-2 px-3">Interpretación</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-[#222222] bg-[#00c853]/10">
                    <td className="py-2 px-3"><span className="text-[#00c853] font-bold">≥ 90%</span></td>
                    <td className="py-2 px-3 text-center text-[#00c853] font-bold">+3x</td>
                    <td className="py-2 px-3 text-center">UPRO</td>
                    <td className="py-2 px-3">Señal alcista extrema - máxima convicción long</td>
                  </tr>
                  <tr className="border-b border-[#222222] bg-[#22d3ee]/5">
                    <td className="py-2 px-3"><span className="text-[#22d3ee]">70% - 89%</span></td>
                    <td className="py-2 px-3 text-center text-[#22d3ee] font-bold">+1x</td>
                    <td className="py-2 px-3 text-center">SPY</td>
                    <td className="py-2 px-3">Señal alcista moderada - posición long estándar</td>
                  </tr>
                  <tr className="bg-[#525252]/10">
                    <td className="py-2 px-3"><span className="text-[#737373]">&lt; 70%</span></td>
                    <td className="py-2 px-3 text-center text-[#737373] font-bold">0x</td>
                    <td className="py-2 px-3 text-center">CASH</td>
                    <td className="py-2 px-3">Señal no alcista - sin exposición al mercado</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="p-4 bg-gradient-to-r from-[#1a1a2e] to-[#0f1729] rounded-lg border border-[#333333]">
            <h4 className="font-medium text-white mb-3">Fórmula de Retorno de la Estrategia</h4>
            <div className="bg-[#0a0a0a] p-4 rounded font-mono text-sm text-center">
              <span className="text-white">r</span>
              <sub className="text-[#6366f1]">estrategia</sub>
              <span className="text-white"> = r</span>
              <sub className="text-[#737373]">f</sub>
              <span className="text-white"> + posición × (r</span>
              <sub className="text-[#00c853]">mercado</sub>
              <span className="text-white"> - r</span>
              <sub className="text-[#737373]">f</sub>
              <span className="text-white">) - costos</span>
            </div>
            <div className="mt-4 text-xs text-[#a3a3a3] space-y-2">
              <p>Donde:</p>
              <ul className="pl-4 space-y-1">
                <li><strong className="text-white">r<sub>f</sub></strong> = Tasa libre de riesgo diaria (≈ 0.02%)</li>
                <li><strong className="text-white">posición</strong> = 0, 1, o 3 según el percentil</li>
                <li><strong className="text-white">r<sub>mercado</sub></strong> = Retorno del S&P 500</li>
                <li><strong className="text-white">costos</strong> = Bid-ask spread + comisiones (si hubo cambio de posición)</li>
              </ul>
              <div className="mt-4 p-3 bg-[#1a1a1a] rounded">
                <div className="text-white font-medium mb-2">Ejemplos:</div>
                <div className="space-y-1 font-mono text-xs">
                  <div><span className="text-[#00c853]">3x + mercado +1%:</span> 0.02% + 3 × (1% - 0.02%) = <span className="text-[#00c853]">+2.96%</span></div>
                  <div><span className="text-[#c41e3a]">3x + mercado -1%:</span> 0.02% + 3 × (-1% - 0.02%) = <span className="text-[#c41e3a]">-3.04%</span></div>
                  <div><span className="text-[#737373]">0x + mercado ±X%:</span> 0.02% + 0 × (X% - 0.02%) = <span className="text-[#737373]">+0.02%</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
