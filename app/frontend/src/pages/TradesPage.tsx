import { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getModels, getModelTrades, getModelDetail } from '../services/api';
import type { ModelsResponse, Trade, ModelData } from '../types';
import { createChart, IChartApi } from 'lightweight-charts';
import { Filter } from 'lucide-react';
import { clsx } from 'clsx';
import { calculateTradesWithCapital, INITIAL_CAPITAL } from '../utils/transactionCosts';

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
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'winning' | 'losing'>('all');
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

  // Use shared utility function for consistent calculations across all pages
  const tradesWithCapital = calculateTradesWithCapital(trades, INITIAL_CAPITAL);

  // Apply filter to trades with capital
  const filteredTradesWithCapital = tradesWithCapital.filter(trade => {
    if (filter === 'winning') return trade.total_return > 0;
    if (filter === 'losing') return trade.total_return < 0;
    return true;
  });

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
    return <div className="text-center py-8 text-[#737373]">Loading...</div>;
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
              ({activeTrades.length > 0 ? ((winningTrades.length / activeTrades.length) * 100).toFixed(0) : 0}%)
            </span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Losing</div>
          <div className="metric-value text-[#c41e3a]">
            {losingTrades.length}
            <span className="text-sm text-[#525252] ml-1">
              ({activeTrades.length > 0 ? ((losingTrades.length / activeTrades.length) * 100).toFixed(0) : 0}%)
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
                  <span>{histogramBins[0]?.min.toFixed(1)}%</span>
                  <span className="font-medium text-white">0%</span>
                  <span>{histogramBins[histogramBins.length - 1]?.max.toFixed(1)}%</span>
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
                  {histogramBins[0]?.min.toFixed(2)}%
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
                    ? ([...activeTrades].sort((a, b) => a.total_return - b.total_return)[Math.floor(activeTrades.length / 2)]?.total_return * 100).toFixed(2)
                    : 0}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-[#525252]">Max Return</div>
                <div className={clsx(
                  "font-mono font-medium",
                  histogramBins[histogramBins.length - 1]?.max >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                )}>
                  {histogramBins[histogramBins.length - 1]?.max.toFixed(2)}%
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

      {/* Trade Log */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-medium text-white">Trade Log ({filteredTradesWithCapital.length} trades)</h3>
            <p className="text-xs text-[#525252] mt-1">
              Initial Capital: ${INITIAL_CAPITAL.toLocaleString()} | Sorted by date (oldest first)
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-[#525252]" />
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as any)}
              className="bg-[#1a1a1a] border border-[#222222] rounded px-3 py-1.5 text-sm text-white"
            >
              <option value="all">All Trades</option>
              <option value="winning">Winning Only</option>
              <option value="losing">Losing Only</option>
            </select>
          </div>
        </div>

        <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-[#111111] z-10">
              <tr className="border-b border-[#333]">
                <th className="px-2 py-2 text-center text-xs font-semibold text-[#737373] uppercase tracking-wider w-10">#</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-[#737373] uppercase tracking-wider w-24">Entry</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-[#737373] uppercase tracking-wider w-24">Exit</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-[#737373] uppercase tracking-wider w-12">Days</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-[#737373] uppercase tracking-wider w-12">Pos</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-[#737373] uppercase tracking-wider w-20">Mkt Ret</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-[#737373] uppercase tracking-wider w-20">Strat Ret</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-[#737373] uppercase tracking-wider w-20">Tx Cost</th>
                <th className="px-2 py-2 text-right text-xs font-semibold text-[#737373] uppercase tracking-wider w-28">Net Gained</th>
                <th className="px-2 py-2 text-right text-xs font-semibold text-[#737373] uppercase tracking-wider w-28">Cumulative</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#222]">
              {filteredTradesWithCapital.map((trade, idx) => (
                <tr key={idx} className="hover:bg-[#1a1a1a] transition-colors">
                  <td className="px-2 py-1.5 text-center text-xs text-[#525252]">{idx + 1}</td>
                  <td className="px-2 py-1.5 text-center text-xs text-[#a3a3a3] font-mono">{trade.entry_date}</td>
                  <td className="px-2 py-1.5 text-center text-xs text-[#a3a3a3] font-mono">{trade.exit_date}</td>
                  <td className="px-2 py-1.5 text-center text-xs font-mono text-[#a3a3a3]">{trade.duration}</td>
                  <td className={clsx(
                    "px-2 py-1.5 text-center text-xs font-mono font-semibold",
                    trade.entry_position > 0 ? "text-[#00c853]" : trade.entry_position < 0 ? "text-[#c41e3a]" : "text-[#737373]"
                  )}>
                    {trade.entry_position > 0 ? '+' : ''}{trade.entry_position}x
                  </td>
                  <td className={clsx(
                    "px-2 py-1.5 text-center text-xs font-mono",
                    trade.market_return >= 0 ? "text-[#4ade80]" : "text-[#f87171]"
                  )}>
                    {trade.market_return >= 0 ? '+' : ''}{(trade.market_return * 100).toFixed(2)}%
                  </td>
                  <td className={clsx(
                    "px-2 py-1.5 text-center text-xs font-mono font-semibold",
                    trade.total_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                  )}>
                    {trade.total_return >= 0 ? '+' : ''}{(trade.total_return * 100).toFixed(2)}%
                  </td>
                  <td className="px-2 py-1.5 text-center text-xs font-mono text-[#f59e0b]">
                    {trade.txCostDollars > 0 ? `-$${trade.txCostDollars.toFixed(0)}` : '-'}
                  </td>
                  <td className={clsx(
                    "px-2 py-1.5 text-right text-xs font-mono",
                    trade.capitalGained >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                  )}>
                    {trade.capitalGained >= 0 ? '+$' : '-$'}{Math.abs(trade.capitalGained).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className={clsx(
                    "px-2 py-1.5 text-right text-xs font-mono font-semibold",
                    trade.cumulativeCapital >= INITIAL_CAPITAL ? "text-[#00c853]" : "text-[#c41e3a]"
                  )}>
                    ${trade.cumulativeCapital.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Summary row */}
        {tradesWithCapital.length > 0 && (
          <>
            <div className="mt-4 pt-4 border-t border-[#222222]">
              <div className="flex justify-between items-center mb-3">
                <div className="text-sm text-[#737373]">
                  Final Capital after {tradesWithCapital.length} trades (with tx costs):
                </div>
                <div className={clsx(
                  "text-xl font-mono font-bold",
                  tradesWithCapital[tradesWithCapital.length - 1].cumulativeCapital >= INITIAL_CAPITAL
                    ? "text-[#00c853]"
                    : "text-[#c41e3a]"
                )}>
                  ${tradesWithCapital[tradesWithCapital.length - 1].cumulativeCapital.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  <span className="text-sm ml-2">
                    ({tradesWithCapital[tradesWithCapital.length - 1].cumulativeCapital >= INITIAL_CAPITAL ? '+' : ''}
                    {(((tradesWithCapital[tradesWithCapital.length - 1].cumulativeCapital / INITIAL_CAPITAL) - 1) * 100).toFixed(2)}%)
                  </span>
                </div>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-[#737373]">Total Transaction Costs (entry + exit per trade):</span>
                <span className="font-mono text-[#f59e0b]">
                  -${tradesWithCapital.reduce((sum, t) => sum + t.txCostDollars, 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>
            </div>

            {/* Transaction cost breakdown */}
            <div className="mt-4 p-3 bg-[#1a1a1a] rounded-lg border border-[#333]">
              <h4 className="text-sm font-medium text-[#737373] mb-2">Transaction Costs (bid-ask spreads per trade)</h4>
              <div className="grid grid-cols-5 gap-2 text-xs text-center">
                <div className="p-2 bg-[#0a0a0a] rounded">
                  <div className="text-[#00c853] font-mono">+3x</div>
                  <div className="text-[#525252]">UPRO</div>
                  <div className="text-[#f59e0b]">0.10%</div>
                </div>
                <div className="p-2 bg-[#0a0a0a] rounded">
                  <div className="text-[#00c853] font-mono">+1x</div>
                  <div className="text-[#525252]">SPY</div>
                  <div className="text-[#f59e0b]">0.04%</div>
                </div>
                <div className="p-2 bg-[#0a0a0a] rounded">
                  <div className="text-[#737373] font-mono">0</div>
                  <div className="text-[#525252]">Cash</div>
                  <div className="text-[#525252]">0%</div>
                </div>
                <div className="p-2 bg-[#0a0a0a] rounded">
                  <div className="text-[#c41e3a] font-mono">-1x</div>
                  <div className="text-[#525252]">SH</div>
                  <div className="text-[#f59e0b]">0.08%</div>
                </div>
                <div className="p-2 bg-[#0a0a0a] rounded">
                  <div className="text-[#c41e3a] font-mono">-3x</div>
                  <div className="text-[#525252]">SPXU</div>
                  <div className="text-[#f59e0b]">0.12%</div>
                </div>
              </div>
              <p className="text-xs text-[#525252] mt-2 text-center">
                Cumulative = Previous Capital + (Strategy Return × Capital) - Transaction Cost
              </p>
            </div>
          </>
        )}

        {filteredTradesWithCapital.length === 0 && (
          <div className="text-center py-8 text-[#525252]">No trades to display</div>
        )}
      </div>
    </div>
  );
}
