import { useEffect, useState, useRef } from 'react';
import { getModels, getModelDetail } from '../services/api';
import type { ModelsResponse, ModelData } from '../types';
import { createChart, IChartApi } from 'lightweight-charts';
import { X } from 'lucide-react';
import { clsx } from 'clsx';
import { calculateSummaryFromTrades, buildNetEquityCurve, INITIAL_CAPITAL } from '../utils/transactionCosts';

// Axe Capital color palette
const MODEL_COLORS = [
  '#c41e3a', '#00c853', '#ff6b35', '#4ecdc4', '#a855f7',
  '#06b6d4', '#84cc16', '#f97316', '#64748b'
];

export default function ComparePage() {
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>(['RandomForest', 'Prophet']);
  const [modelDetails, setModelDetails] = useState<Record<string, ModelData>>({});
  const [loading, setLoading] = useState(true);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModels.length > 0) {
      loadModelDetails();
    }
  }, [selectedModels]);

  useEffect(() => {
    if (Object.keys(modelDetails).length > 0 && chartContainerRef.current) {
      initChart();
    }
    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [modelDetails]);

  async function loadModels() {
    try {
      const data = await getModels();
      setModelsData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function loadModelDetails() {
    try {
      const details: Record<string, ModelData> = {};
      await Promise.all(
        selectedModels.map(async (name) => {
          try {
            const detail = await getModelDetail(name);
            details[name] = detail;
          } catch (e) {
            console.error(`Failed to load ${name}:`, e);
          }
        })
      );
      setModelDetails(details);
    } catch (err) {
      console.error(err);
    }
  }

  function initChart() {
    if (!chartContainerRef.current) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
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
      localization: {
        priceFormatter: (price: number) => '$' + price.toLocaleString('en-US', { maximumFractionDigits: 0 }),
      },
    });

    chartRef.current = chart;

    selectedModels.forEach((modelName, idx) => {
      const data = modelDetails[modelName];
      if (!data) return;

      const series = chart.addLineSeries({
        color: MODEL_COLORS[idx % MODEL_COLORS.length],
        lineWidth: 2,
        title: modelName,
      });

      // Build net equity curve with transaction costs applied at correct trade dates
      const netEquity = buildNetEquityCurve(data.dates, data.equity_curve, data.trades, INITIAL_CAPITAL);

      const chartData = data.dates.map((date, i) => ({
        time: date,
        value: netEquity[i],
      }));
      series.setData(chartData as any);
    });

    const firstModel = modelDetails[selectedModels[0]];
    if (firstModel) {
      const marketSeries = chart.addLineSeries({
        color: '#525252',
        lineWidth: 2,
        lineStyle: 2,
        title: 'Buy & Hold',
      });

      // Convert market equity to actual dollar values
      const marketData = firstModel.dates.map((date, i) => ({
        time: date,
        value: firstModel.market_equity[i] * INITIAL_CAPITAL,
      }));
      marketSeries.setData(marketData as any);
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }

  function toggleModel(model: string) {
    if (selectedModels.includes(model)) {
      setSelectedModels(selectedModels.filter(m => m !== model));
    } else if (selectedModels.length < 4) {
      setSelectedModels([...selectedModels, model]);
    }
  }

  if (loading) {
    return <div className="text-center py-8 text-[#737373]">Loading...</div>;
  }

  if (!modelsData) {
    return <div className="text-center py-8 text-[#c41e3a]">Failed to load data</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-white">Model Comparison</h2>
        <span className="text-sm text-[#737373]">
          Select 2-4 models to compare
        </span>
      </div>

      {/* Model Selector */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4 text-white">Select Models ({selectedModels.length}/4)</h3>
        <div className="flex flex-wrap gap-2">
          {modelsData.models.map(model => (
            <button
              key={model.model}
              onClick={() => toggleModel(model.model)}
              className={clsx(
                "px-3 py-1.5 rounded-full text-sm font-medium transition-colors",
                selectedModels.includes(model.model)
                  ? "bg-[#c41e3a] text-white"
                  : "bg-[#1a1a1a] text-[#737373] hover:bg-[#222222] hover:text-white"
              )}
            >
              {model.model}
            </button>
          ))}
        </div>
      </div>

      {/* Selected Models */}
      <div className="flex gap-2 flex-wrap">
        {selectedModels.map((model, idx) => (
          <div
            key={model}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm text-white"
            style={{ backgroundColor: MODEL_COLORS[idx % MODEL_COLORS.length] }}
          >
            {model}
            <button onClick={() => toggleModel(model)} className="hover:text-gray-300">
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>

      {/* Comparison Cards */}
      {Object.keys(modelDetails).length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {selectedModels.map((modelName, idx) => {
            const model = modelDetails[modelName];
            if (!model) return null;

            return (
              <div key={modelName} className="card border-l-4" style={{ borderLeftColor: MODEL_COLORS[idx % MODEL_COLORS.length] }}>
                <h4 className="font-medium mb-3 text-white">{modelName}</h4>
                <div className="text-xs text-[#525252] mb-4">{model.category}</div>

                <div className="space-y-3">
                  {(() => {
                    const m = model.metrics;
                    // Calculate from trades (consistent with Trades page)
                    const tradeSummary = model.trades && model.trades.length > 0
                      ? calculateSummaryFromTrades(model.trades, INITIAL_CAPITAL)
                      : { finalCapital: INITIAL_CAPITAL, totalTxCosts: 0, netReturn: 0, grossReturn: 0 };
                    return (
                      <>
                        <div className="flex justify-between">
                          <span className="text-[#737373]">Net Return</span>
                          <span className={clsx(
                            "font-mono font-semibold",
                            tradeSummary.netReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                          )}>
                            {(tradeSummary.netReturn * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-[#737373]">Tx Costs</span>
                          <span className="font-mono text-[#f59e0b]">
                            -${tradeSummary.totalTxCosts.toFixed(0)}
                          </span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-[#737373]">Final Capital</span>
                          <span className="font-mono text-white">
                            ${tradeSummary.finalCapital.toFixed(0)}
                          </span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-[#737373]">Sharpe</span>
                          <span className="font-mono text-white">{m.sharpe?.toFixed(3) || '-'}</span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-[#737373]">Max DD</span>
                          <span className="font-mono text-[#c41e3a]">
                            {((m.max_drawdown ?? 0) * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-[#737373]">Trades</span>
                          <span className="font-mono text-[#a3a3a3]">
                            {m.n_trades || 0}
                          </span>
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Equity Curves Comparison */}
      {Object.keys(modelDetails).length > 0 && (
        <div className="card">
          <h3 className="text-lg font-medium mb-2 text-white">Equity Curves Comparison</h3>
          <div className="flex flex-wrap gap-3 mb-4 text-xs">
            {selectedModels.map((name, idx) => (
              <div key={name} className="flex items-center gap-1">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: MODEL_COLORS[idx % MODEL_COLORS.length] }}
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
      )}
    </div>
  );
}
