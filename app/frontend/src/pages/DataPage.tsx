import { useEffect, useState, useMemo } from 'react';
import { getModels, getModelDailyLog, DailyLogEntry } from '../services/api';
import type { ModelsResponse } from '../types';
import { Database, Download, ChevronDown, CheckCircle, XCircle } from 'lucide-react';
import { clsx } from 'clsx';

const INITIAL_CAPITAL = 10000;

function formatPercent(value: number, decimals: number = 2): string {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(decimals)}%`;
}

function formatPercentile(value: number): string {
  return `${value.toFixed(0)}%`;
}

function formatPrediction(value: number): string {
  // Format as percentage return
  const pct = value * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
}

function downloadDailyLogCSV(dailyLog: DailyLogEntry[], modelName: string) {
  const headers = [
    'Day',
    'Date',
    'Prediction_BP',
    'Percentile',
    'Position_Base',
    'Position_Final',
    'Market_Return',
    'Strategy_Return',
    'Equity',
    'HWM',
    'Drawdown',
    'Trading_Cost',
    'Regime',
    'Position_Changed',
    'Direction_Correct'
  ];

  const rows = dailyLog.map((day) => {
    return [
      day.day_num,
      day.date,
      (day.prediction * 10000).toFixed(2),
      day.percentile.toFixed(1),
      day.position_base,
      day.position_final,
      (day.market_return * 100).toFixed(4),
      (day.strategy_return * 100).toFixed(4),
      (day.equity * INITIAL_CAPITAL).toFixed(2),
      (day.high_water_mark * INITIAL_CAPITAL).toFixed(2),
      (day.drawdown * 100).toFixed(2),
      (day.trading_cost * 100).toFixed(4),
      day.regime,
      day.position_changed ? 'YES' : 'NO',
      day.direction_correct ? 'YES' : 'NO'
    ];
  });

  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.join(','))
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', `${modelName}_daily_log.csv`);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Get percentile color based on value
function getPercentileColor(percentile: number): string {
  if (percentile >= 90) return 'text-[#00c853]';
  if (percentile >= 70) return 'text-[#4caf50]';
  if (percentile <= 10) return 'text-[#c41e3a]';
  if (percentile <= 30) return 'text-[#ef5350]';
  return 'text-[#737373]';
}

// Get prediction color based on value (percentage)
function getPredictionColor(prediction: number): string {
  const pct = prediction * 100;
  if (pct >= 0.5) return 'text-[#00c853]';
  if (pct >= 0.2) return 'text-[#4caf50]';
  if (pct <= -0.5) return 'text-[#c41e3a]';
  if (pct <= -0.2) return 'text-[#ef5350]';
  return 'text-[#a3a3a3]';
}

export default function DataPage() {
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [dailyLog, setDailyLog] = useState<DailyLogEntry[]>([]);
  const [warmupUsed, setWarmupUsed] = useState<boolean>(false);
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadModelData(selectedModel);
    }
  }, [selectedModel]);

  async function loadModels() {
    try {
      const data = await getModels();
      setModelsData(data);
      if (data.models.length > 0) {
        setSelectedModel(data.models[0].model);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function loadModelData(modelName: string) {
    try {
      setLoadingDetail(true);
      const response = await getModelDailyLog(modelName);
      setDailyLog(response.daily_log || []);
      setWarmupUsed(response.warmup_used);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingDetail(false);
    }
  }

  // Statistics
  const stats = useMemo(() => {
    if (dailyLog.length === 0) return null;

    const totalDays = dailyLog.length;
    const positionChanges = dailyLog.filter(d => d.position_changed).length;
    // Direction accuracy: basado en posición vs movimiento del mercado
    // Acierto: (tengo posición Y mercado sube) O (no tengo posición Y mercado baja)
    const correctDirections = dailyLog.filter(d => {
      const hasPosition = d.position_final > 0;
      const marketUp = d.market_return > 0;
      return (hasPosition && marketUp) || (!hasPosition && !marketUp);
    }).length;
    // LONG-ONLY: separate 3x UPRO and 1x SPY days
    const days3x = dailyLog.filter(d => d.position_final === 3).length;
    const days1x = dailyLog.filter(d => d.position_final === 1).length;
    const cashDays = dailyLog.filter(d => d.position_final === 0).length;
    const winningDays = dailyLog.filter(d => d.strategy_return > 0).length;
    const finalEquity = dailyLog[dailyLog.length - 1].equity * INITIAL_CAPITAL;
    const totalReturn = (dailyLog[dailyLog.length - 1].equity - 1) * 100;
    const maxDrawdown = Math.min(...dailyLog.map(d => d.drawdown)) * 100;

    return {
      totalDays,
      positionChanges,
      correctDirections,
      dirAccuracy: totalDays > 0 ? (correctDirections / totalDays) * 100 : 0,
      days3x,
      days1x,
      cashDays,
      winningDays,
      winRate: (winningDays / totalDays) * 100,
      finalEquity,
      totalReturn,
      maxDrawdown
    };
  }, [dailyLog]);

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
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-[#737373]" />
          <h2 className="text-xl font-semibold">Daily Trading Log - Hedge Fund View</h2>
        </div>
        {dailyLog.length > 0 && (
          <button
            onClick={() => downloadDailyLogCSV(dailyLog, selectedModel)}
            className="flex items-center gap-2 px-4 py-2 bg-[#00c853] text-black rounded font-medium hover:bg-[#00c853]/90 transition-colors"
          >
            <Download className="w-4 h-4" />
            Download CSV
          </button>
        )}
      </div>

      {/* Model Selector */}
      <div className="card">
        <div className="flex items-center gap-4 flex-wrap">
          <span className="text-sm text-[#737373]">Select Model:</span>
          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center gap-2 px-4 py-2 bg-[#1a1a1a] rounded border border-[#333333] hover:border-[#444444] transition-colors min-w-[200px]"
            >
              <span className="flex-1 text-left">{selectedModel || 'Select a model'}</span>
              <ChevronDown className={clsx("w-4 h-4 transition-transform", dropdownOpen && "rotate-180")} />
            </button>
            {dropdownOpen && (
              <div className="absolute top-full left-0 mt-1 w-full bg-[#1a1a1a] border border-[#333333] rounded shadow-lg z-10 max-h-64 overflow-y-auto">
                {modelsData.models.map((m) => (
                  <button
                    key={m.model}
                    onClick={() => {
                      setSelectedModel(m.model);
                      setDropdownOpen(false);
                    }}
                    className={clsx(
                      "w-full px-4 py-2 text-left hover:bg-[#222222] transition-colors text-sm",
                      selectedModel === m.model && "bg-[#222222] text-[#c41e3a]"
                    )}
                  >
                    {m.model}
                  </button>
                ))}
              </div>
            )}
          </div>
          {dailyLog.length > 0 && (
            <div className="flex items-center gap-4 text-sm text-[#737373]">
              <span>{dailyLog.length} trading days</span>
              {warmupUsed && (
                <span className="px-2 py-0.5 bg-[#6366f1]/20 text-[#6366f1] rounded text-xs">
                  Warmup: 63d training data
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <div className="metric-card">
            <div className="metric-label">Trading Days</div>
            <div className="metric-value text-white">{stats.totalDays}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Final Equity</div>
            <div className={clsx("metric-value", stats.totalReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]")}>
              ${stats.finalEquity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Total Return</div>
            <div className={clsx("metric-value", stats.totalReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]")}>
              {stats.totalReturn >= 0 ? '+' : ''}{stats.totalReturn.toFixed(1)}%
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Dir. Accuracy</div>
            <div className={clsx("metric-value", stats.dirAccuracy >= 50 ? "text-[#00c853]" : "text-[#c41e3a]")}>
              {stats.dirAccuracy.toFixed(1)}%
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Win Rate</div>
            <div className={clsx("metric-value", stats.winRate >= 50 ? "text-[#00c853]" : "text-[#c41e3a]")}>
              {stats.winRate.toFixed(1)}%
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Max Drawdown</div>
            <div className="metric-value text-[#c41e3a]">
              {stats.maxDrawdown.toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Position Distribution - LONG-ONLY */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="metric-card border-[#00c853]/30">
            <div className="metric-label text-[#00c853]">3x UPRO Days</div>
            <div className="metric-value text-[#00c853]">
              {stats.days3x}
              <span className="text-sm text-[#525252] ml-1">
                ({((stats.days3x / stats.totalDays) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
          <div className="metric-card border-[#22d3ee]/30">
            <div className="metric-label text-[#22d3ee]">1x SPY Days</div>
            <div className="metric-value text-[#22d3ee]">
              {stats.days1x}
              <span className="text-sm text-[#525252] ml-1">
                ({((stats.days1x / stats.totalDays) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Cash Days</div>
            <div className="metric-value text-[#737373]">
              {stats.cashDays}
              <span className="text-sm text-[#525252] ml-1">
                ({((stats.cashDays / stats.totalDays) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
          <div className="metric-card border-[#6366f1]/30">
            <div className="metric-label text-[#6366f1]">Position Changes</div>
            <div className="metric-value text-[#6366f1]">
              {stats.positionChanges}
              <span className="text-sm text-[#525252] ml-1">
                ({((stats.positionChanges / stats.totalDays) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Column Definitions - Above the table */}
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

      {/* Daily Log Table */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Daily Log - {selectedModel}</h3>

        {loadingDetail ? (
          <div className="text-center py-8 text-[#737373]">Loading daily data...</div>
        ) : dailyLog.length > 0 ? (
          <>
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
              <table className="table-dark">
                <thead className="sticky top-0 bg-[#111111] z-10">
                  <tr>
                    <th className="text-center px-2">#</th>
                    <th className="text-center px-2">Fecha</th>
                    <th className="text-center px-2 bg-[#6366f1]/10">Pred.</th>
                    <th className="text-center px-2 bg-[#6366f1]/10">%ile</th>
                    <th className="text-center px-2">Pos</th>
                    <th className="text-center px-2">Mkt</th>
                    <th className="text-center px-2">Strat</th>
                    <th className="text-center px-2 bg-[#00c853]/5">Capital</th>
                    <th className="text-center px-2 bg-[#00c853]/5">Acum.</th>
                    <th className="text-center px-2 bg-[#c41e3a]/5">DD</th>
                    <th className="text-center px-1">✓/✗</th>
                    <th className="text-center px-2">Rég.</th>
                  </tr>
                </thead>
                <tbody>
                  {dailyLog.map((day, idx) => {
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
                        {/* Predicción */}
                        <td className={clsx(
                          "text-center font-mono text-xs px-2 bg-[#6366f1]/5",
                          getPredictionColor(day.prediction)
                        )}>
                          {formatPrediction(day.prediction)}
                        </td>
                        {/* Percentil */}
                        <td className={clsx(
                          "text-center font-mono text-xs font-semibold px-2 bg-[#6366f1]/5",
                          getPercentileColor(day.percentile)
                        )}>
                          {formatPercentile(day.percentile)}
                        </td>
                        {/* Posición */}
                        <td className={clsx(
                          "text-center font-mono text-xs font-bold px-2",
                          position === 3 && "text-[#00c853]",
                          position === 1 && "text-[#22d3ee]",
                          position === 0 && "text-[#737373]"
                        )}>
                          {position === 3 ? '3x' : position === 1 ? '1x' : '0x'}
                        </td>
                        {/* Ret. Mercado */}
                        <td className={clsx(
                          "text-center font-mono text-xs px-2",
                          day.market_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          {formatPercent(day.market_return, 2)}
                        </td>
                        {/* Ret. Estrategia */}
                        <td className={clsx(
                          "text-center font-mono text-xs font-semibold px-2",
                          day.strategy_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          {formatPercent(day.strategy_return, 2)}
                        </td>
                        {/* Capital */}
                        <td className={clsx(
                          "text-center font-mono text-xs px-2 bg-[#00c853]/5",
                          equity >= INITIAL_CAPITAL ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          ${equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        {/* Retorno Acumulado */}
                        <td className={clsx(
                          "text-center font-mono text-xs px-2 bg-[#00c853]/5 font-semibold",
                          cumReturn >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          {cumReturn >= 0 ? '+' : ''}{cumReturn.toFixed(1)}%
                        </td>
                        {/* Drawdown */}
                        <td className={clsx(
                          "text-center font-mono text-xs px-2 bg-[#c41e3a]/5",
                          day.drawdown < -0.10 ? "text-[#c41e3a] font-semibold" :
                          day.drawdown < -0.05 ? "text-[#f59e0b]" : "text-[#737373]"
                        )}>
                          {day.drawdown !== 0 ? `${(day.drawdown * 100).toFixed(1)}%` : '-'}
                        </td>
                        {/* Acierto: basado en posición vs movimiento del mercado */}
                        {(() => {
                          const hasPosition = position > 0;
                          const marketUp = day.market_return > 0;
                          // Acierto: (tengo posición Y mercado sube) O (no tengo posición Y mercado baja)
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
                        {/* Régimen */}
                        <td className={clsx(
                          "text-center font-mono text-xs px-2",
                          day.regime === 'bull' && "text-[#00c853]",
                          day.regime === 'bear' && "text-[#c41e3a]",
                          day.regime === 'high_vol' && "text-[#f59e0b]",
                          day.regime === 'sideways' && "text-[#737373]",
                          day.regime === 'unknown' && "text-[#525252]"
                        )}>
                          {day.regime === 'unknown' ? '-' : day.regime}
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
                Showing {dailyLog.length} trading days.
                <span className="text-[#6366f1] ml-2">
                  {dailyLog.filter(d => d.position_changed).length} position changes.
                </span>
                <span className="text-[#f59e0b] ml-2">
                  {dailyLog.filter(d => d.position_base !== d.position_final).length} days with risk management reduction (marked with !).
                </span>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-8 text-[#737373]">Select a model to view daily data</div>
        )}
      </div>

      {/* ============================================================ */}
      {/* PERCENTILE SYSTEM EXPLANATION */}
      {/* ============================================================ */}
      <div className="card border-[#6366f1]/30">
        <h3 className="text-lg font-medium mb-4">
          Sistema de Percentiles: Conversión de Predicciones a Posiciones
        </h3>

        <div className="text-sm text-[#a3a3a3] space-y-6">
          {/* Why Percentiles */}
          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">¿Por qué usar Percentiles en lugar de Predicciones Brutas?</h4>
            <div className="space-y-3">
              <p>
                Los modelos de machine learning generan predicciones de retorno (ej: +0.15%, -0.08%) que varían
                significativamente en magnitud según las condiciones del mercado. Utilizar umbrales fijos
                (ej: "si predicción &gt; 0.1%, ir long") presenta problemas fundamentales:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="p-3 bg-[#0a0a0a] rounded border-l-4 border-[#c41e3a]">
                  <div className="text-[#c41e3a] font-medium mb-2">Problema con Umbrales Fijos</div>
                  <ul className="text-xs space-y-1">
                    <li>• En mercados volátiles, predicciones de ±0.5% son comunes</li>
                    <li>• En mercados tranquilos, predicciones de ±0.05% son significativas</li>
                    <li>• Un umbral fijo de 0.1% estaría siempre long en volatilidad, siempre cash en calma</li>
                  </ul>
                </div>
                <div className="p-3 bg-[#0a0a0a] rounded border-l-4 border-[#00c853]">
                  <div className="text-[#00c853] font-medium mb-2">Solución: Percentiles Adaptativos</div>
                  <ul className="text-xs space-y-1">
                    <li>• Compara cada predicción con las últimas 63 predicciones</li>
                    <li>• Pregunta: "¿Esta predicción es alta o baja <em>para el contexto actual</em>?"</li>
                    <li>• Se adapta automáticamente a diferentes regímenes de mercado</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Rolling Window Explanation */}
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
                  <div className="pl-4">• 58 son menores que +0.12%</div>
                  <div className="pl-4">• 5 son mayores que +0.12%</div>
                  <div className="mt-2">
                    Percentil = (58 / 63) × 100 = <span className="text-[#00c853] font-bold">92%</span>
                  </div>
                </div>
              </div>
              <p className="text-[#6366f1]">
                <strong>Interpretación:</strong> Una predicción en el percentil 92 significa que es más optimista
                que el 92% de las predicciones recientes. Esto indica una señal alcista relativamente fuerte
                para el contexto actual del mercado.
              </p>
            </div>
          </div>

          {/* Percentile to Position Mapping */}
          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">Mapeo de Percentiles a Posiciones (LONG-ONLY)</h4>
            <div className="space-y-3">
              <p>
                Una vez calculado el percentil, se aplica la siguiente tabla de conversión para determinar
                la <strong className="text-white">posición</strong>:
              </p>
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
              <div className="bg-[#0a0a0a] p-3 rounded mt-4">
                <div className="text-white font-medium mb-2">Estrategia LONG-ONLY</div>
                <p className="text-xs">
                  Esta estrategia solo toma posiciones largas (+3x UPRO, +1x SPY) o se queda en efectivo.
                  <strong className="text-[#00c853]"> Percentil ≥ 90</strong> → UPRO (3x),
                  <strong className="text-[#22d3ee]"> Percentil 70-89</strong> → SPY (1x),
                  <strong className="text-[#737373]"> Percentil &lt; 70</strong> → Cash.
                  No hay posiciones cortas (shorts).
                </p>
              </div>
            </div>
          </div>

          {/* ============================================================ */}
          {/* METRICS EXPLANATION */}
          {/* ============================================================ */}
          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">Métricas de Evaluación</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-3 bg-[#0a0a0a] rounded border-l-4 border-[#6366f1]">
                <div className="text-[#6366f1] font-medium mb-2">Dir. Accuracy (Precisión Direccional)</div>
                <p className="text-xs text-[#a3a3a3] mb-2">
                  Porcentaje de días donde la <strong className="text-white">decisión de posición fue correcta</strong>:
                </p>
                <ul className="text-xs space-y-1 text-[#a3a3a3]">
                  <li><span className="text-[#00c853]">✓ Acierto:</span> Posición (1x/3x) + mercado sube</li>
                  <li><span className="text-[#00c853]">✓ Acierto:</span> Cash (0x) + mercado baja</li>
                  <li><span className="text-[#c41e3a]">✗ Error:</span> Posición (1x/3x) + mercado baja</li>
                  <li><span className="text-[#c41e3a]">✗ Error:</span> Cash (0x) + mercado sube</li>
                </ul>
                <p className="text-xs text-[#6366f1] mt-2">
                  Fórmula: (días correctos / total días) × 100
                </p>
              </div>
              <div className="p-3 bg-[#0a0a0a] rounded border-l-4 border-[#00c853]">
                <div className="text-[#00c853] font-medium mb-2">Win Rate (Tasa de Ganancia)</div>
                <p className="text-xs text-[#a3a3a3] mb-2">
                  Porcentaje de días donde el <strong className="text-white">retorno de la estrategia fue positivo</strong>.
                </p>
                <p className="text-xs text-[#a3a3a3]">
                  Incluye días en cash donde se gana el risk-free rate (~0.02% diario).
                </p>
                <p className="text-xs text-[#00c853] mt-2">
                  Fórmula: (días con Strat &gt; 0 / total días) × 100
                </p>
              </div>
              <div className="p-3 bg-[#0a0a0a] rounded border-l-4 border-[#c41e3a]">
                <div className="text-[#c41e3a] font-medium mb-2">Max Drawdown (Máxima Caída)</div>
                <p className="text-xs text-[#a3a3a3] mb-2">
                  <strong className="text-white">Mayor pérdida desde un máximo</strong> durante todo el período.
                </p>
                <p className="text-xs text-[#a3a3a3]">
                  Mide el peor escenario para un inversor que entró en el punto más alto.
                </p>
                <p className="text-xs text-[#a3a3a3] mt-2">
                  <strong className="text-white">HWM</strong> = High Water Mark (máximo histórico del capital)
                </p>
                <p className="text-xs text-[#c41e3a] mt-1">
                  Fórmula: min((Capital - HWM) / HWM) × 100
                </p>
              </div>
            </div>
          </div>

          {/* ============================================================ */}
          {/* WHY POSITIVE PREDICTION CAN RESULT IN CASH */}
          {/* ============================================================ */}
          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">¿Por qué una predicción positiva puede resultar en Cash?</h4>
            <div className="space-y-3">
              <p className="text-[#a3a3a3]">
                Este es un punto clave del sistema de percentiles. <strong className="text-white">Una predicción positiva
                no garantiza una posición larga</strong>. Lo que importa es cómo se compara con las predicciones recientes:
              </p>
              <div className="bg-[#0a0a0a] p-4 rounded font-mono text-xs">
                <div className="text-[#f59e0b] font-bold mb-2">Ejemplo concreto:</div>
                <div className="space-y-2">
                  <div>Predicción de hoy: <span className="text-[#00c853]">+0.10%</span> (positiva)</div>
                  <div>Últimas 63 predicciones: rango de <span className="text-[#00c853]">+0.05%</span> a <span className="text-[#00c853]">+0.50%</span></div>
                  <div className="mt-2 pt-2 border-t border-[#333]">
                    Percentil de +0.10% = <span className="text-[#f59e0b] font-bold">15%</span> (está en el extremo bajo del rango)
                  </div>
                  <div className="mt-2">
                    Posición resultante: <span className="text-[#737373] font-bold">0x (CASH)</span>
                  </div>
                </div>
              </div>
              <p className="text-[#6366f1]">
                <strong>Interpretación:</strong> Aunque +0.10% es positivo, el modelo ha estado prediciendo retornos
                más altos recientemente (+0.20% a +0.50%). Una predicción de +0.10% indica <em>menor convicción</em>
                que lo habitual, por lo que la estrategia opta por no tomar riesgo.
              </p>
              <div className="bg-[#0a0a0a] p-3 rounded mt-2 border border-[#333]">
                <div className="text-white font-medium mb-2">Lógica inversa también aplica:</div>
                <p className="text-xs text-[#a3a3a3]">
                  Una predicción de <span className="text-[#c41e3a]">-0.05%</span> (negativa) podría resultar en
                  <span className="text-[#00c853]"> 3x UPRO</span> si las últimas 63 predicciones fueron muy negativas
                  (ej: -0.50% a -0.10%). En ese contexto, -0.05% sería una señal alcista relativa.
                </p>
              </div>
            </div>
          </div>

          {/* ============================================================ */}
          {/* TRANSACTION COSTS */}
          {/* ============================================================ */}
          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">Costos de Transacción (Por qué el capital baja en Cash)</h4>
            <div className="space-y-3">
              <p className="text-[#a3a3a3]">
                El modelo incorpora <strong className="text-white">costos realistas de trading</strong>, lo que explica
                por qué el capital puede disminuir ligeramente incluso cuando la posición es 0x (Cash):
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-3 bg-[#0a0a0a] rounded">
                  <div className="text-[#f59e0b] font-medium mb-2">Costos por Cambio de Posición</div>
                  <ul className="text-xs space-y-1 text-[#a3a3a3]">
                    <li>• <strong className="text-white">Bid-Ask Spread UPRO:</strong> 0.05%</li>
                    <li>• <strong className="text-white">Bid-Ask Spread SPY:</strong> 0.02%</li>
                    <li>• <strong className="text-white">Comisión fija:</strong> 15 bps (0.15%)</li>
                  </ul>
                  <p className="text-xs text-[#f59e0b] mt-2">
                    Estos costos se aplican cada vez que hay un cambio de posición.
                  </p>
                </div>
                <div className="p-3 bg-[#0a0a0a] rounded">
                  <div className="text-[#22d3ee] font-medium mb-2">Expense Ratios (Diarios)</div>
                  <ul className="text-xs space-y-1 text-[#a3a3a3]">
                    <li>• <strong className="text-white">UPRO:</strong> 0.89% anual ≈ 0.0035% diario</li>
                    <li>• <strong className="text-white">SPY:</strong> 0.09% anual ≈ 0.0004% diario</li>
                    <li>• <strong className="text-white">Cash:</strong> Gana risk-free rate (~0.02% diario)</li>
                  </ul>
                </div>
              </div>
              <div className="bg-[#0a0a0a] p-4 rounded font-mono text-xs mt-2">
                <div className="text-white font-bold mb-2">Ejemplo de impacto:</div>
                <div className="space-y-1">
                  <div>Día 3: Posición 3x (UPRO), Capital = $10,835</div>
                  <div>Día 4: Cambia a 0x (Cash)</div>
                  <div className="pl-4 text-[#f59e0b]">→ Paga costo de salida de UPRO: ~0.05%</div>
                  <div className="pl-4">→ Capital = $10,835 × (1 - 0.0005) = <span className="text-white">$10,830</span></div>
                </div>
              </div>
            </div>
          </div>

          {/* ============================================================ */}
          {/* MARKET REGIME DEFINITION */}
          {/* ============================================================ */}
          <div className="p-4 bg-[#1a1a1a] rounded-lg">
            <h4 className="font-medium text-white mb-3">Definición del Régimen de Mercado</h4>
            <div className="space-y-3">
              <p className="text-[#a3a3a3] text-xs">
                El régimen se calcula usando una <strong className="text-white">ventana móvil de 60 días</strong> de retornos del S&P 500.
                Se evalúan dos métricas: el retorno acumulado y la volatilidad anualizada.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-[#333333]">
                      <th className="text-left py-2 px-2">Régimen</th>
                      <th className="text-left py-2 px-2">Condición</th>
                      <th className="text-left py-2 px-2">Interpretación</th>
                    </tr>
                  </thead>
                  <tbody className="text-[#a3a3a3]">
                    <tr className="border-b border-[#222222]">
                      <td className="py-2 px-2 font-mono text-[#00c853] font-bold">bull</td>
                      <td className="py-2 px-2">Retorno 60d &gt; +10% <strong className="text-white">Y</strong> Volatilidad &lt; 20%</td>
                      <td className="py-2 px-2">Mercado alcista con baja volatilidad</td>
                    </tr>
                    <tr className="border-b border-[#222222]">
                      <td className="py-2 px-2 font-mono text-[#c41e3a] font-bold">bear</td>
                      <td className="py-2 px-2">Retorno 60d &lt; -10%</td>
                      <td className="py-2 px-2">Mercado bajista (corrección o crash)</td>
                    </tr>
                    <tr className="border-b border-[#222222]">
                      <td className="py-2 px-2 font-mono text-[#f59e0b] font-bold">high_vol</td>
                      <td className="py-2 px-2">Volatilidad &gt; 25%</td>
                      <td className="py-2 px-2">Alta incertidumbre (sin importar dirección)</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-2 font-mono text-[#737373] font-bold">sideways</td>
                      <td className="py-2 px-2">Ninguna de las anteriores</td>
                      <td className="py-2 px-2">Mercado lateral o transición</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="bg-[#0a0a0a] p-3 rounded mt-2 text-xs">
                <p className="text-[#a3a3a3]">
                  <strong className="text-white">Volatilidad anualizada:</strong> σ<sub>anual</sub> = σ<sub>diaria</sub> × √252
                </p>
                <p className="text-[#a3a3a3] mt-1">
                  <strong className="text-white">Retorno acumulado 60d:</strong> ∏(1 + r<sub>t</sub>) - 1, donde t ∈ [día-60, día-1]
                </p>
              </div>
            </div>
          </div>

          {/* ============================================================ */}
          {/* STRATEGY FORMULA */}
          {/* ============================================================ */}
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
                <li>• <strong className="text-white">r<sub>f</sub></strong> = Tasa libre de riesgo diaria (≈ 0.02%)</li>
                <li>• <strong className="text-white">posición</strong> = 0, 1, o 3 según el percentil</li>
                <li>• <strong className="text-white">r<sub>mercado</sub></strong> = Retorno del S&P 500</li>
                <li>• <strong className="text-white">costos</strong> = Bid-ask spread + comisiones (si hubo cambio de posición)</li>
              </ul>
              <div className="mt-4 p-3 bg-[#1a1a1a] rounded">
                <div className="text-white font-medium mb-2">Ejemplos (sin cambio de posición):</div>
                <div className="space-y-1 font-mono text-xs">
                  <div><span className="text-[#00c853]">3x + mercado +1%:</span> 0.02% + 3 × (1% - 0.02%) - 0% = <span className="text-[#00c853]">+2.96%</span></div>
                  <div><span className="text-[#c41e3a]">3x + mercado -1%:</span> 0.02% + 3 × (-1% - 0.02%) - 0% = <span className="text-[#c41e3a]">-3.04%</span></div>
                  <div><span className="text-[#737373]">0x + mercado ±X%:</span> 0.02% + 0 × (X% - 0.02%) - 0% = <span className="text-[#737373]">+0.02%</span></div>
                </div>
                <div className="text-white font-medium mb-2 mt-3">Ejemplos (con cambio de posición):</div>
                <div className="space-y-1 font-mono text-xs">
                  <div><span className="text-[#f59e0b]">0x→3x + mercado +1%:</span> 0.02% + 3 × (1% - 0.02%) - 0.05% = <span className="text-[#00c853]">+2.91%</span></div>
                  <div><span className="text-[#f59e0b]">3x→0x + mercado -0.5%:</span> 0.02% + 0 × (-0.5% - 0.02%) - 0.05% = <span className="text-[#c41e3a]">-0.03%</span></div>
                  <div><span className="text-[#f59e0b]">0x→1x + mercado +0.5%:</span> 0.02% + 1 × (0.5% - 0.02%) - 0.02% = <span className="text-[#00c853]">+0.48%</span></div>
                </div>
                <p className="text-[#737373] text-[10px] mt-2">* Costos: UPRO bid-ask 0.05%, SPY bid-ask 0.02%</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
