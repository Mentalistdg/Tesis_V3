import { useEffect, useState } from 'react';
import { getModels, getModelTrades } from '../services/api';
import type { ModelsResponse, Trade } from '../types';
import { Database, Download, ChevronDown, Info, AlertTriangle, TrendingDown, Activity } from 'lucide-react';
import { clsx } from 'clsx';

const POSITION_LABELS: { [key: number]: string } = {
  3: 'UPRO (3x Long)',
  1: 'SPY (1x Long)',
  0: 'CASH',
};
POSITION_LABELS[-1] = 'SH (1x Short)';
POSITION_LABELS[-3] = 'SPXU (3x Short)';

function formatPercent(value: number, decimals: number = 2): string {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(decimals)}%`;
}

function formatPercentile(value: number): string {
  return `${value.toFixed(0)}%ile`;
}

function downloadTradesCSV(trades: Trade[], modelName: string) {
  const headers = [
    'Trade_Number',
    'Entry_Date',
    'Exit_Date',
    'Duration_Days',
    'Base_Position',
    'Final_Position',
    'Risk_Mgmt_Applied',
    'Entry_Percentile',
    'Entry_Prediction',
    'Market_Return',
    'Strategy_Return'
  ];

  const rows = trades.map((trade, i) => {
    const base = trade.base_position ?? trade.entry_position;
    const final = trade.entry_position;
    const riskApplied = base !== final ? 'YES' : 'NO';
    return [
      i + 1,
      trade.entry_date,
      trade.exit_date,
      trade.duration,
      base,
      final,
      riskApplied,
      (trade.entry_percentile ?? 50).toFixed(1),
      (trade.entry_prediction ?? 0).toFixed(6),
      trade.market_return.toFixed(6),
      trade.total_return.toFixed(6),
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
  link.setAttribute('download', `${modelName}_trades_complete.csv`);
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

export default function DataPage() {
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [trades, setTrades] = useState<Trade[]>([]);
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
      const tradesResponse = await getModelTrades(modelName);
      setTrades(tradesResponse.trades || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingDetail(false);
    }
  }

  if (loading) {
    return <div className="text-center py-8 text-[#737373]">Loading...</div>;
  }

  if (!modelsData) {
    return <div className="text-center py-8 text-[#c41e3a]">Failed to load data</div>;
  }

  // Statistics
  const activeTrades = trades.filter(t => Math.abs(t.entry_position) >= 0.5);
  const winningTrades = activeTrades.filter(t => t.total_return > 0);
  const losingTrades = activeTrades.filter(t => t.total_return < 0);
  const riskManagedTrades = trades.filter(t => {
    const base = t.base_position ?? t.entry_position;
    return base !== t.entry_position;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-[#737373]" />
          <h2 className="text-xl font-semibold">Trade Data & Decision Analysis</h2>
        </div>
        {trades.length > 0 && (
          <button
            onClick={() => downloadTradesCSV(trades, selectedModel)}
            className="flex items-center gap-2 px-4 py-2 bg-[#00c853] text-black rounded font-medium hover:bg-[#00c853]/90 transition-colors"
          >
            <Download className="w-4 h-4" />
            Download CSV
          </button>
        )}
      </div>

      {/* Model Selector */}
      <div className="card">
        <div className="flex items-center gap-4">
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
          {trades.length > 0 && (
            <span className="text-sm text-[#737373]">
              {activeTrades.length} active trades | {modelsData.test_period.start} to {modelsData.test_period.end}
            </span>
          )}
        </div>
      </div>

      {/* Statistics */}
      {activeTrades.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="metric-card">
            <div className="metric-label">Total Trades</div>
            <div className="metric-value text-white">{activeTrades.length}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Winning</div>
            <div className="metric-value text-[#00c853]">
              {winningTrades.length}
              <span className="text-sm text-[#525252] ml-1">
                ({((winningTrades.length / activeTrades.length) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Losing</div>
            <div className="metric-value text-[#c41e3a]">
              {losingTrades.length}
              <span className="text-sm text-[#525252] ml-1">
                ({((losingTrades.length / activeTrades.length) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Avg Duration</div>
            <div className="metric-value text-[#a3a3a3]">
              {(activeTrades.reduce((sum, t) => sum + t.duration, 0) / activeTrades.length).toFixed(1)}d
            </div>
          </div>
          <div className="metric-card border-[#f59e0b]/30">
            <div className="metric-label text-[#f59e0b]">Risk Managed</div>
            <div className="metric-value text-[#f59e0b]">
              {riskManagedTrades.length}
              <span className="text-sm text-[#525252] ml-1">
                ({((riskManagedTrades.length / trades.length) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Key Insight Box */}
      <div className="card bg-[#1a1a2e] border-[#2a2a4e]">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-[#6366f1] mt-0.5 flex-shrink-0" />
          <div className="text-sm">
            <p className="text-white font-medium mb-1">Understanding Base vs Final Position</p>
            <p className="text-[#a3a3a3]">
              <strong className="text-white">Base Position</strong>: Determined purely by the model's prediction percentile (quantile strategy).
              <strong className="text-white ml-2">Final Position</strong>: After applying risk management rules (volatility targeting, drawdown control).
              When Base ≠ Final, risk management reduced leverage to protect capital.
            </p>
          </div>
        </div>
      </div>

      {/* Trades Table */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Trade Log - {selectedModel}</h3>

        {loadingDetail ? (
          <div className="text-center py-8 text-[#737373]">Loading trade data...</div>
        ) : trades.length > 0 ? (
          <>
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
              <table className="table-dark">
                <thead className="sticky top-0 bg-[#111111] z-10">
                  <tr>
                    <th className="w-8">#</th>
                    <th className="w-24">Entry</th>
                    <th className="w-24">Exit</th>
                    <th className="w-12 text-center">Days</th>
                    <th className="w-20 text-right">Percentile</th>
                    <th className="w-16 text-center">Base</th>
                    <th className="w-16 text-center">Final</th>
                    <th className="w-24 text-right">Prediction</th>
                    <th className="w-24 text-right">Market Ret</th>
                    <th className="w-24 text-right">Strategy Ret</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((trade, idx) => {
                    const finalPos = trade.entry_position;
                    const basePos = trade.base_position ?? finalPos;
                    const isActive = Math.abs(finalPos) >= 0.5;
                    const percentile = trade.entry_percentile ?? 50;
                    const prediction = trade.entry_prediction ?? 0;
                    const wasReduced = basePos !== finalPos;

                    return (
                      <tr key={idx} className={clsx(!isActive && "opacity-50", wasReduced && "bg-[#f59e0b]/5")}>
                        <td className="text-[#737373]">{idx + 1}</td>
                        <td className="font-mono">{trade.entry_date}</td>
                        <td className="font-mono">{trade.exit_date}</td>
                        <td className="text-center font-mono text-[#a3a3a3]">{trade.duration}</td>
                        <td className={clsx(
                          "text-right font-mono font-semibold",
                          getPercentileColor(percentile)
                        )}>
                          {formatPercentile(percentile)}
                        </td>
                        <td className={clsx(
                          "text-center font-mono",
                          basePos > 0 ? "text-[#00c853]" :
                          basePos < 0 ? "text-[#c41e3a]" : "text-[#737373]"
                        )}>
                          {basePos > 0 ? `+${basePos}` : basePos}x
                        </td>
                        <td className={clsx(
                          "text-center font-mono font-bold",
                          wasReduced && "text-[#f59e0b]",
                          !wasReduced && finalPos > 0 && "text-[#00c853]",
                          !wasReduced && finalPos < 0 && "text-[#c41e3a]",
                          !wasReduced && finalPos === 0 && "text-[#737373]"
                        )}>
                          {wasReduced && "→ "}
                          {finalPos > 0 ? `+${finalPos}` : finalPos}x
                        </td>
                        <td className="text-right font-mono text-[#a3a3a3]">
                          {formatPercent(prediction, 4)}
                        </td>
                        <td className={clsx(
                          "text-right font-mono",
                          trade.market_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          {formatPercent(trade.market_return, 2)}
                        </td>
                        <td className={clsx(
                          "text-right font-mono font-semibold",
                          trade.total_return >= 0 ? "text-[#00c853]" : "text-[#c41e3a]"
                        )}>
                          {formatPercent(trade.total_return, 2)}
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
                Showing {trades.length} trades ({activeTrades.length} with market exposure, {trades.length - activeTrades.length} in cash).
                <span className="text-[#f59e0b] ml-2">
                  {riskManagedTrades.length} trades had position reduced by risk management (highlighted in orange).
                </span>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-8 text-[#737373]">Select a model to view trade data</div>
        )}
      </div>

      {/* ============================================================ */}
      {/* DETAILED EXPLANATION SECTION */}
      {/* ============================================================ */}

      {/* Position Decision Process */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-[#6366f1]" />
          Proceso de Toma de Posiciones (Position Decision Process)
        </h3>

        {/* Step 1: Model Prediction */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-full bg-[#6366f1] flex items-center justify-center text-white font-bold">1</div>
            <h4 className="font-medium text-white">Predicción del Modelo (Model Prediction)</h4>
          </div>
          <div className="ml-10 text-sm text-[#a3a3a3] space-y-2">
            <p>
              El modelo ML genera una <strong className="text-white">predicción de retorno</strong> para el día siguiente.
              Esta predicción es un número que representa el retorno esperado del S&P 500.
            </p>
            <div className="bg-[#1a1a1a] p-3 rounded font-mono text-xs">
              prediction = model.predict(features_t) → e.g., +0.0170%
            </div>
          </div>
        </div>

        {/* Step 2: Percentile Calculation */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-full bg-[#6366f1] flex items-center justify-center text-white font-bold">2</div>
            <h4 className="font-medium text-white">Cálculo del Percentil (Percentile Calculation)</h4>
          </div>
          <div className="ml-10 text-sm text-[#a3a3a3] space-y-2">
            <p>
              La predicción se compara con las <strong className="text-white">últimas 63 predicciones</strong> (ventana rolling de ~3 meses).
              El percentil indica qué tan alta o baja es la predicción actual comparada con el histórico reciente.
            </p>
            <div className="bg-[#1a1a1a] p-3 rounded font-mono text-xs">
              percentile = rank(prediction, window_63) × 100 → e.g., 95%ile
            </div>
            <p className="text-[#737373]">
              <em>¿Por qué percentil y no valor absoluto?</em> Porque normaliza las predicciones entre modelos y
              adapta las decisiones al régimen actual del mercado.
            </p>
          </div>
        </div>

        {/* Step 3: Base Position (Quantile Strategy) */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-full bg-[#6366f1] flex items-center justify-center text-white font-bold">3</div>
            <h4 className="font-medium text-white">Posición Base - Estrategia Quantile (Base Position)</h4>
          </div>
          <div className="ml-10 text-sm space-y-3">
            <p className="text-[#a3a3a3]">
              El percentil se mapea a una <strong className="text-white">posición discreta</strong> usando umbrales asimétricos:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
              <div className="flex items-center gap-2 px-3 py-2 bg-[#00c853]/20 rounded">
                <span className="font-mono font-bold text-[#00c853]">≥90%ile</span>
                <span className="text-[#00c853]">→ +3x UPRO</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-2 bg-[#4caf50]/15 rounded">
                <span className="font-mono font-bold text-[#4caf50]">70-89%ile</span>
                <span className="text-[#4caf50]">→ +1x SPY</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-2 bg-[#737373]/20 rounded">
                <span className="font-mono font-bold text-[#737373]">30-70%ile</span>
                <span className="text-[#737373]">→ CASH</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-2 bg-[#ef5350]/15 rounded">
                <span className="font-mono font-bold text-[#ef5350]">10-29%ile</span>
                <span className="text-[#ef5350]">→ -1x SH</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-2 bg-[#c41e3a]/20 rounded">
                <span className="font-mono font-bold text-[#c41e3a]">≤10%ile</span>
                <span className="text-[#c41e3a]">→ -3x SPXU</span>
              </div>
            </div>
            <div className="bg-[#1a1a1a] p-3 rounded font-mono text-xs">
              if percentile ≥ 90: base_position = +3 (UPRO)<br/>
              elif percentile ≥ 70: base_position = +1 (SPY)<br/>
              elif percentile ≤ 10: base_position = -3 (SPXU)<br/>
              elif percentile ≤ 30: base_position = -1 (SH)<br/>
              else: base_position = 0 (CASH)
            </div>
          </div>
        </div>
      </div>

      {/* Risk Management Section */}
      <div className="card border-[#f59e0b]/30">
        <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-[#f59e0b]" />
          Sistema de Risk Management (Risk Management System)
        </h3>

        <p className="text-sm text-[#a3a3a3] mb-6">
          Después de calcular la posición base, el sistema aplica <strong className="text-white">tres filtros de risk management</strong>
          que pueden <strong className="text-[#f59e0b]">reducir</strong> la posición para proteger el capital.
          La posición <strong>nunca se incrementa</strong>, solo se reduce.
        </p>

        {/* RM 1: Volatility Targeting */}
        <div className="mb-6 p-4 bg-[#1a1a1a] rounded-lg border border-[#333333]">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-5 h-5 text-[#22d3ee]" />
            <h4 className="font-medium text-white">1. Volatility Targeting (Target: 15% anual)</h4>
          </div>
          <div className="text-sm text-[#a3a3a3] space-y-2">
            <p>
              Escala la posición para mantener una <strong className="text-white">volatilidad objetivo del 15% anual</strong>.
              Si la volatilidad reciente del mercado es alta, reduce la posición.
            </p>
            <div className="bg-[#0a0a0a] p-3 rounded font-mono text-xs">
              recent_vol = std(returns_21d) × √252<br/>
              vol_scalar = target_vol / recent_vol<br/>
              vol_scalar = clip(vol_scalar, 0.5, 2.0)<br/>
              adjusted_position = base_position × vol_scalar → round to nearest valid position
            </div>
            <p className="text-[#22d3ee]">
              <strong>Ejemplo:</strong> Si volatilidad reciente = 30% y target = 15%, entonces vol_scalar = 0.5,
              reduciendo +3x → +1x
            </p>
          </div>
        </div>

        {/* RM 2: Position Filter */}
        <div className="mb-6 p-4 bg-[#1a1a1a] rounded-lg border border-[#333333]">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-5 h-5 text-[#a78bfa]" />
            <h4 className="font-medium text-white">2. Position Filter (Anti-Churning)</h4>
          </div>
          <div className="text-sm text-[#a3a3a3] space-y-2">
            <p>
              Evita cambios de posición excesivos (churning) que generarían altos costos de transacción.
              Solo permite cambios si el <strong className="text-white">cambio mínimo es ≥ 1 nivel</strong>.
            </p>
            <div className="bg-[#0a0a0a] p-3 rounded font-mono text-xs">
              if abs(new_position - current_position) &lt; min_change:<br/>
              &nbsp;&nbsp;keep current_position  # No trade
            </div>
            <p className="text-[#a78bfa]">
              <strong>Ejemplo:</strong> Si la posición actual es +1x y la nueva sería +1x (sin cambio), no se opera.
            </p>
          </div>
        </div>

        {/* RM 3: Drawdown Control */}
        <div className="mb-6 p-4 bg-[#1a1a1a] rounded-lg border border-[#f59e0b]/50">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-5 h-5 text-[#f59e0b]" />
            <h4 className="font-medium text-white">3. Drawdown Control (Protección de Capital)</h4>
          </div>
          <div className="text-sm text-[#a3a3a3] space-y-2">
            <p>
              <strong className="text-[#f59e0b]">El más importante.</strong> Cuando el equity cae significativamente desde su máximo (drawdown),
              reduce el leverage para evitar pérdidas catastróficas.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 my-3">
              <div className="p-3 bg-[#0a0a0a] rounded border border-[#f59e0b]/30">
                <div className="font-mono text-[#f59e0b] font-bold">Drawdown 10-15%</div>
                <div className="text-xs mt-1">Reduce +3x → +1x</div>
                <div className="text-xs text-[#737373]">Max leverage: 1x</div>
              </div>
              <div className="p-3 bg-[#0a0a0a] rounded border border-[#ef4444]/50">
                <div className="font-mono text-[#ef4444] font-bold">Drawdown 15-20%</div>
                <div className="text-xs mt-1">Reduce todo a máx ±1x</div>
                <div className="text-xs text-[#737373]">Max leverage: 1x</div>
              </div>
              <div className="p-3 bg-[#0a0a0a] rounded border border-[#c41e3a]/50">
                <div className="font-mono text-[#c41e3a] font-bold">Drawdown &gt;20%</div>
                <div className="text-xs mt-1">Fuerza CASH (0x)</div>
                <div className="text-xs text-[#737373]">Stop-loss total</div>
              </div>
            </div>
            <div className="bg-[#0a0a0a] p-3 rounded font-mono text-xs">
              drawdown = (current_equity - max_equity) / max_equity<br/>
              <br/>
              if drawdown &gt; 20%: final_position = 0  # CASH<br/>
              elif drawdown &gt; 15%: final_position = sign(pos) × min(abs(pos), 1)<br/>
              elif drawdown &gt; 10%: if abs(pos) == 3: final_position = sign(pos) × 1
            </div>
            <p className="text-[#f59e0b]">
              <strong>Ejemplo:</strong> Si equity cayó 12% desde el máximo y la posición base era +3x (UPRO),
              el drawdown control la reduce a +1x (SPY) para proteger capital.
            </p>
          </div>
        </div>

        {/* Final Position */}
        <div className="p-4 bg-[#0f1729] rounded-lg border border-[#1e3a5f]">
          <h4 className="font-medium text-white mb-2">Posición Final (Final Position)</h4>
          <div className="text-sm text-[#a3a3a3]">
            <p>
              La posición final es el resultado de aplicar los tres filtros en secuencia:
            </p>
            <div className="bg-[#0a0a0a] p-3 rounded font-mono text-xs mt-2">
              base_position = quantile_strategy(percentile)<br/>
              → volatility_targeting(base_position, market_vol)<br/>
              → position_filter(previous_position)<br/>
              → drawdown_control(equity_drawdown)<br/>
              = <strong className="text-[#00c853]">final_position</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Return Calculation */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Cálculo del Retorno (Return Calculation)</h3>
        <div className="text-sm text-[#a3a3a3] space-y-3">
          <p>El retorno de la estrategia se calcula como:</p>
          <div className="bg-[#1a1a1a] p-4 rounded font-mono text-sm">
            <span className="text-[#00c853]">strategy_return</span> = risk_free_rate + position × (market_return - risk_free_rate) - costs
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div className="p-3 bg-[#1a1a1a] rounded">
              <div className="font-medium text-white mb-1">Costos incluidos:</div>
              <ul className="text-xs space-y-1">
                <li>• <strong>Expense ratio:</strong> UPRO 0.91%, SPY 0.09%, SPXU 0.89%</li>
                <li>• <strong>Bid-ask spread:</strong> 5-6 bps por operación</li>
                <li>• <strong>Volatility drag:</strong> Para instrumentos 3x leverage</li>
              </ul>
            </div>
            <div className="p-3 bg-[#1a1a1a] rounded">
              <div className="font-medium text-white mb-1">Ejemplo (+3x UPRO):</div>
              <div className="text-xs font-mono">
                market_return = +1%<br/>
                strategy_return ≈ 0% + 3 × 1% - costs<br/>
                strategy_return ≈ +2.9% (después de costos)
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
