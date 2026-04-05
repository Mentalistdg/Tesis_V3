import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getModels, getModelDetail } from '../services/api';
import type { ModelsResponse, ModelData } from '../types';
import { AlertTriangle, TrendingDown, Shield } from 'lucide-react';
import { calculateSummaryFromTrades, INITIAL_CAPITAL } from '../utils/transactionCosts';
import LoadingScreen from '../components/LoadingScreen';

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);
}

export default function RiskPage() {
  const { modelName } = useParams();
  const navigate = useNavigate();
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadModelsData();
  }, []);

  useEffect(() => {
    if (modelName) {
      loadModelData(modelName);
    }
  }, [modelName]);

  async function loadModelsData() {
    try {
      const data = await getModels();
      setModelsData(data);
      if (!modelName && data.models.length > 0) {
        navigate(`/risk/${data.models[0].model}`);
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
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  // Historical VaR: percentile of actual returns
  function percentile(arr: number[], p: number): number {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * p);
    return sorted[Math.max(0, idx)];
  }

  // Calculate risk metrics using Historical VaR method
  function calculateRiskMetrics() {
    if (!modelData || !modelsData) return null;

    const returns = modelData.strategy_returns;
    const dates = modelData.dates;
    const drawdown = modelData.drawdown;

    // Get net return from trades (consistent with Overview page)
    const tradeSummary = modelData.trades && modelData.trades.length > 0
      ? calculateSummaryFromTrades(modelData.trades, INITIAL_CAPITAL)
      : null;
    const netReturn = tradeSummary?.netReturn ?? modelData.metrics?.total_return ?? 0;

    // Annualize using same formula as Overview: CAGR = (1 + total)^(1/years) - 1
    const years = modelsData.test_period.n_days / 252;
    const annualReturn = Math.pow(1 + netReturn, 1 / years) - 1;

    // Daily statistics (full period)
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
    const stdDaily = Math.sqrt(variance);

    // Historical VaR: use last N days of actual returns
    const last30Returns = returns.slice(-30);
    const last30Dates = dates.slice(-30);
    const last252Returns = returns.slice(-252);
    const last252Dates = dates.slice(-252);

    // VaR 95% = percentile 5 of returns (the 5th percentile worst return)
    const var95_30d = percentile(last30Returns, 0.05);
    const var99_30d = percentile(last30Returns, 0.01);
    const var95_1y = percentile(last252Returns, 0.05);
    const var99_1y = percentile(last252Returns, 0.01);

    // Get critical days (worst returns) for 30 days
    const critical30Days = last30Returns
      .map((ret, idx) => ({ date: last30Dates[idx], return: ret }))
      .sort((a, b) => a.return - b.return)
      .slice(0, Math.max(3, Math.ceil(last30Returns.length * 0.1))); // Top 10% worst or at least 3

    // Get critical days (worst returns) for 252 days
    const critical252Days = last252Returns
      .map((ret, idx) => ({ date: last252Dates[idx], return: ret }))
      .sort((a, b) => a.return - b.return)
      .slice(0, Math.max(5, Math.ceil(last252Returns.length * 0.05))); // Top 5% worst or at least 5

    // CVaR (Expected Shortfall) = average of returns below VaR threshold
    // For 30 days
    const sorted30 = [...last30Returns].sort((a, b) => a - b);
    const varIdx30 = Math.max(1, Math.floor(last30Returns.length * 0.05));
    const cvar95_30d = sorted30.slice(0, varIdx30).reduce((a, b) => a + b, 0) / varIdx30;

    // For 252 days
    const sorted252 = [...last252Returns].sort((a, b) => a - b);
    const varIdx252 = Math.max(1, Math.floor(last252Returns.length * 0.05));
    const cvar95_1y = sorted252.slice(0, varIdx252).reduce((a, b) => a + b, 0) / varIdx252;

    // Daily VaR for reference (using full period)
    const sortedAll = [...returns].sort((a, b) => a - b);
    const varIdxAll = Math.floor(returns.length * 0.05);
    const var95Daily = sortedAll[varIdxAll];
    const var99Daily = sortedAll[Math.floor(returns.length * 0.01)];
    const cvar95Daily = sortedAll.slice(0, varIdxAll + 1).reduce((a, b) => a + b, 0) / (varIdxAll + 1);

    // Annual volatility
    const annualVol = stdDaily * Math.sqrt(252);

    // Downside volatility
    const negativeReturns = returns.filter(r => r < 0);
    const downsideVar = negativeReturns.length > 0
      ? negativeReturns.reduce((a, b) => a + Math.pow(b, 2), 0) / negativeReturns.length
      : 0;
    const downsideVol = Math.sqrt(downsideVar) * Math.sqrt(252);

    // Sortino (uses annualReturn calculated above from net return)
    const sortino = downsideVol > 0 ? annualReturn / downsideVol : 0;

    // Calmar
    const maxDD = Math.min(...drawdown);
    const calmar = maxDD !== 0 ? annualReturn / Math.abs(maxDD) : 0;

    // Find drawdown periods
    const ddPeriods: { start: string; end: string; depth: number; duration: number }[] = [];
    let inDrawdown = false;
    let ddStart = '';
    let currentDDMax = 0;

    for (let i = 0; i < drawdown.length; i++) {
      if (drawdown[i] < -0.05 && !inDrawdown) {
        inDrawdown = true;
        ddStart = modelData.dates[i];
        currentDDMax = drawdown[i];
      } else if (inDrawdown) {
        currentDDMax = Math.min(currentDDMax, drawdown[i]);
        if (drawdown[i] >= -0.01) {
          inDrawdown = false;
          ddPeriods.push({
            start: ddStart,
            end: modelData.dates[i],
            depth: currentDDMax,
            duration: i - modelData.dates.indexOf(ddStart)
          });
        }
      }
    }

    // Worst daily return as reference
    const worstDay = Math.min(...returns);
    const worstDay252 = Math.min(...last252Returns);
    const bestDay = Math.max(...returns);

    return {
      var95Daily,
      var99Daily,
      cvar95Daily,
      var95_30d,
      var99_30d,
      cvar95_30d,
      var95_1y,
      var99_1y,
      cvar95_1y,
      worstDay,
      worstDay252,
      bestDay,
      stdDaily,
      annualVol,
      downsideVol,
      sortino,
      calmar,
      maxDD,
      ddPeriods: ddPeriods.sort((a, b) => a.depth - b.depth).slice(0, 5),
      critical30Days,
      critical252Days,
      nDays30: last30Returns.length,
      nDays252: last252Returns.length
    };
  }

  const riskMetrics = calculateRiskMetrics();

  if (loading) {
    return <LoadingScreen />;
  }

  if (!modelData || !riskMetrics) {
    return <div className="text-center py-8 text-[#c41e3a]">Failed to load data</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Risk Analysis</h2>
          <span className="text-sm text-[#737373]">{modelName}</span>
        </div>

        <select
          value={modelName || ''}
          onChange={(e) => navigate(`/risk/${e.target.value}`)}
          className="bg-[#1a1a1a] border border-[#222222] rounded px-3 py-2 text-sm"
        >
          {modelsData?.models.map(m => (
            <option key={m.model} value={m.model}>{m.model}</option>
          ))}
        </select>
      </div>

      {/* Nota sobre período de análisis */}
      <div className="p-3 bg-[#1a1a1a] border border-[#333] rounded text-sm text-[#737373]">
        <strong className="text-[#d1d4dc]">Período de Análisis:</strong> Out-of-Sample Testing ({modelsData?.test_period.start} - {modelsData?.test_period.end}) · {modelsData?.test_period.n_days} días de trading
      </div>

      {/* Risk-Adjusted Metrics */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Shield className="w-5 h-5 text-[#00c853]" />
          <h3 className="text-lg font-medium">Risk-Adjusted Metrics</h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-[#1a1a1a] rounded p-4 text-center">
            <div className="text-xs text-[#737373] uppercase mb-2">Sharpe</div>
            <div className="text-2xl font-bold">{(modelData.metrics?.sharpe ?? 0).toFixed(3)}</div>
          </div>
          <div className="bg-[#1a1a1a] rounded p-4 text-center">
            <div className="text-xs text-[#737373] uppercase mb-2">Sortino</div>
            <div className="text-2xl font-bold">{riskMetrics.sortino.toFixed(3)}</div>
          </div>
          <div className="bg-[#1a1a1a] rounded p-4 text-center">
            <div className="text-xs text-[#737373] uppercase mb-2">Calmar</div>
            <div className="text-2xl font-bold">{riskMetrics.calmar.toFixed(3)}</div>
          </div>
          <div className="bg-[#1a1a1a] rounded p-4 text-center">
            <div className="text-xs text-[#737373] uppercase mb-2">Info Ratio</div>
            <div className="text-2xl font-bold">-</div>
          </div>
        </div>
      </div>

      {/* VaR Section - Focus on 30 days and Annual */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Monthly VaR - Historical */}
        <div className="card border-l-4 border-[#f59e0b]">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-[#f59e0b]" />
            <h3 className="text-lg font-medium">VaR Mensual (Últimos 30 días)</h3>
            <span className="text-xs bg-[#f59e0b]/20 text-[#f59e0b] px-2 py-0.5 rounded">Histórico</span>
          </div>

          {(() => {
            const currentEquity = modelData.equity_curve[modelData.equity_curve.length - 1] * (modelsData?.config.initial_capital || 10000);
            const var95Dollar = Math.abs(riskMetrics.var95_30d) * currentEquity;
            const cvar95Dollar = Math.abs(riskMetrics.cvar95_30d) * currentEquity;

            return (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-[#1a1a1a] rounded p-3 text-center">
                    <div className="text-xs text-[#737373] uppercase mb-1">VaR 95%</div>
                    <div className="text-2xl font-bold text-[#c41e3a]">
                      {(riskMetrics.var95_30d * 100).toFixed(2)}%
                    </div>
                    <div className="text-xs text-[#737373] mt-1">{formatCurrency(var95Dollar)}</div>
                  </div>
                  <div className="bg-[#1a1a1a] rounded p-3 text-center">
                    <div className="text-xs text-[#737373] uppercase mb-1">CVaR 95%</div>
                    <div className="text-2xl font-bold text-[#f59e0b]">
                      {(riskMetrics.cvar95_30d * 100).toFixed(2)}%
                    </div>
                    <div className="text-xs text-[#737373] mt-1">{formatCurrency(cvar95Dollar)}</div>
                  </div>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#737373]">VaR 99% (últimos 30 días)</span>
                  <span className="font-mono text-[#c41e3a]">{(riskMetrics.var99_30d * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#737373]">Volatilidad diaria (σ)</span>
                  <span className="font-mono text-[#737373]">{(riskMetrics.stdDaily * 100).toFixed(2)}%</span>
                </div>

                {/* Critical Days Table */}
                <div className="mt-4">
                  <div className="text-xs text-[#737373] uppercase mb-2">Días Críticos ({riskMetrics.nDays30} días analizados)</div>
                  <div className="overflow-x-auto max-h-32 overflow-y-auto">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-[#1a1a1a]">
                        <tr className="text-[#737373]">
                          <th className="text-left py-1 px-2">Fecha</th>
                          <th className="text-right py-1 px-2">Retorno</th>
                          <th className="text-right py-1 px-2">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {riskMetrics.critical30Days.map((day, idx) => (
                          <tr key={idx} className="border-t border-[#222]">
                            <td className="py-1 px-2 text-[#a3a3a3]">{day.date}</td>
                            <td className="py-1 px-2 text-right font-mono text-[#c41e3a]">
                              {(day.return * 100).toFixed(2)}%
                            </td>
                            <td className="py-1 px-2 text-right font-mono text-[#c41e3a]">
                              {formatCurrency(day.return * currentEquity)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="p-2 bg-[#0a0a0a] rounded text-xs text-[#737373]">
                  VaR Histórico = Percentil 5 de retornos diarios del último mes. CVaR = promedio de los peores retornos (debajo del percentil 5).
                </div>
              </div>
            );
          })()}
        </div>

        {/* Annual VaR - Historical */}
        <div className="card border-l-4 border-[#ef4444]">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-[#ef4444]" />
            <h3 className="text-lg font-medium">VaR Anual (Últimos 252 días)</h3>
            <span className="text-xs bg-[#ef4444]/20 text-[#ef4444] px-2 py-0.5 rounded">Histórico</span>
          </div>

          {(() => {
            const currentEquity = modelData.equity_curve[modelData.equity_curve.length - 1] * (modelsData?.config.initial_capital || 10000);
            const var95Dollar = Math.abs(riskMetrics.var95_1y) * currentEquity;
            const cvar95Dollar = Math.abs(riskMetrics.cvar95_1y) * currentEquity;

            return (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-[#1a1a1a] rounded p-3 text-center">
                    <div className="text-xs text-[#737373] uppercase mb-1">VaR 95%</div>
                    <div className="text-2xl font-bold text-[#ef4444]">
                      {(riskMetrics.var95_1y * 100).toFixed(2)}%
                    </div>
                    <div className="text-xs text-[#737373] mt-1">{formatCurrency(var95Dollar)}</div>
                  </div>
                  <div className="bg-[#1a1a1a] rounded p-3 text-center">
                    <div className="text-xs text-[#737373] uppercase mb-1">CVaR 95%</div>
                    <div className="text-2xl font-bold text-[#ef4444]">
                      {(riskMetrics.cvar95_1y * 100).toFixed(2)}%
                    </div>
                    <div className="text-xs text-[#737373] mt-1">{formatCurrency(cvar95Dollar)}</div>
                  </div>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#737373]">VaR 99% (últimos 252 días)</span>
                  <span className="font-mono text-[#ef4444]">{(riskMetrics.var99_1y * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[#737373]">Peor día (252 días)</span>
                  <span className="font-mono text-[#c41e3a]">{(riskMetrics.worstDay252 * 100).toFixed(2)}%</span>
                </div>

                {/* Critical Days Table */}
                <div className="mt-4">
                  <div className="text-xs text-[#737373] uppercase mb-2">Días Críticos ({riskMetrics.nDays252} días analizados)</div>
                  <div className="overflow-x-auto max-h-40 overflow-y-auto">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-[#1a1a1a]">
                        <tr className="text-[#737373]">
                          <th className="text-left py-1 px-2">Fecha</th>
                          <th className="text-right py-1 px-2">Retorno</th>
                          <th className="text-right py-1 px-2">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {riskMetrics.critical252Days.map((day, idx) => (
                          <tr key={idx} className="border-t border-[#222]">
                            <td className="py-1 px-2 text-[#a3a3a3]">{day.date}</td>
                            <td className="py-1 px-2 text-right font-mono text-[#ef4444]">
                              {(day.return * 100).toFixed(2)}%
                            </td>
                            <td className="py-1 px-2 text-right font-mono text-[#ef4444]">
                              {formatCurrency(day.return * currentEquity)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="p-2 bg-[#0a0a0a] rounded text-xs text-[#737373]">
                  VaR Histórico = Percentil 5 de retornos diarios del último año. CVaR = promedio de los peores retornos (debajo del percentil 5).
                </div>
              </div>
            );
          })()}
        </div>

      </div>

      {/* Drawdown Analysis */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <TrendingDown className="w-5 h-5 text-[#c41e3a]" />
          <h3 className="text-lg font-medium">Drawdown Analysis</h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-[#1a1a1a] rounded p-3">
            <div className="text-xs text-[#737373] uppercase">Max Drawdown</div>
            <div className="text-xl font-semibold text-[#c41e3a] mt-1">
              {(riskMetrics.maxDD * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-[#1a1a1a] rounded p-3">
            <div className="text-xs text-[#737373] uppercase">Calmar Ratio</div>
            <div className="text-xl font-semibold mt-1">
              {riskMetrics.calmar.toFixed(2)}
            </div>
          </div>
        </div>

        <h4 className="font-medium mb-3">Major Drawdown Periods</h4>
        <table className="table-dark">
          <thead>
            <tr>
              <th className="w-8">#</th>
              <th className="w-28">Start</th>
              <th className="w-28">End</th>
              <th className="w-24 text-right">Duration</th>
              <th className="w-20 text-right">Depth</th>
            </tr>
          </thead>
          <tbody>
            {riskMetrics.ddPeriods.map((period, idx) => (
              <tr key={idx}>
                <td className="text-[#737373]">{idx + 1}</td>
                <td>{period.start}</td>
                <td>{period.end}</td>
                <td className="text-right font-mono">{period.duration} days</td>
                <td className="text-right font-mono text-[#c41e3a]">
                  {(period.depth * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
}
