import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi } from 'lightweight-charts';
import { AlertTriangle, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';
import { clsx } from 'clsx';
import { getHistorialVivo, getSenales } from '../services/api';
import type { HistorialVivo, Senal, SenalActivo, EscenarioReporte } from '../types';
import LoadingScreen from '../components/LoadingScreen';

const ETIQUETA: Record<Senal, string> = { CASH: 'CASH', SPY: 'SPY 1x', UPRO: 'UPRO 3x' };
const COLOR: Record<Senal, string> = { CASH: '#737373', SPY: '#00c853', UPRO: '#c41e3a' };

const pct = (v?: number, dec = 1) => (v === undefined || v === null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(dec)}%`);

function EstadoDatos({ estado, advertencias }: { estado: SenalActivo['estado']; advertencias: string[] }) {
  const cfg = {
    verde: { Icon: CheckCircle2, color: 'text-[#00c853]', texto: 'Datos completos' },
    amarillo: { Icon: AlertTriangle, color: 'text-yellow-400', texto: 'Datos con advertencias' },
    rojo: { Icon: XCircle, color: 'text-[#c41e3a]', texto: 'Datos incompletos: no operar' },
  }[estado];
  return (
    <details className="mt-4 text-sm">
      <summary className={clsx('flex items-center gap-2 cursor-pointer', cfg.color)}>
        <cfg.Icon className="w-4 h-4" /> {cfg.texto}
        {advertencias.length > 0 && <span className="text-[#525252]">({advertencias.length})</span>}
      </summary>
      <ul className="mt-2 space-y-1 text-[#737373] list-disc pl-6">
        {advertencias.map((a, i) => <li key={i}>{a}</li>)}
      </ul>
    </details>
  );
}

function Escenario({ titulo, e }: { titulo: string; e: EscenarioReporte }) {
  if (!e?.dias) return <div className="text-[#525252] text-sm">{titulo}: sin días realizados aún</div>;
  return (
    <div className="bg-black border border-[#222222] rounded p-3">
      <div className="text-xs text-[#737373] mb-1">{titulo}</div>
      <div className="text-lg font-semibold text-white">{pct(e.total_return)}</div>
      <div className="text-xs text-[#737373]">
        SPY {pct(e.spy_total_return)} · Sharpe {e.sharpe?.toFixed(2)} · maxDD {pct(-Math.abs(e.max_drawdown ?? 0))}
      </div>
      <div className="text-xs text-[#737373]">
        {e.entradas} entradas · UPRO/SPY/CASH {e.pct_3x?.toFixed(0)}/{e.pct_1x?.toFixed(0)}/{e.pct_cash?.toFixed(0)}%
      </div>
    </div>
  );
}

function GraficoVivo({ historial }: { historial: HistorialVivo }) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  useEffect(() => {
    if (!ref.current || !historial.B) return;
    chartRef.current?.remove();
    const chart = createChart(ref.current, {
      width: ref.current.clientWidth,
      height: 320,
      layout: { background: { color: '#111111' }, textColor: '#737373' },
      grid: { vertLines: { color: '#1a1a1a' }, horzLines: { color: '#1a1a1a' } },
      rightPriceScale: { borderColor: '#222222' },
      timeScale: { borderColor: '#222222' },
      localization: { priceFormatter: (p: number) => '$' + p.toLocaleString('en-US', { maximumFractionDigits: 0 }) },
    });
    chartRef.current = chart;
    const serie = (d: HistorialVivo['B'], key: 'equity_curve' | 'market_equity', color: string, title: string, lineStyle = 0) => {
      if (!d) return;
      const s = chart.addLineSeries({ color, lineWidth: 2, title, lineStyle });
      s.setData(d.dates.map((t, i) => ({ time: t, value: d[key][i] * 10000 })) as any);
    };
    serie(historial.B, 'equity_curve', '#c41e3a', 'B (operativa)');
    serie(historial.A, 'equity_curve', '#f59e0b', 'A (diagnóstico)', 1);
    serie(historial.B, 'market_equity', '#525252', 'SPY', 2);
    chart.timeScale().fitContent();
    const onResize = () => ref.current && chart.applyOptions({ width: ref.current.clientWidth });
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); chart.remove(); chartRef.current = null; };
  }, [historial]);
  return <div ref={ref} />;
}

function TarjetaActivo({ s }: { s: SenalActivo }) {
  const [historial, setHistorial] = useState<HistorialVivo | null>(null);
  useEffect(() => { getHistorialVivo(s.activo).then(setHistorial).catch(() => setHistorial({})); }, [s.activo]);
  const rep = s.reporte;
  return (
    <section className="bg-[#111111] border border-[#222222] rounded-lg p-6 space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-6">
        <div>
          <div className="text-sm text-[#737373]">{s.nombre}</div>
          <div className="text-xs text-[#525252]">Señal para el cierre del <span className="text-white">{s.fecha}</span> · orden MOC antes de 15:50 NY</div>
          <div className="mt-3 flex items-center gap-4">
            <span className="text-5xl font-bold" style={{ color: COLOR[s.senal_b] }}>{ETIQUETA[s.senal_b]}</span>
            {s.cambio_vs_ayer && (
              <span className="px-2 py-1 text-xs font-semibold rounded bg-[#c41e3a] text-white">
                CAMBIÓ (antes {s.senal_anterior_b && ETIQUETA[s.senal_anterior_b]})
              </span>
            )}
          </div>
          {!s.senal_valida && <div className="mt-2 text-xs text-yellow-400">Señal degenerada (baja dispersión): CASH forzado</div>}
          <EstadoDatos estado={s.estado} advertencias={s.advertencias} />
        </div>
        <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
          <span className="text-[#737373]">Predicción</span><span className="text-white text-right">{pct(s.prediccion, 3)}</span>
          <span className="text-[#737373]">Percentil 63d</span><span className="text-white text-right">{s.percentil.toFixed(1)}</span>
          <span className="text-[#737373]">Umbral UPRO</span><span className="text-white text-right">≥ {s.umbrales.UPRO}</span>
          <span className="text-[#737373]">Umbral SPY</span><span className="text-white text-right">≥ {s.umbrales.SPY}</span>
          <span className="text-[#737373]">Versión A</span><span className="text-right" style={{ color: COLOR[s.senal_a] }}>{ETIQUETA[s.senal_a]}</span>
          <span className="text-[#737373]">Actualizado</span><span className="text-[#525252] text-right">{s.actualizado.replace('T', ' ')}</span>
        </div>
      </div>

      <div>
        <h3 className="text-sm text-[#737373] mb-2">Período en vivo desde {rep.desde} (capital $10.000)</h3>
        {historial ? <GraficoVivo historial={historial} /> : <div className="text-[#525252] text-sm">Cargando…</div>}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mt-3">
          <Escenario titulo="B · ejecución según la tesis" e={rep.B.ejecucion_tesis} />
          <Escenario titulo="B · con 1 día de retraso" e={rep.B.retraso_1_dia} />
          <Escenario titulo="A · ejecución según la tesis" e={rep.A.ejecucion_tesis} />
          <Escenario titulo="A · con 1 día de retraso" e={rep.A.retraso_1_dia} />
        </div>
        <p className="mt-2 text-xs text-[#525252]">{rep.nota} La versión B (macro en fecha de publicación) es la operativa.</p>
      </div>

      <div>
        <h3 className="text-sm text-[#737373] mb-2">Últimas señales</h3>
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-sm">
            <thead className="text-[#737373] text-xs sticky top-0 bg-[#111111]">
              <tr><th className="text-left py-2">Fecha</th><th className="text-left">B</th><th className="text-left">A</th><th className="text-right">Predicción</th><th className="text-right">Percentil</th></tr>
            </thead>
            <tbody>
              {[...s.historial].reverse().map((h, i, arr) => {
                const cambio = i < arr.length - 1 && arr[i + 1].senal_b !== h.senal_b;
                return (
                  <tr key={h.date} className={clsx('border-t border-[#1a1a1a]', cambio && 'bg-[#1a0a0d]')}>
                    <td className="py-1.5 text-white">{h.date}</td>
                    <td style={{ color: COLOR[h.senal_b] }}>{ETIQUETA[h.senal_b]}</td>
                    <td style={{ color: COLOR[h.senal_a] }}>{ETIQUETA[h.senal_a]}</td>
                    <td className="text-right text-[#a3a3a3]">{pct(h.prediccion, 3)}</td>
                    <td className="text-right text-[#a3a3a3]">{h.percentil.toFixed(1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default function SenalesPage() {
  const [datos, setDatos] = useState<SenalActivo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cargar = () => {
    setError(null);
    getSenales().then((r) => setDatos(r.activos)).catch((e) => setError(e?.response?.data?.detail ?? 'No se pudo cargar las señales'));
  };
  useEffect(() => {
    cargar();
    const t = setInterval(cargar, 5 * 60 * 1000);
    return () => clearInterval(t);
  }, []);
  if (error) return <div className="text-[#c41e3a] p-6">{error}</div>;
  if (!datos) return <LoadingScreen />;
  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-white">Señales de producción</h1>
        <button onClick={cargar} className="flex items-center gap-2 text-sm text-[#737373] hover:text-white">
          <RefreshCw className="w-4 h-4" /> Recargar
        </button>
      </div>
      {datos.map((s) => <TarjetaActivo key={s.activo} s={s} />)}
    </div>
  );
}
