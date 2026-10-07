// API Service for Strategy Visualizer - With Caching

import axios from 'axios';
import type {
  ModelsResponse,
  ModelData,
  MarketData,
  RegimeData,
  ModelMetrics,
  SenalActivo,
  HistorialVivo
} from '../types';

const API_BASE = '';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Simple in-memory cache
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function getCached<T>(key: string): T | null {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > CACHE_TTL) {
    cache.delete(key);
    return null;
  }
  return entry.data as T;
}

function setCache(key: string, data: any): void {
  cache.set(key, { data, timestamp: Date.now() });
}

// Models
export async function getModels(): Promise<ModelsResponse> {
  const cacheKey = 'models';
  const cached = getCached<ModelsResponse>(cacheKey);
  if (cached) return cached;

  const response = await api.get('/api/models');
  setCache(cacheKey, response.data);
  return response.data;
}

export async function getModelDetail(modelName: string): Promise<ModelData> {
  const cacheKey = `model:${modelName}`;
  const cached = getCached<ModelData>(cacheKey);
  if (cached) return cached;

  const response = await api.get(`/api/models/${modelName}`);
  setCache(cacheKey, response.data);
  return response.data;
}

export async function getModelMetrics(
  modelName: string,
  startDate?: string,
  endDate?: string
): Promise<{ period: string; metrics: ModelMetrics; full_period_metrics?: ModelMetrics }> {
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);

  const response = await api.get(`/api/models/${modelName}/metrics?${params}`);
  return response.data;
}

export async function getModelTrades(modelName: string): Promise<{ model: string; trades: any[] }> {
  const cacheKey = `trades:${modelName}`;
  const cached = getCached<{ model: string; trades: any[] }>(cacheKey);
  if (cached) return cached;

  const response = await api.get(`/api/models/${modelName}/trades`);
  setCache(cacheKey, response.data);
  return response.data;
}

export interface DailyLogEntry {
  date: string;
  day_num: number;
  prediction: number;
  percentile: number;
  position_base: number;
  position_final: number;
  strategy_return: number;
  market_return: number;
  equity: number;
  drawdown: number;
  high_water_mark: number;
  trading_cost: number;
  regime: string;
  position_changed: boolean;
  direction_correct: boolean;
}

export async function getModelDailyLog(modelName: string): Promise<{
  model: string;
  daily_log: DailyLogEntry[];
  warmup_used: boolean;
}> {
  const cacheKey = `dailylog:${modelName}`;
  const cached = getCached<{ model: string; daily_log: DailyLogEntry[]; warmup_used: boolean }>(cacheKey);
  if (cached) return cached;

  const response = await api.get(`/api/models/${modelName}/daily-log`);
  setCache(cacheKey, response.data);
  return response.data;
}

// Market
export async function getMarketData(): Promise<MarketData> {
  const cacheKey = 'market';
  const cached = getCached<MarketData>(cacheKey);
  if (cached) return cached;

  const response = await api.get('/api/market');
  setCache(cacheKey, response.data);
  return response.data;
}

// Regimes
export async function getRegimes(): Promise<RegimeData> {
  const cacheKey = 'regimes';
  const cached = getCached<RegimeData>(cacheKey);
  if (cached) return cached;

  const response = await api.get('/api/regimes');
  setCache(cacheKey, response.data);
  return response.data;
}

// Clear cache (useful for manual refresh)
export function clearCache(): void {
  cache.clear();
}

export default api;

// Senales de produccion (sin cache: cambian con cada corrida del pipeline)
export async function getSenales(): Promise<{ activos: SenalActivo[] }> {
  const response = await api.get('/api/senales');
  return response.data;
}

export async function getHistorialVivo(activo: string): Promise<HistorialVivo> {
  const response = await api.get(`/api/senales/${activo}/historial`);
  return response.data;
}
