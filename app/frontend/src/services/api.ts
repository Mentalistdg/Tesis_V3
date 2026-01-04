// API Service for Strategy Visualizer

import axios from 'axios';
import type {
  ModelsResponse,
  ModelData,
  MarketData,
  SignalsData,
  RegimeData,
  ModelMetrics
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Models
export async function getModels(): Promise<ModelsResponse> {
  const response = await api.get('/api/models');
  return response.data;
}

export async function getModelDetail(modelName: string): Promise<ModelData> {
  const response = await api.get(`/api/models/${modelName}`);
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
  const response = await api.get(`/api/models/${modelName}/trades`);
  return response.data;
}

// Market
export async function getMarketData(): Promise<MarketData> {
  const response = await api.get('/api/market');
  return response.data;
}

// Signals
export async function getSignals(): Promise<SignalsData> {
  const response = await api.get('/api/signals');
  return response.data;
}

// Regimes
export async function getRegimes(): Promise<RegimeData> {
  const response = await api.get('/api/regimes');
  return response.data;
}

// Compare
export async function compareModels(
  models: string[],
  startDate?: string,
  endDate?: string
): Promise<{ models: any[] }> {
  const params = new URLSearchParams();
  params.append('models', models.join(','));
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);

  const response = await api.get(`/api/compare?${params}`);
  return response.data;
}

export default api;
