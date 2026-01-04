// Types for the Strategy Visualizer App - Optimized Version

export interface OptimalParams {
  q_long_extreme: number;
  q_long_moderate: number;
  q_short_extreme: number;
  q_short_moderate: number;
}

export interface ModelMetrics {
  total_return: number;
  annual_return: number;
  market_return: number;
  excess_return: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
  win_rate: number;
  dir_accuracy: number;
  mean_position: number;
  n_trades: number;
  n_days: number;
  // Bootstrap confidence intervals
  sharpe_ci_lower: number;
  sharpe_ci_upper: number;
  prob_sharpe_positive: number;
  // Transaction costs
  transaction_costs: number;
  // Position distribution
  pct_long: number;
  pct_short: number;
  pct_3x_long: number;
  pct_3x_short: number;
  pct_cash: number;
  // Original vs optimized
  orig_return: number;
  orig_sharpe: number;
  return_improvement: number;
  sharpe_improvement: number;
}

export interface ModelSummary {
  model: string;
  category: string;
  total_return: number;
  annual_return: number;
  market_return: number;
  excess_return: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
  win_rate: number;
  mean_position: number;
  n_trades: number;
  // Bootstrap
  sharpe_ci_lower: number;
  sharpe_ci_upper: number;
  prob_sharpe_positive: number;
  // Costs
  transaction_costs: number;
  // Position distribution
  pct_long: number;
  pct_short: number;
  pct_3x_long: number;
  pct_3x_short: number;
  pct_cash: number;
  // Optimal params
  optimal_params: OptimalParams;
  final_equity: number;
  profit_loss: number;
  // Improvement
  return_improvement: number;
  sharpe_improvement: number;
}

export interface Trade {
  entry_idx: number;
  entry_date: string;
  entry_position: number;     // Final position AFTER risk management
  exit_idx: number;
  exit_date: string;
  exit_position: number;
  duration: number;
  total_return: number;       // Strategy return (position * market_return)
  market_return: number;      // Raw market return (SPY)
  entry_prediction?: number;  // Raw model prediction at entry
  entry_percentile?: number;  // Percentile of prediction in 63-day rolling window (0-100)
  base_position?: number;     // Position BEFORE risk management (from quantile strategy)
  avg_position?: number;      // Optional - average position during trade
}

export interface ModelData {
  category: string;
  dates: string[];
  predictions: number[];
  positions: number[];
  strategy_returns: number[];
  market_returns: number[];
  equity_curve: number[];
  market_equity: number[];
  drawdown: number[];
  regimes: string[];
  trades: Trade[];
  metrics: ModelMetrics;
  optimal_params: OptimalParams;
}

export interface MarketData {
  dates: string[];
  returns: number[];
  equity_curve: number[];
  drawdown: number[];
  regimes: string[];
  metrics: {
    total_return: number;
    annual_return: number;
    sharpe: number;
    max_drawdown: number;
    volatility: number;
    final_value: number;
  };
}

export interface Signal {
  model: string;
  category: string;
  position: number;
  signal: 'STRONG_LONG' | 'LONG' | 'WEAK_LONG' | 'CASH' | 'SHORT' | 'STRONG_SHORT';
  change: number;
  date: string;
}

export interface SignalsData {
  date: string;
  consensus: {
    position: number;
    bullish_count: number;
    neutral_count: number;
    bearish_count: number;
    total_models: number;
  };
  signals: Signal[];
}

export interface RegimeData {
  current_regime: string;
  regime_counts: Record<string, number>;
  regime_performance: Record<string, Record<string, { total_return: number; n_days: number }>>;
}

export interface BestModel {
  name: string;
  method: string;
  params: OptimalParams;
  final_value: number;
  profit_loss: number;
  sharpe: number;
  sharpe_ci: [number, number];
  prob_sharpe_positive: number;
}

export interface ModelsResponse {
  timestamp: string;
  config: {
    initial_capital: number;
    volatility_target: number;
    n_bootstrap: number;
    confidence_level: number;
  };
  test_period: {
    start: string;
    end: string;
    n_days: number;
  };
  benchmark: {
    spy_final_value: number;
    spy_total_return: number;
  };
  best_model: BestModel;
  models_beating_benchmark: number;
  total_models: number;
  models: ModelSummary[];
}

export interface DateRange {
  start: string;
  end: string;
}

export type TabType =
  | 'signals'
  | 'overview'
  | 'compare'
  | 'detail'
  | 'trades'
  | 'risk'
  | 'regime'
  | 'costs'
  | 'data';
