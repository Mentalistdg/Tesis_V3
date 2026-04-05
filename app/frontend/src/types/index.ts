// Types for the CRONOS Strategy Visualizer App — LONG-ONLY strategy

export interface OptimalParams {
  q_3x?: number;  // Top X% for 3x (UPRO)
  q_1x?: number;  // Top X% for 1x (SPY)
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
  // Position distribution (LONG-ONLY: 3x/1x/cash)
  pct_long: number;
  pct_3x: number;
  pct_1x: number;
  pct_cash: number;
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
  win_rate?: number;
  mean_position?: number;
  n_trades: number;
  // Bootstrap (optional — not always provided by API)
  sharpe_ci_lower?: number;
  sharpe_ci_upper?: number;
  prob_sharpe_positive?: number;
  // Costs
  transaction_costs?: number;
  // Position distribution (LONG-ONLY)
  pct_long: number;
  pct_3x?: number;
  pct_1x?: number;
  pct_cash: number;
  // Optimal params (optional)
  optimal_params?: OptimalParams;
  final_equity?: number;
  profit_loss?: number;
}

export interface Trade {
  entry_idx: number;
  entry_date: string;
  entry_position: number;
  exit_idx: number;
  exit_date: string;
  exit_position: number;
  duration: number;
  total_return: number;       // Strategy return NET (after costs)
  market_return: number;      // Raw market return (SPY)
  entry_prediction?: number;  // Raw model prediction at entry
  entry_percentile?: number;  // Percentile of prediction in 63-day rolling window (0-100)
  avg_position?: number;
  tx_cost?: number;           // Transaction cost of the trade (total)
  costs?: {                   // Cost breakdown
    expense: number;          // Expense ratio cost
    trading: number;          // Trading (bid-ask) cost
    vol_drag: number;         // Volatility drag cost
  };
  cumulative?: number;        // Cumulative equity at end of trade (multiplier, 1.0 = $10K)
  entry_equity?: number;      // Actual equity at entry
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
  optimal_params?: OptimalParams;
}

export interface MarketData {
  dates: string[];
  returns: number[];
  equity_curve: number[];
  drawdown: number[];
}

export interface RegimeData {
  current_regime: string;
  regime_counts: Record<string, number>;
  regime_performance: Record<string, Record<string, { total_return: number; n_days: number }>>;
}

export interface ModelsResponse {
  timestamp: string;
  strategy?: string;  // 'LONG-ONLY'
  config: {
    initial_capital: number;
  };
  test_period: {
    start: string;
    end: string;
    n_days: number;
  };
  benchmark: {
    spy_final_value?: number;
    total_return?: number;
    sharpe?: number;
    max_drawdown?: number;
    final_value?: number;
  };
  models_beating_spy?: number;
  total_models: number;
  models: ModelSummary[];
}

export interface DateRange {
  start: string;
  end: string;
}

export type TabType =
  | 'overview'
  | 'detail'
  | 'trades'
  | 'risk'
  | 'regime'
  | 'costs';
