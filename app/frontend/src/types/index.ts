// Types for the Strategy Visualizer App - Optimized Version

// For LONG-ONLY strategy, only q_3x and q_1x are used
// Legacy fields kept for backward compatibility
export interface OptimalParams {
  // LONG-ONLY params
  q_3x?: number;  // Top X% for 3x (UPRO)
  q_1x?: number;  // Top X% for 1x (SPY)
  // Legacy params (backward compatibility)
  q_long_extreme?: number;
  q_long_moderate?: number;
  q_short_extreme?: number;
  q_short_moderate?: number;
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
  win_rate?: number;
  mean_position?: number;
  n_trades: number;
  // Bootstrap (optional - not always provided by API)
  sharpe_ci_lower?: number;
  sharpe_ci_upper?: number;
  prob_sharpe_positive?: number;
  // Costs
  transaction_costs?: number;
  // Position distribution
  pct_long: number;
  pct_short?: number;
  pct_3x_long?: number;
  pct_3x_short?: number;
  pct_cash: number;
  // LONG-ONLY fields (from API)
  pct_3x?: number;  // Same as pct_3x_long for UPRO
  pct_1x?: number;  // SPY percentage
  // Optimal params (optional - not always provided)
  optimal_params?: OptimalParams;
  final_equity?: number;
  profit_loss?: number;
  // Improvement
  return_improvement?: number;
  sharpe_improvement?: number;
}

export interface Trade {
  entry_idx: number;
  entry_date: string;
  entry_position: number;     // Final position AFTER risk management
  exit_idx: number;
  exit_date: string;
  exit_position: number;
  duration: number;
  total_return: number;       // Strategy return NET (after costs)
  gross_return?: number;      // Strategy return GROSS (before costs)
  market_return: number;      // Raw market return (SPY)
  entry_prediction?: number;  // Raw model prediction at entry
  entry_percentile?: number;  // Percentile of prediction in 63-day rolling window (0-100)
  base_position?: number;     // Position BEFORE risk management (from quantile strategy)
  position_before_dd?: number; // Position AFTER vol targeting, BEFORE drawdown control
  reduced_by?: string;        // What reduced the position: "DD_CTRL", "VOL_TGT", or "-"
  avg_position?: number;      // Optional - average position during trade
  tx_cost?: number;           // Transaction cost of the trade (total)
  costs?: {                   // Cost breakdown
    expense: number;          // Expense ratio cost
    trading: number;          // Trading (bid-ask) cost
    vol_drag: number;         // Volatility drag cost
  };
  net_gained?: number;        // Net gain after costs
  cumulative?: number;        // Cumulative equity at end of trade (multiplier, 1.0 = $10K)

  // ============================================================
  // DUAL CURVE SYSTEM - Decision vs Final curves
  // ============================================================

  // DECISION CURVE (cumulative_temp - what the system SAW when deciding)
  entry_dd_decision?: number;      // DD that triggered DD_CTRL (>= 10% when DD_CTRL acted)
  entry_max_eq_decision?: number;  // HWM of decision curve (consistent with entry_dd_decision)

  // FINAL CURVE (equity_curve - actual protected portfolio results)
  entry_equity?: number;           // Actual equity at entry
  entry_dd_final?: number;         // Actual DD of protected portfolio
  entry_max_eq_final?: number;     // Actual HWM (consistent with entry_equity and cumulative)
  // User can verify: DD_final = (entry_equity - max_eq_final) / max_eq_final

  // DEPRECATED - kept for backwards compatibility
  entry_drawdown?: number;    // Use entry_dd_decision instead
  entry_max_equity?: number;  // Use entry_max_eq_final instead
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
  optimal_params?: OptimalParams;  // Optional - not always provided by API
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
  // LONG-ONLY signals: UPRO_3X, SPY_1X, or CASH
  signal: 'UPRO_3X' | 'SPY_1X' | 'CASH' | 'STRONG_LONG' | 'LONG' | 'WEAK_LONG' | 'SHORT' | 'STRONG_SHORT';
  instrument?: 'UPRO' | 'SPY' | 'CASH';
  change: number;
  date: string;
  prediction?: number;
  percentile?: number;
}

export interface SignalsData {
  date: string;
  strategy?: string;  // 'LONG-ONLY'
  consensus: {
    position: number;
    // LONG-ONLY counts
    upro_count?: number;
    spy_count?: number;
    cash_count?: number;
    recommended?: 'UPRO' | 'SPY' | 'CASH';
    // Legacy counts (for backward compatibility)
    bullish_count?: number;
    neutral_count?: number;
    bearish_count?: number;
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
  strategy?: string;  // 'LONG-ONLY'
  config: {
    initial_capital: number;
    volatility_target?: number;
    n_bootstrap?: number;
    confidence_level?: number;
  };
  test_period: {
    start: string;
    end: string;
    n_days: number;
  };
  benchmark: {
    spy_final_value?: number;
    spy_total_return?: number;
    total_return?: number;
    sharpe?: number;
    max_drawdown?: number;
    final_value?: number;
  };
  best_model?: BestModel | string;
  models_beating_benchmark?: number;
  models_beating_spy?: number;
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
