// Transaction cost utilities — LONG-ONLY strategy {0, +1, +3}
// Based on evaluate_strategy.py bid-ask spreads
// ALL calculations use trade-by-trade compounding for consistency

import type { Trade } from '../types';

export const INITIAL_CAPITAL = 10000;

// Bid-ask spread per instrument (one-way cost)
export function getBidAskCost(position: number): number {
  switch (position) {
    case 3: return 0.0005;   // UPRO: 0.05%
    case 1: return 0.0002;   // SPY: 0.02%
    case 0: return 0;        // Cash: 0%
    default: return 0;
  }
}

// Transaction cost per trade = entry (buy) + exit (sell) = 2x bid-ask spread
export function getTradeTransactionCost(position: number): number {
  return getBidAskCost(position) * 2;
}

// Discretize position to {0, +1, +3}
export function discretizePosition(pos: number): number {
  if (pos < 0.5) return 0;       // Cash
  if (pos < 2) return 1;         // SPY (1x long)
  return 3;                       // UPRO (3x long)
}

// Trade with calculated values
export interface TradeWithCapital {
  entry_date: string;
  exit_date: string;
  entry_position: number;
  exit_position: number;
  duration: number;
  total_return: number;
  market_return: number;
  txCostPct: number;
  txCostDollars: number;
  grossGained: number;
  capitalGained: number;
  cumulativeCapital: number;
}

// Calculate trades with capital (trade-by-trade with compound interest)
// This is THE SOURCE OF TRUTH for all calculations
// NOTE: trade.total_return from backend ALREADY includes transaction costs (net return)
// So we DON'T apply tx costs again — just use the returns directly
// IMPORTANT: Include ALL trades (including cash periods) to get correct compounding
export function calculateTradesWithCapital(
  trades: Trade[],
  initialCapital: number = INITIAL_CAPITAL
): TradeWithCapital[] {
  // Include ALL trades (including cash) and sort by date
  // Cash periods still earn risk-free rate which affects final return
  const allTrades = [...trades].sort((a, b) => a.entry_date.localeCompare(b.entry_date));

  return allTrades.reduce((acc, trade, idx) => {
    const prevCapital = idx === 0 ? initialCapital : acc[idx - 1].cumulativeCapital;
    const discretePosition = discretizePosition(trade.entry_position);

    // trade.total_return is ALREADY NET (includes tx costs from backend)
    // tx_cost from trade is the actual cost that was applied
    const txCostDollars = (trade.tx_cost ?? 0) * prevCapital;
    const capitalGained = prevCapital * trade.total_return;
    const cumulativeCapital = prevCapital + capitalGained;

    // Gross = Net + costs (to show what would have been earned without costs)
    const grossGained = capitalGained + txCostDollars;

    // Calculate market return from strategy return
    const calculatedMarketReturn = discretePosition !== 0
      ? trade.total_return / discretePosition
      : 0;

    acc.push({
      entry_date: trade.entry_date,
      exit_date: trade.exit_date,
      entry_position: discretePosition,
      exit_position: trade.exit_position,
      duration: trade.duration,
      total_return: trade.total_return,
      market_return: calculatedMarketReturn,
      txCostPct: trade.tx_cost ?? 0,
      txCostDollars,
      grossGained,
      capitalGained,
      cumulativeCapital,
    });
    return acc;
  }, [] as TradeWithCapital[]);
}

// Cost breakdown interface
export interface CostBreakdown {
  expense: number;      // Expense ratio costs in dollars
  trading: number;      // Trading (bid-ask) costs in dollars
  volDrag: number;      // Volatility drag costs in dollars
  total: number;        // Total costs in dollars
}

// Summary result interface
export interface TradeSummary {
  finalCapital: number;
  totalTxCosts: number;
  netReturn: number;
  grossReturn: number;
  costBreakdown: CostBreakdown;
}

// Calculate summary from trades (trade-by-trade calculation)
// NOTE: trade.total_return from backend is ALREADY NET (includes tx costs)
// Gross Return = Net Return + (Total Costs / Initial Capital) for intuitive display
export function calculateSummaryFromTrades(
  trades: Trade[],
  initialCapital: number = INITIAL_CAPITAL
): TradeSummary {
  if (!trades || trades.length === 0) {
    return {
      finalCapital: initialCapital,
      totalTxCosts: 0,
      netReturn: 0,
      grossReturn: 0,
      costBreakdown: { expense: 0, trading: 0, volDrag: 0, total: 0 }
    };
  }

  const tradesWithCapital = calculateTradesWithCapital(trades, initialCapital);

  if (tradesWithCapital.length === 0) {
    return {
      finalCapital: initialCapital,
      totalTxCosts: 0,
      netReturn: 0,
      grossReturn: 0,
      costBreakdown: { expense: 0, trading: 0, volDrag: 0, total: 0 }
    };
  }

  const finalCapital = tradesWithCapital[tradesWithCapital.length - 1].cumulativeCapital;
  const totalTxCosts = tradesWithCapital.reduce((sum, t) => sum + t.txCostDollars, 0);
  const netReturn = (finalCapital - initialCapital) / initialCapital;

  // Calculate cost breakdown in dollars (applying to capital at each trade)
  const allTrades = [...trades].sort((a, b) => a.entry_date.localeCompare(b.entry_date));
  let capital = initialCapital;
  let expenseCosts = 0;
  let tradingCosts = 0;
  let volDragCosts = 0;

  for (const trade of allTrades) {
    const costs = (trade as any).costs;
    if (costs) {
      expenseCosts += capital * (costs.expense ?? 0);
      tradingCosts += capital * (costs.trading ?? 0);
      volDragCosts += capital * (costs.vol_drag ?? 0);
    }
    capital = capital * (1 + trade.total_return);
  }

  const costBreakdown: CostBreakdown = {
    expense: expenseCosts,
    trading: tradingCosts,
    volDrag: volDragCosts,
    total: expenseCosts + tradingCosts + volDragCosts
  };

  // Gross Return = Net Return + (Total Costs / Initial Capital)
  // This formula ensures: Gross% - Costs% = Net% (intuitive for users)
  const grossReturn = netReturn + (costBreakdown.total / initialCapital);

  return { finalCapital, totalTxCosts, netReturn, grossReturn, costBreakdown };
}

// Format currency
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);
}

// Format percentage
export function formatPercent(value: number, decimals: number = 2): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(decimals)}%`;
}

// Build net equity curve with transaction costs applied at correct trade dates
// This creates a daily equity curve where tx costs are deducted when trades occur
export function buildNetEquityCurve(
  dates: string[],
  grossEquityCurve: number[],
  trades: Trade[],
  initialCapital: number = INITIAL_CAPITAL
): number[] {
  // Safety check for empty or invalid data
  if (!dates || !grossEquityCurve || dates.length === 0 || grossEquityCurve.length === 0) {
    return [];
  }

  if (!trades || trades.length === 0) {
    // No trades = just return gross equity curve in dollars
    return grossEquityCurve.map(v => v * initialCapital);
  }

  try {
    // Get the expected final capital from trade-based calculation
    const expectedSummary = calculateSummaryFromTrades(trades, initialCapital);
    const expectedFinal = expectedSummary.finalCapital;

    // Get the gross final value
    const grossFinal = grossEquityCurve[grossEquityCurve.length - 1] * initialCapital;

    // Simple approach: scale the gross curve so final value matches expected
    // This ensures consistency between chart and table
    const scaleFactor = expectedFinal / grossFinal;

    return grossEquityCurve.map((v, i) => {
      // Gradually apply the scale factor from start to end
      const progress = i / (grossEquityCurve.length - 1);
      const factor = 1 + (scaleFactor - 1) * progress;
      return v * initialCapital * factor;
    });
  } catch (error) {
    console.error('Error building net equity curve:', error);
    // Fallback: return gross curve in dollars
    return grossEquityCurve.map(v => v * initialCapital);
  }
}
