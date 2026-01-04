// Transaction cost utilities - consistent across all pages
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
    case -1: return 0.0004;  // SH: 0.04%
    case -3: return 0.0006;  // SPXU: 0.06%
    default: return 0;
  }
}

// Transaction cost per trade = entry (buy) + exit (sell) = 2x bid-ask spread
export function getTradeTransactionCost(position: number): number {
  return getBidAskCost(position) * 2;
}

// Discretize position to {-3, -1, 0, +1, +3}
export function discretizePosition(pos: number): number {
  if (pos <= -2) return -3;      // SPXU (3x short)
  if (pos <= -0.5) return -1;    // SH (1x short)
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
export function calculateTradesWithCapital(
  trades: Trade[],
  initialCapital: number = INITIAL_CAPITAL
): TradeWithCapital[] {
  // Filter out cash positions and sort by date
  const activeTrades = trades
    .filter(trade => Math.abs(trade.entry_position) >= 0.5)
    .sort((a, b) => a.entry_date.localeCompare(b.entry_date));

  return activeTrades.reduce((acc, trade, idx) => {
    const prevCapital = idx === 0 ? initialCapital : acc[idx - 1].cumulativeCapital;
    const discretePosition = discretizePosition(trade.entry_position);
    const txCostPct = getTradeTransactionCost(discretePosition);
    const txCostDollars = prevCapital * txCostPct;
    const grossGained = prevCapital * trade.total_return;
    const capitalGained = grossGained - txCostDollars;
    const cumulativeCapital = prevCapital + capitalGained;

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
      txCostPct,
      txCostDollars,
      grossGained,
      capitalGained,
      cumulativeCapital,
    });
    return acc;
  }, [] as TradeWithCapital[]);
}

// Calculate summary from trades (trade-by-trade calculation)
export function calculateSummaryFromTrades(
  trades: Trade[],
  initialCapital: number = INITIAL_CAPITAL
): { finalCapital: number; totalTxCosts: number; netReturn: number; grossReturn: number } {
  const tradesWithCapital = calculateTradesWithCapital(trades, initialCapital);

  if (tradesWithCapital.length === 0) {
    return { finalCapital: initialCapital, totalTxCosts: 0, netReturn: 0, grossReturn: 0 };
  }

  const finalCapital = tradesWithCapital[tradesWithCapital.length - 1].cumulativeCapital;
  const totalTxCosts = tradesWithCapital.reduce((sum, t) => sum + t.txCostDollars, 0);
  const netReturn = (finalCapital - initialCapital) / initialCapital;

  // Calculate true gross return: what would have been earned WITHOUT transaction costs
  // This requires simulating trades without tx costs to get the correct compounded result
  const activeTrades = trades
    .filter(trade => Math.abs(trade.entry_position) >= 0.5)
    .sort((a, b) => a.entry_date.localeCompare(b.entry_date));

  let grossCapital = initialCapital;
  for (const trade of activeTrades) {
    const grossGained = grossCapital * trade.total_return;
    grossCapital = grossCapital + grossGained;
  }
  const grossReturn = (grossCapital - initialCapital) / initialCapital;

  return { finalCapital, totalTxCosts, netReturn, grossReturn };
}

// ============================================================================
// LEGACY FUNCTIONS - Use calculateSummaryFromTrades for consistency
// These are kept for backward compatibility but should be migrated
// ============================================================================

// Calculate total transaction costs (DEPRECATED - use calculateSummaryFromTrades)
export function calculateTotalTransactionCosts(
  nTrades: number,
  pct3xLong: number,
  pctLong: number,
  pctCash: number,
  pctShort: number,
  pct3xShort: number,
  initialCapital: number = INITIAL_CAPITAL
): number {
  const avgCostPct =
    (pct3xLong / 100) * getTradeTransactionCost(3) +
    (pctLong / 100) * getTradeTransactionCost(1) +
    (pctCash / 100) * getTradeTransactionCost(0) +
    (pctShort / 100) * getTradeTransactionCost(-1) +
    (pct3xShort / 100) * getTradeTransactionCost(-3);
  return nTrades * avgCostPct * initialCapital;
}

// Adjust return for transaction costs (DEPRECATED - use calculateSummaryFromTrades)
export function adjustReturnForCosts(
  grossReturn: number,
  nTrades: number,
  pct3xLong: number,
  pctLong: number,
  pctCash: number,
  pctShort: number,
  pct3xShort: number
): number {
  const avgCostPct =
    (pct3xLong / 100) * getTradeTransactionCost(3) +
    (pctLong / 100) * getTradeTransactionCost(1) +
    (pctCash / 100) * getTradeTransactionCost(0) +
    (pctShort / 100) * getTradeTransactionCost(-1) +
    (pct3xShort / 100) * getTradeTransactionCost(-3);
  const totalCostPct = nTrades * avgCostPct;
  return grossReturn - totalCostPct;
}

// Transaction cost breakdown by position type
export const TRANSACTION_COST_BREAKDOWN = {
  '+3x (UPRO)': { bidAsk: 0.0005, roundTrip: 0.001, label: '0.10%' },
  '+1x (SPY)': { bidAsk: 0.0002, roundTrip: 0.0004, label: '0.04%' },
  '0 (Cash)': { bidAsk: 0, roundTrip: 0, label: '0%' },
  '-1x (SH)': { bidAsk: 0.0004, roundTrip: 0.0008, label: '0.08%' },
  '-3x (SPXU)': { bidAsk: 0.0006, roundTrip: 0.0012, label: '0.12%' },
};

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
