#!/usr/bin/env python3
"""
PIMCO-style Quant Model — Historical Backtest Module
=====================================================
Tests whether the 4-factor model would have flagged Samsung & SK Hynix
as SELL before the Q2 2025 semiconductor correction.

Key innovation: MOMENTUM DECELERATION DETECTOR
  - PIMCO doesn't just look at absolute momentum
  - They look at whether momentum is ACCELERATING or DECELERATING
  - A stock with 12M +50% but 1M -5% = momentum exhaustion = SELL signal
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# MOMENTUM DECELERATION — The Key PIMCO Signal
# ============================================================
def momentum_deceleration_score(closes, end_date=None):
    """
    Detects momentum exhaustion: strong long-term but weakening short-term.
    
    Signal types:
      STRONG: 12M and 6M and 3M all positive (trending up)
      DECEL: 12M positive but 3M or 1M turning negative (rolling over) ← PIMCO SELL
      REVERSAL: 6M turning negative after strong 12M
      CRASH: All timeframes negative
    
    Returns: (deceleration_flag, details_dict)
    """
    if end_date:
        closes = closes[closes.index <= end_date]
    
    if len(closes) < 252:
        return None, {'error': 'insufficient data'}
    
    c = closes.iloc[-1]
    c_1m = closes.iloc[-22] if len(closes) > 22 else closes.iloc[0]
    c_3m = closes.iloc[-66] if len(closes) > 66 else closes.iloc[0]
    c_6m = closes.iloc[-132] if len(closes) > 132 else closes.iloc[0]
    c_12m = closes.iloc[-252] if len(closes) > 252 else closes.iloc[0]
    
    m_1m = (c / c_1m - 1)
    m_3m = (c / c_3m - 1)
    m_6m = (c / c_6m - 1)
    m_12m = (c / c_12m - 1)
    
    # 5-day and 20-day for short-term
    c_5d = closes.iloc[-6] if len(closes) > 6 else closes.iloc[0]
    c_20d = closes.iloc[-21] if len(closes) > 21 else closes.iloc[0]
    m_5d = (c / c_5d - 1)
    m_20d = (c / c_20d - 1)
    
    # MA50 / MA200 trend
    ma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else c
    ma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else c
    above_ma50 = c > ma50
    above_ma200 = c > ma200
    
    # Volatility (30-day)
    returns = closes.pct_change().dropna()
    vol_30d = returns.iloc[-30:].std() * np.sqrt(252) if len(returns) >= 30 else 0.30
    vol_90d = returns.iloc[-90:].std() * np.sqrt(252) if len(returns) >= 90 else 0.30
    
    # Volatility expansion (recent vol vs longer-term)
    vol_expansion = vol_30d / vol_90d if vol_90d > 0 else 1.0
    
    # === DECELERATION LOGIC ===
    decel_score = 0  # 0 = strong momentum, negative = decelerating
    
    # Factor 1: Short-term vs long-term divergence
    if m_12m > 0.20 and m_1m < 0:
        decel_score -= 2  # Big run-up but recent month negative
    elif m_12m > 0.20 and m_3m < 0:
        decel_score -= 1  # Big run-up but 3M turning
    elif m_6m > 0.15 and m_1m < -0.03:
        decel_score -= 1  # Decent run but sharp recent drop
    
    # Factor 2: Sequential deceleration (each window weaker than last)
    windows = [m_12m, m_6m, m_3m, m_1m]
    if all(w > windows[i+1] - 0.05 for i, w in enumerate(windows[:-1])):
        if windows[0] > 0.10:  # Only flag if there was meaningful run-up
            decel_score -= 1
    
    # Factor 3: Below MA50 (short-term trend break)
    if not above_ma50 and m_6m > 0.10:
        decel_score -= 1
    
    # Factor 4: Volatility expansion (fear entering)
    if vol_expansion > 1.3:
        decel_score -= 1
    
    # Factor 5: 5-day sharp drop
    if m_5d < -0.05:
        decel_score -= 1
    
    # === SIGNAL CLASSIFICATION ===
    if decel_score <= -3:
        signal = '🔴 STRONG SELL — Momentum Exhaustion'
    elif decel_score <= -2:
        signal = '🟠 SELL — Decelerating'
    elif decel_score <= -1:
        signal = '🟡 CAUTION — Monitor'
    elif m_12m > 0.20 and m_3m > 0.05:
        signal = '🟢 STRONG BUY — Momentum Accelerating'
    elif m_6m > 0.05:
        signal = '🔵 BUY — Positive Trend'
    else:
        signal = '⚪ NEUTRAL'
    
    details = {
        'm_12m': m_12m, 'm_6m': m_6m, 'm_3m': m_3m, 'm_1m': m_1m,
        'm_20d': m_20d, 'm_5d': m_5d,
        'above_ma50': above_ma50, 'above_ma200': above_ma200,
        'vol_30d': vol_30d, 'vol_expansion': vol_expansion,
        'decel_score': decel_score,
        'signal': signal,
    }
    
    return decel_score, details

# ============================================================
# MONTHLY BACKTEST: Track scores over time
# ============================================================
def monthly_backtest(ticker, start="2024-06-01", end="2025-07-31"):
    """
    Track momentum deceleration score month by month.
    Shows when the SELL signal first appeared.
    """
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, interval="1d")
    if len(hist) < 252:
        print(f"  {ticker}: insufficient data ({len(hist)} bars)")
        return []
    
    closes = hist['Close']
    
    # Score at end of each month
    monthly_dates = closes.resample('ME').last().index
    
    results = []
    for date in monthly_dates:
        score, details = momentum_deceleration_score(closes, end_date=date)
        if score is not None:
            results.append({
                'date': date.strftime('%Y-%m'),
                'decel_score': score,
                'signal': details['signal'],
                'm_12m': details['m_12m'],
                'm_6m': details['m_6m'],
                'm_3m': details['m_3m'],
                'm_1m': details['m_1m'],
                'm_5d': details['m_5d'],
                'price': closes[closes.index <= date].iloc[-1],
            })
    
    return results

# ============================================================
# RUN BACKTEST
# ============================================================
def run_backtest():
    print("="*100)
    print(" PIMCO-STYLE QUANT MODEL — HISTORICAL BACKTEST")
    print(" Can the model catch the Samsung/Hynix sell-off before Q2 2025?")
    print("="*100)
    
    targets = [
        ('005930.KS', 'Samsung Electronics'),
        ('000660.KS', 'SK Hynix'),
        ('NVDA', 'NVIDIA'),
        ('ASML', 'ASML'),
        ('TSM', 'TSMC'),
        ('AMD', 'AMD'),
        ('2330.TW', 'TSMC (TW)'),
    ]
    
    for ticker, name in targets:
        print(f"\n{'─'*100}")
        print(f"  {name} ({ticker})")
        print(f"{'─'*100}")
        
        results = monthly_backtest(ticker)
        if not results:
            continue
        
        print(f"  {'Month':<10} {'Signal':<42} {'Decel':>6} {'12M%':>8} {'6M%':>8} {'3M%':>8} {'1M%':>8} {'Price':>12}")
        print(f"  {'─'*94}")
        
        for r in results:
            emoji = ''
            if 'STRONG SELL' in r['signal']:
                emoji = ' ⚠️⚠️'
            elif 'SELL' in r['signal']:
                emoji = ' ⚠️'
            
            print(f"  {r['date']:<10} {r['signal']:<42} {r['decel_score']:>+6} "
                  f"{r['m_12m']*100:>+7.1f}% {r['m_6m']*100:>+7.1f}% "
                  f"{r['m_3m']*100:>+7.1f}% {r['m_1m']*100:>+7.1f}% "
                  f"{r['price']:>12,.0f}{emoji}")
        
        # Find first SELL signal
        first_sell = next((r for r in results if 'SELL' in r['signal']), None)
        if first_sell:
            print(f"\n  🎯 First SELL signal: {first_sell['date']} — Price: {first_sell['price']:,.0f}")
            
            # How much did it drop after?
            t = yf.Ticker(ticker)
            full = t.history(start="2024-01-01", end="2025-08-01")
            if len(full):
                signal_date = pd.Timestamp(first_sell['date'])
                post = full[full.index > signal_date + pd.Timedelta(days=7)]
                if len(post) > 0:
                    peak = full[full.index <= signal_date]['Close'].max()
                    trough = post['Close'].min()
                    drawdown = (trough / peak - 1) * 100
                    print(f"     Peak before signal: {peak:,.0f} → Subsequent low: {trough:,.0f} ({drawdown:+.1f}%)")
        else:
            print(f"\n  ✅ No SELL signal detected in the period")
    
    # === SUMMARY ===
    print(f"\n{'='*100}")
    print(" SUMMARY: When did each stock first trigger SELL?")
    print("="*100)
    print(f"  {'Stock':<25} {'First SELL':<12} {'Price at Signal':>15} {'Subsequent Drawdown':>20}")
    print(f"  {'─'*72}")
    
    for ticker, name in targets:
        results = monthly_backtest(ticker)
        first_sell = next((r for r in results if 'SELL' in r['signal']), None)
        if first_sell:
            print(f"  {name:<25} {first_sell['date']:<12} {first_sell['price']:>15,.0f}")
        else:
            print(f"  {name:<25} {'No signal':<12} {'—':>15}")

# ============================================================
# CURRENT SCORES
# ============================================================
def current_scores():
    """Current momentum deceleration scores for key stocks."""
    print("\n" + "="*100)
    print(" CURRENT MOMENTUM DECELERATION SCORES (as of today)")
    print("="*100)
    
    tickers = [
        ('005930.KS', 'Samsung'), ('000660.KS', 'SK Hynix'),
        ('NVDA', 'NVIDIA'), ('TSM', 'TSMC ADR'), ('AMD', 'AMD'),
        ('ASML', 'ASML'), ('2330.TW', 'TSMC TW'),
        ('META', 'Meta'), ('MSFT', 'Microsoft'), ('GOOGL', 'Alphabet'),
        ('AAPL', 'Apple'), ('AMZN', 'Amazon'),
        ('0700.HK', 'Tencent'), ('9988.HK', 'Alibaba'), ('3690.HK', 'Meituan'),
        ('1810.HK', 'Xiaomi'),
    ]
    
    print(f"\n  {'Stock':<15} {'Signal':<42} {'Decel':>6} {'12M%':>8} {'6M%':>8} {'3M%':>8} {'1M%':>8}")
    print(f"  {'─'*96}")
    
    for ticker, name in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2y", interval="1d")
            if len(hist) < 252:
                # Try 1y
                hist = t.history(period="1y", interval="1d")
                if len(hist) < 60:
                    continue
            closes = hist['Close']
            score, details = momentum_deceleration_score(closes)
            if score is not None:
                print(f"  {name+' ('+ticker+')':<22} {details['signal']:<42} {score:>+6} "
                      f"{details['m_12m']*100:>+7.1f}% {details['m_6m']*100:>+7.1f}% "
                      f"{details['m_3m']*100:>+7.1f}% {details['m_1m']*100:>+7.1f}%")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

if __name__ == '__main__':
    run_backtest()
    current_scores()
