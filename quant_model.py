#!/usr/bin/env python3
"""
PIMCO-style 4-Factor Quant Model for Equity Selection
======================================================
Replicates PIMCO's systematic equity approach:
  Quality, Momentum, Value, Growth → composite score → ranked signals

Universe: MSCI ACWI large/mid caps (approximated via liquid tickers)
Rebalance: Monthly
Target: Excess return vs benchmark

Factor definitions (from PIMCO slides + academic research):
  QUALITY: ROE, gross profitability, low accruals, low leverage
  MOMENTUM: 12-1 month residual momentum, 6-1, 3-1
  VALUE: FCF yield, earnings yield, book-to-market
  GROWTH: EPS revision momentum, YoY earnings growth, revenue growth

Usage: python3 quant_model.py [--mode backtest|live]
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings, json, sys, os

warnings.filterwarnings('ignore')

# ============================================================
# DATA LAYER
# ============================================================
def fetch_stock_data(ticker, period="2y"):
    """Fetch price + fundamentals for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval="1d")
        if len(hist) < 60:
            return None

        info = {}
        try:
            raw = t.info
            # Extract what we need, handle missing fields gracefully
            for key in ['returnOnEquity','grossMargins','operatingMargins','netMargins',
                        'priceToBook','priceToEarnings','forwardPE','trailingPE',
                        'priceToSalesTrailing12Months','enterpriseToEbit','enterpriseToRevenue',
                        'totalDebt','totalCash','totalCashPerShare','marketCap',
                        'revenueGrowth','earningsGrowth','earningsQuarterlyGrowth',
                        'debtToEquity','currentRatio','quickRatio',
                        'freeCashflow','operatingCashflow','totalRevenue','grossProfits',
                        'revenuePerShare','earningsGrowth',
                        'totalCash','totalDebt','bookValue','profitMargins',
                        'pegRatio','beta','sharesOutstanding',
                        'targetMeanPrice','recommendationKey','numberOfAnalystOpinions']:
                info[key] = raw.get(key)
        except:
            pass

        # Compute returns for momentum
        closes = hist['Close']
        returns = closes.pct_change().dropna()

        return {
            'ticker': ticker,
            'hist': hist,
            'closes': closes,
            'returns': returns,
            'info': info,
        }
    except Exception as e:
        print(f"  WARN {ticker}: {e}")
        return None

def fetch_batch(tickers, period="2y"):
    """Fetch data for multiple tickers."""
    data = {}
    for t in tickers:
        d = fetch_stock_data(t, period)
        if d is not None:
            data[t] = d
    return data

# ============================================================
# FACTOR 1: QUALITY
# ============================================================
def score_quality(d):
    """
    Quality = profitability + financial health + low accruals.
    PIMCO: "aims to measure company financial status indicators"
    Sub-indicators: profitability metrics, management effectiveness
    """
    info = d['info']
    scores = {}
    issues = []

    # 1a. ROE (Return on Equity) — target >15%
    roe = info.get('returnOnEquity')
    if roe is not None and not np.isnan(roe):
        # ROE 0.30 = excellent, 0.15 = good, 0 = poor
        scores['roe'] = min(1.0, max(0, roe / 0.30))
    else:
        issues.append('ROE missing')

    # 1b. Gross Margin
    gm = info.get('grossMargins')
    if gm is not None and not np.isnan(gm):
        scores['gross_margin'] = min(1.0, max(0, gm / 0.50))
    else:
        issues.append('GM missing')

    # 1c. Operating Margin (management effectiveness proxy)
    om = info.get('operatingMargins')
    if om is not None and not np.isnan(om):
        scores['op_margin'] = min(1.0, max(0, om / 0.25))
    else:
        issues.append('OM missing')

    # 1d. Debt to Equity (lower is better, quality companies less leveraged)
    de = info.get('debtToEquity')
    if de is not None and not np.isnan(de):
        # D/E < 50 = excellent, > 200 = poor
        scores['leverage'] = max(0, 1.0 - de / 200.0)
    else:
        issues.append('D/E missing')

    # 1e. Free Cash Flow / Revenue (FCF generation quality)
    fcf = info.get('freeCashflow')
    rev = info.get('totalRevenue')
    if fcf and rev and rev > 0:
        fcf_margin = fcf / rev
        scores['fcf_margin'] = min(1.0, max(0, fcf_margin / 0.15))
    else:
        issues.append('FCF missing')

    if not scores:
        return 0.5, issues, {}

    avg = np.mean(list(scores.values()))
    return avg, issues, scores

# ============================================================
# FACTOR 2: MOMENTUM (residual-style)
# ============================================================
def score_momentum(d):
    """
    PIMCO: "identify stock price action trends"
    Uses 12-1 month, 6-1, 3-1 momentum (skipping last month to avoid reversal)
    """
    closes = d['closes']
    if len(closes) < 252:
        if len(closes) >= 60:
            # Use what we have
            m12 = (closes.iloc[-1] / closes.iloc[-min(60,len(closes)-1)] - 1) if len(closes)>1 else 0
            m6 = (closes.iloc[-1] / closes.iloc[-min(40,len(closes)-1)] - 1) if len(closes)>1 else 0
            m3 = (closes.iloc[-1] / closes.iloc[-min(20,len(closes)-1)] - 1) if len(closes)>1 else 0
        else:
            return 0.5, ['insufficient data'], {}
    else:
        # Standard 12-1, 6-1, 3-1 (skip last 21 days)
        skip = 21
        m12 = closes.iloc[-1-skip] / closes.iloc[-252] - 1 if len(closes) >= 252+skip else 0
        m6 = closes.iloc[-1-skip] / closes.iloc[-126-skip] - 1 if len(closes) >= 126+skip else 0
        m3 = closes.iloc[-1-skip] / closes.iloc[-63-skip] - 1 if len(closes) >= 63+skip else 0

    scores = {
        'mom_12_1': np.tanh(m12 * 2),   # squash to [-1,1]
        'mom_6_1': np.tanh(m6 * 3),
        'mom_3_1': np.tanh(m3 * 5),
    }

    # Relative strength vs sector/market (using recent 50d slope)
    if len(closes) >= 50:
        ma50 = closes.rolling(50).mean().dropna()
        if len(ma50) >= 10:
            slope = (ma50.iloc[-1] / ma50.iloc[-10] - 1)
            scores['trend_slope'] = np.tanh(slope * 10)

    avg = np.mean(list(scores.values()))
    # Normalize to [0,1]
    avg = (avg + 1) / 2
    return avg, [], scores

# ============================================================
# FACTOR 3: VALUE
# ============================================================
def score_value(d):
    """
    PIMCO: "evaluate company fundamentals to measure valuation"
    Sub-indicators: cash flow-based indicators, asset value
    """
    info = d['info']
    scores = {}
    issues = []

    # 3a. FCF Yield (Free Cash Flow / Market Cap)
    fcf = info.get('freeCashflow')
    mcap = info.get('marketCap')
    if fcf and mcap and mcap > 0:
        fcf_yield = fcf / mcap
        # 5% yield = great, 0% = poor
        scores['fcf_yield'] = min(1.0, max(0, fcf_yield / 0.05))
    else:
        issues.append('FCF yield missing')

    # 3b. Earnings Yield (1/PE)
    pe = info.get('trailingPE') or info.get('priceToEarnings')
    if pe and pe > 0 and pe < 200:
        ey = 1.0 / pe
        # EY of 0.08 (PE=12.5) = great, 0.02 (PE=50) = poor
        scores['earnings_yield'] = min(1.0, max(0, (ey - 0.02) / 0.06))
    else:
        issues.append('PE missing')

    # 3c. Forward PE vs trailing (improving valuation)
    fpe = info.get('forwardPE')
    tpe = info.get('trailingPE')
    if fpe and tpe and tpe > 0 and fpe > 0:
        # Forward PE lower than trailing = earnings growth expected = good value
        scores['pe_trend'] = max(0, min(1.0, (tpe - fpe) / tpe + 0.3))

    # 3d. Price to Book
    pb = info.get('priceToBook')
    if pb and pb > 0 and pb < 30:
        # P/B < 1 = great, P/B > 10 = poor
        scores['book_value'] = max(0, 1.0 - pb / 10.0)
    else:
        issues.append('P/B missing')

    # 3e. EV/EBITDA (enterprise value based — cash flow proxy)
    ev_ebit = info.get('enterpriseToEbit')
    if ev_ebit and ev_ebit > 0 and ev_ebit < 80:
        # EV/EBIT 10 = good, 30+ = expensive
        scores['ev_ebit'] = max(0, 1.0 - (ev_ebit - 5) / 30.0)
    else:
        issues.append('EV/EBIT missing')

    if not scores:
        return 0.5, issues, {}

    avg = np.mean(list(scores.values()))
    return avg, issues, scores

# ============================================================
# FACTOR 4: GROWTH
# ============================================================
def score_growth(d):
    """
    PIMCO: "measure growth potential not yet reflected in price"
    Sub-indicator: revised earnings forecasts
    """
    info = d['info']
    scores = {}
    issues = []

    # 4a. Revenue Growth YoY
    rg = info.get('revenueGrowth')
    if rg is not None and not np.isnan(rg):
        scores['rev_growth'] = min(1.0, max(0, rg / 0.25))
    else:
        issues.append('rev growth missing')

    # 4b. Earnings Growth
    eg = info.get('earningsGrowth')
    if eg is not None and not np.isnan(eg):
        scores['earn_growth'] = min(1.0, max(0, eg / 0.30))
    else:
        issues.append('earn growth missing')

    # 4c. Quarterly Earnings Growth
    eqg = info.get('earningsQuarterlyGrowth')
    if eqg is not None and not np.isnan(eqg):
        scores['qtr_earn'] = min(1.0, max(0, eqg / 0.40))
    else:
        issues.append('qtr earn missing')

    # 4d. Forward PE < Trailing PE (growth accelerating)
    fpe = info.get('forwardPE')
    tpe = info.get('trailingPE')
    if fpe and tpe and tpe > 0:
        # If forward PE is 20% below trailing, strong growth signal
        discount = (tpe - fpe) / tpe
        scores['pe_momentum'] = min(1.0, max(0, discount / 0.20 + 0.5))
    else:
        issues.append('PE momentum missing')

    # 4e. Analyst target upside (growth implied by street)
    target = info.get('targetMeanPrice')
    current = d['closes'].iloc[-1] if len(d['closes']) > 0 else None
    if target and current and current > 0:
        upside = target / current - 1
        scores['analyst_upside'] = min(1.0, max(0, upside / 0.25))
    else:
        issues.append('target missing')

    if not scores:
        return 0.5, issues, {}

    avg = np.mean(list(scores.values()))
    return avg, issues, scores

# ============================================================
# COMPOSITE SCORE (PIMCO-style equal weight + conviction)
# ============================================================
def compute_composite(d, weights=None):
    """
    Combine 4 factors into composite score.
    PIMCO uses equal-weighted factors to reduce style bias.
    """
    if weights is None:
        weights = {'quality': 0.30, 'momentum': 0.25, 'value': 0.20, 'growth': 0.25}

    q_score, q_issues, q_detail = score_quality(d)
    m_score, m_issues, m_detail = score_momentum(d)
    v_score, v_issues, v_detail = score_value(d)
    g_score, g_issues, g_detail = score_growth(d)

    composite = (weights['quality'] * q_score +
                 weights['momentum'] * m_score +
                 weights['value'] * v_score +
                 weights['growth'] * g_score)

    # Signal: SELL if < 0.35, REDUCE if < 0.45, HOLD if < 0.55, ACCUMULATE if < 0.70, BUY if >= 0.70
    if composite < 0.35:
        signal = '🔴 SELL'
    elif composite < 0.45:
        signal = '🟠 REDUCE'
    elif composite < 0.55:
        signal = '🟡 HOLD'
    elif composite < 0.70:
        signal = '🔵 ACCUMULATE'
    else:
        signal = '🟢 BUY'

    return {
        'ticker': d['ticker'],
        'composite': composite,
        'signal': signal,
        'quality': q_score,
        'momentum': m_score,
        'value': v_score,
        'growth': g_score,
        'details': {
            'quality': q_detail, 'momentum': m_detail,
            'value': v_detail, 'growth': g_detail
        },
        'issues': {
            'quality': q_issues, 'momentum': m_issues,
            'value': v_issues, 'growth': g_issues
        },
        'price': d['closes'].iloc[-1],
        'info': {k: v for k, v in d['info'].items()
                 if v is not None and not (isinstance(v, float) and np.isnan(v))}
    }

# ============================================================
# RANKING & DISPLAY
# ============================================================
def rank_universe(data, weights=None):
    """Score and rank all stocks in universe."""
    results = []
    for ticker, d in data.items():
        r = compute_composite(d, weights)
        results.append(r)
    results.sort(key=lambda x: x['composite'], reverse=True)
    return results

def print_dashboard(results, title="Quant Model Ranking"):
    """Print a formatted dashboard."""
    print(f"\n{'='*90}")
    print(f" {title}")
    print(f" Model: PIMCO-style 4-Factor (Quality 30% / Momentum 25% / Value 20% / Growth 25%)")
    print(f" Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Signal':<18} {'Ticker':<12} {'Score':>6} {'Q':>6} {'M':>6} {'V':>6} {'G':>6} {'Price':>10}")
    print(f"{'-'*90}")

    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['signal']:<18} {r['ticker']:<12} "
              f"{r['composite']:.3f}  {r['quality']:.3f}  {r['momentum']:.3f}  "
              f"{r['value']:.3f}  {r['growth']:.3f}  "
              f"{r['price']:>10.2f}")

    print(f"{'='*90}")
    buys = [r for r in results if 'BUY' in r['signal']]
    sells = [r for r in results if 'SELL' in r['signal'] or 'REDUCE' in r['signal']]
    print(f" 🟢 BUY: {len(buys)}  |  🔴 SELL/REDUCE: {len(sells)}  |  Total: {len(results)}")

# ============================================================
# CASE STUDY: Samsung & SK Hynix — "Why did PIMCO sell early Q2?"
# ============================================================
def case_study_samsung_hynix():
    """
    Test: Does our model flag Samsung (005930.KS) and SK Hynix (000660.KS)
    as SELL/REDUCE based on the same signals PIMCO saw in early Q2 2025?

    PIMCO likely saw:
    - Momentum rolling over (semiconductor peak)
    - Valuations stretched after AI run
    - Growth decelerating (memory cycle peak)
    - Quality still OK but degrading (capex surge)
    """
    print("\n" + "="*90)
    print(" CASE STUDY: Samsung (005930.KS) & SK Hynix (000660.KS)")
    print(" Did our model flag a SELL signal before the Q2 2025 selloff?")
    print("="*90)

    tickers = ['005930.KS', '000660.KS', 'NVDA', 'TSM', 'AMD', 'ASML',
               'META', 'MSFT', 'GOOGL', 'AAPL',
               '0700.HK', '9988.HK', '3690.HK', '2330.TW',
               'SOXX']

    data = fetch_batch(tickers)
    print(f" Fetched: {len(data)}/{len(tickers)} tickers")

    results = rank_universe(data)
    print_dashboard(results, "4-Factor Model — Semiconductor + Big Tech Universe")

    # Focus on Samsung & Hynix
    print("\n" + "-"*90)
    print(" SAMSUNG & SK HYNIX DETAILED BREAKDOWN")
    print("-"*90)

    for ticker in ['005930.KS', '000660.KS']:
        r = next((x for x in results if x['ticker'] == ticker), None)
        if r:
            print(f"\n  ▶ {ticker}")
            print(f"    Composite Score: {r['composite']:.3f}  →  {r['signal']}")
            print(f"    Quality:  {r['quality']:.3f}  ({', '.join(f'{k}={v:.2f}' for k,v in r['details']['quality'].items())})")
            print(f"    Momentum: {r['momentum']:.3f}  ({', '.join(f'{k}={v:.2f}' for k,v in r['details']['momentum'].items())})")
            print(f"    Value:    {r['value']:.3f}  ({', '.join(f'{k}={v:.2f}' for k,v in r['details']['value'].items())})")
            print(f"    Growth:   {r['growth']:.3f}  ({', '.join(f'{k}={v:.2f}' for k,v in r['details']['growth'].items())})")
            print(f"    Price:    {r['price']:.2f}")

            # Key metrics
            info = r['info']
            if 'returnOnEquity' in info:
                print(f"    ROE:      {info.get('returnOnEquity',0)*100:.1f}%")
            if 'revenueGrowth' in info:
                print(f"    Rev Gr:   {info.get('revenueGrowth',0)*100:.1f}%")
            if 'earningsGrowth' in info:
                print(f"    Earn Gr:  {info.get('earningsGrowth',0)*100:.1f}%")
            if 'trailingPE' in info:
                print(f"    P/E:      {info.get('trailingPE',0):.1f}x")
            if 'priceToBook' in info:
                print(f"    P/B:      {info.get('priceToBook',0):.1f}x")

    # Historical backtest: simulate Q1 2025 scores
    print("\n" + "-"*90)
    print(" HISTORICAL SIMULATION: What did momentum look like at end of Q1 2025?")
    print(" (Using price-only momentum as proxy, since fundamentals are point-in-time)")
    print("-"*90)

    for ticker in ['005930.KS', '000660.KS', 'NVDA', 'TSM', 'ASML']:
        try:
            t = yf.Ticker(ticker)
            # Get data up to ~March 31, 2025
            hist = t.history(start="2024-04-01", end="2025-04-01")
            if len(hist) < 100:
                continue
            closes = hist['Close']

            # Momentum at end of Q1 2025
            m12 = closes.iloc[-1] / closes.iloc[-252] - 1 if len(closes) >= 252 else 0
            m6 = closes.iloc[-1] / closes.iloc[-126] - 1 if len(closes) >= 126 else 0
            m3 = closes.iloc[-1] / closes.iloc[-63] - 1 if len(closes) >= 63 else 0
            m1 = closes.iloc[-1] / closes.iloc[-21] - 1 if len(closes) >= 21 else 0

            # Short-term reversal signal (last 5 days)
            m5d = closes.iloc[-1] / closes.iloc[-5] - 1 if len(closes) >= 5 else 0

            print(f"  {ticker:>12}:  12M={m12*100:>+7.1f}%  6M={m6*100:>+7.1f}%  "
                  f"3M={m3*100:>+7.1f}%  1M={m1*100:>+7.1f}%  5D={m5d*100:>+6.1f}%")

        except Exception as e:
            print(f"  {ticker}: ERROR {e}")

    print("\n  Interpretation:")
    print("  If 12M momentum was very high (>50%) but 3M/1M was decelerating,")
    print("  the model would have flagged momentum exhaustion → momentum score dropping")
    print("  → composite score falling below 0.45 → REDUCE/SELL signal.")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    case_study_samsung_hynix()
