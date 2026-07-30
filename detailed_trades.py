#!/usr/bin/env python3
"""
Detailed trade log: every buy/sell with full 4-factor breakdown,
decision rationale, and price determination.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings, json

warnings.filterwarnings('ignore')

TICKERS = ['AMAT', 'KLAC', 'LRCX', 'INTC', 'AMD', 'MRVL', 'CRDO', 'BE', 'GLW']
USD_HKD = 7.80

def score_at_date_full(hist_full, info_cache, ticker, eval_date):
    """Score stock with FULL detail for display."""
    ts = pd.Timestamp(eval_date)
    if hist_full.index.tz is not None:
        ed = ts.tz_localize(hist_full.index.tz) if ts.tz is None else ts.tz_convert(hist_full.index.tz)
    else:
        ed = ts.tz_localize(None) if ts.tz else ts
    hist = hist_full[hist_full.index <= ed].copy()
    if len(hist) < 60:
        return None
    closes = hist['Close']
    n = len(closes)
    price = closes.iloc[-1]
    info = info_cache.get(ticker, {})

    # QUALITY
    qs = {}; q_details = {}
    roe = info.get('returnOnEquity')
    if roe is not None and not (isinstance(roe, float) and np.isnan(roe)):
        qs['roe'] = min(1.0, max(0, roe / 0.30)); q_details['ROE'] = f"{roe*100:.1f}%"
    gm = info.get('grossMargins')
    if gm is not None and not (isinstance(gm, float) and np.isnan(gm)):
        qs['gm'] = min(1.0, max(0, gm / 0.50)); q_details['毛利率'] = f"{gm*100:.1f}%"
    om = info.get('operatingMargins')
    if om is not None and not (isinstance(om, float) and np.isnan(om)):
        qs['om'] = min(1.0, max(0, om / 0.25)); q_details['營業利潤率'] = f"{om*100:.1f}%"
    de = info.get('debtToEquity')
    if de is not None and not (isinstance(de, float) and np.isnan(de)):
        qs['de'] = max(0, 1.0 - de / 200.0); q_details['負債/權益'] = f"{de:.0f}"
    fcf = info.get('freeCashflow'); rev = info.get('totalRevenue')
    if fcf and rev and rev > 0:
        qs['fcf'] = min(1.0, max(0, (fcf/rev) / 0.15)); q_details['FCF利潤率'] = f"{fcf/rev*100:.1f}%"
    quality = np.mean(list(qs.values())) if qs else 0.5

    # MOMENTUM
    skip = 21
    def mom(days):
        if n > days + skip: return closes.iloc[-1-skip] / closes.iloc[-days-skip] - 1
        elif n > skip + 1: return closes.iloc[-1-skip] / closes.iloc[0] - 1
        return 0
    m12, m6, m3 = mom(252), mom(126), mom(63)
    m1m = closes.iloc[-1] / closes.iloc[-22] - 1 if n > 22 else 0
    m5d = closes.iloc[-1] / closes.iloc[-6] - 1 if n > 6 else 0
    ms = {'12': np.tanh(m12*2), '6': np.tanh(m6*3), '3': np.tanh(m3*5)}
    if n >= 60:
        ma50 = closes.rolling(50).mean().dropna()
        if len(ma50) >= 10:
            ms['slope'] = np.tanh((ma50.iloc[-1]/ma50.iloc[-10]-1)*10)
    momentum = (np.mean(list(ms.values())) + 1) / 2

    # VALUE
    vs = {}; v_details = {}
    pe = info.get('trailingPE') or info.get('priceToEarnings')
    if pe and pe > 0 and pe < 200:
        vs['ey'] = min(1.0, max(0, (1.0/pe - 0.02) / 0.06)); v_details['P/E'] = f"{pe:.1f}x"
    pb = info.get('priceToBook')
    if pb and pb > 0 and pb < 30:
        vs['pb'] = max(0, 1.0 - pb/10.0); v_details['P/B'] = f"{pb:.1f}x"
    ev_ebit = info.get('enterpriseToEbit')
    if ev_ebit and ev_ebit > 0 and ev_ebit < 80:
        vs['ev'] = max(0, 1.0 - (ev_ebit-5)/30.0); v_details['EV/EBIT'] = f"{ev_ebit:.1f}x"
    if fcf and info.get('marketCap') and info['marketCap'] > 0:
        fy = fcf/info['marketCap']
        vs['fcfy'] = min(1.0, max(0, fy / 0.05)); v_details['FCF收益率'] = f"{fy*100:.1f}%"
    value = np.mean(list(vs.values())) if vs else 0.5

    # GROWTH
    gs = {}; g_details = {}
    rg = info.get('revenueGrowth')
    if rg is not None and not (isinstance(rg, float) and np.isnan(rg)):
        gs['rev'] = min(1.0, max(0, rg/0.25)); g_details['收入增長'] = f"{rg*100:.1f}%"
    eg = info.get('earningsGrowth')
    if eg is not None and not (isinstance(eg, float) and np.isnan(eg)):
        gs['earn'] = min(1.0, max(0, eg/0.30)); g_details['盈利增長'] = f"{eg*100:.1f}%"
    eqg = info.get('earningsQuarterlyGrowth')
    if eqg is not None and not (isinstance(eqg, float) and np.isnan(eqg)):
        gs['qtr'] = min(1.0, max(0, eqg/0.40)); g_details['季度盈利增長'] = f"{eqg*100:.1f}%"
    target = info.get('targetMeanPrice')
    if target and price > 0:
        upside = target/price-1
        gs['analyst'] = min(1.0, max(0, upside/0.25)); g_details['分析員目標'] = f"${target:.0f} ({upside*100:+.0f}%)"
    growth = np.mean(list(gs.values())) if gs else 0.5

    composite = 0.30*quality + 0.25*momentum + 0.20*value + 0.25*growth

    # DECELERATION
    decel = 0; decel_reasons = []
    if m12 > 0.20 and m1m < 0:
        decel -= 2; decel_reasons.append(f"12M漲幅{m12*100:.0f}%但1M跌{m1m*100:.1f}%")
    elif m12 > 0.20 and m3 < 0:
        decel -= 1; decel_reasons.append(f"12M漲幅{m12*100:.0f}%但3M跌{m3*100:.1f}%")
    if m6 > 0.15 and m1m < -0.03:
        decel -= 1; decel_reasons.append(f"6M漲{m6*100:.0f}%但1M急跌{m1m*100:.1f}%")
    if n >= 50 and closes.iloc[-1] < closes.rolling(50).mean().iloc[-1] and m6 > 0.10:
        decel -= 1; decel_reasons.append("跌破MA50且6M仍漲>10%")
    ret = closes.pct_change().dropna()
    vol30 = ret.iloc[-30:].std()*np.sqrt(252) if len(ret) >= 30 else 0.30
    vol90 = ret.iloc[-90:].std()*np.sqrt(252) if len(ret) >= 90 else 0.30
    if vol90 > 0 and vol30/vol90 > 1.3:
        decel -= 1; decel_reasons.append(f"30日波幅{vol30*100:.0f}%>90日{vol90*100:.0f}%")
    if m5d < -0.05:
        decel -= 1; decel_reasons.append(f"5日跌{m5d*100:.1f}%")

    combined = composite
    if decel <= -3: combined -= 0.15
    elif decel <= -2: combined -= 0.08
    combined = max(0, min(1, combined))

    if combined < 0.35: signal = 'SELL'
    elif combined < 0.45: signal = 'REDUCE'
    elif combined < 0.55: signal = 'HOLD'
    elif combined < 0.70: signal = 'ACCUMULATE'
    else: signal = 'BUY'

    return {
        'ticker': ticker, 'price': price, 'score': combined, 'composite_raw': composite,
        'signal': signal, 'decel': decel, 'decel_reasons': decel_reasons,
        'quality': quality, 'momentum': momentum, 'value': value, 'growth': growth,
        'q_details': q_details, 'v_details': v_details, 'g_details': g_details,
        'm12': m12, 'm6': m6, 'm3': m3, 'm1m': m1m, 'm5d': m5d,
        'vol30': vol30, 'vol90': vol90, 'date': eval_date.strftime('%Y-%m-%d') if hasattr(eval_date,'strftime') else str(eval_date)[:10],
    }

# ============================================================
# TRADE DEFINITIONS (from simulation)
# ============================================================
TRADES = [
    # date, action, ticker, shares, price, rebalance_num
    ('2026-01-02', 'BUY',  'LRCX', 5,  184.71, 1),
    ('2026-01-02', 'BUY',  'AMAT', 3,  268.20, 1),
    ('2026-01-02', 'BUY',  'AMD',  4,  223.47, 1),
    ('2026-01-02', 'BUY',  'KLAC', 8,  127.12, 1),
    ('2026-01-02', 'BUY',  'MRVL', 11, 89.26,  1),
    ('2026-01-02', 'BUY',  'GLW',  11, 90.36,  1),
    ('2026-02-02', 'BUY',  'INTC', 2,  48.81,  2),
    ('2026-03-02', 'SELL', 'LRCX', 5,  230.56, 3),
    ('2026-03-02', 'SELL', 'INTC', 2,  45.50,  3),
    ('2026-03-02', 'SELL', 'AMD',  4,  198.62, 3),
    ('2026-03-02', 'SELL', 'KLAC', 8,  153.30, 3),
    ('2026-03-02', 'BUY',  'BE',   6,  166.00, 3),
    ('2026-04-01', 'SELL', 'GLW',  11, 142.16, 4),
    ('2026-04-01', 'SELL', 'BE',   6,  132.45, 4),
    ('2026-04-01', 'BUY',  'LRCX', 3,  221.85, 4),
    ('2026-04-01', 'BUY',  'INTC', 17, 48.03,  4),
    ('2026-04-01', 'BUY',  'KLAC', 5,  151.79, 4),
    ('2026-04-01', 'BUY',  'AMD',  4,  210.21, 4),
    ('2026-05-01', 'BUY',  'CRDO', 1,  184.38, 5),
    ('2026-05-01', 'BUY',  'GLW',  1,  158.02, 5),
]

# Stocks that were scored but NOT bought (skipped), with reason
SKIPS = [
    ('2026-01-02', 'CRDO', '評分0.629(ACCUMULATE) 但 Decel=-4 → 動量耗盡，跳過'),
    ('2026-01-02', 'INTC', '評分0.558(ACCUMULATE) 但 Decel=-3 → 動量耗盡，跳過'),
    ('2026-01-02', 'BE',   '評分0.438(REDUCE) → 評分太低，不買'),
    ('2026-02-02', 'BE',   '評分0.577(ACCUMULATE) 但價格$156 > 每股配額$106 → 資金不足'),
    ('2026-02-02', 'CRDO', '評分0.579(ACCUMULATE) 但 Decel=-5 → 嚴重動量耗盡，跳過'),
    ('2026-04-01', 'MRVL', '評分0.523(HOLD) → 信號不足，不加倉'),
    ('2026-04-01', 'CRDO', '評分0.496(HOLD) Decel=-3 → 不買'),
]

# ============================================================
# RUN
# ============================================================
def run():
    print("Fetching data...")
    hist_all = {}
    info_cache = {}
    for t in TICKERS:
        hist_all[t] = yf.Ticker(t).history(start="2024-06-01", end="2026-12-31")
        try: info_cache[t] = yf.Ticker(t).info
        except: info_cache[t] = {}

    # Score each trade's stock at the trade date
    trade_scores = {}
    for date_str, action, ticker, shares, price, rb_num in TRADES:
        eval_date = pd.Timestamp(date_str)
        s = score_at_date_full(hist_all[ticker], info_cache, ticker, eval_date)
        if s:
            key = (date_str, ticker)
            trade_scores[key] = s

    # Print detailed report
    current_rb = 0
    for date_str, action, ticker, shares, price, rb_num in TRADES:
        if rb_num != current_rb:
            current_rb = rb_num
            print(f"\n{'='*110}")
            print(f" 📅 再平衡 #{rb_num} — {date_str}")
            print(f"{'='*110}")

        s = trade_scores.get((date_str, ticker))
        if not s:
            continue

        print(f"\n  {'─'*100}")
        action_emoji = '🟢 買入' if action == 'BUY' else '🔴 沽出'
        print(f"  {action_emoji} {ticker} | {shares}股 @ ${price:.2f} | 交易金額 ${shares*price:,.2f}")
        print(f"  {'─'*100}")

        print(f"  📊 4-Factor 評分明細：")
        print(f"     {'':>3}{'因子':<12}{'得分':>8}{'加權':>8}{'加權貢獻':>10}    子指標")
        print(f"     {'─'*90}")
        print(f"     {'':>3}{'質素 Quality':<12}{s['quality']:>8.3f}{'×30%':>8}{s['quality']*0.30:>10.3f}    {s['q_details']}")
        print(f"     {'':>3}{'動量 Momentum':<12}{s['momentum']:>8.3f}{'×25%':>8}{s['momentum']*0.25:>10.3f}    12M={s['m12']*100:+.1f}% 6M={s['m6']*100:+.1f}% 3M={s['m3']*100:+.1f}% 1M={s['m1m']*100:+.1f}%")
        print(f"     {'':>3}{'價值 Value':<12}{s['value']:>8.3f}{'×20%':>8}{s['value']*0.20:>10.3f}    {s['v_details']}")
        print(f"     {'':>3}{'增長 Growth':<12}{s['growth']:>8.3f}{'×25%':>8}{s['growth']*0.25:>10.3f}    {s['g_details']}")
        print(f"     {'─'*90}")
        print(f"     {'':>3}{'原始總分':<12}{'':>8}{'':>8}{s['composite_raw']:>10.3f}")

        if s['decel'] != 0:
            adj = -0.15 if s['decel'] <= -3 else -0.08 if s['decel'] <= -2 else 0
            print(f"     {'':>3}{'動量減速懲罰':<12}{'Decel='+str(s['decel']):>8}{'':>8}{adj:>10.3f}    {'; '.join(s['decel_reasons'])}")

        print(f"     {'':>3}{'調整後總分':<12}{'':>8}{'':>8}{s['score']:>10.3f}    → 信號: {s['signal']}")
        print(f"  {'─'*100}")

        # Decision rationale
        if action == 'BUY':
            reasons = []
            if s['signal'] == 'BUY':
                reasons.append("總分≥0.70 → BUY信號，強烈買入")
            elif s['signal'] == 'ACCUMULATE':
                reasons.append("總分0.55-0.70 → ACCUMULATE信號，適量買入")
            if s['quality'] >= 0.80:
                reasons.append(f"質素極佳({s['quality']:.2f})，基本面穩健")
            if s['growth'] >= 0.80:
                reasons.append(f"增長強勁({s['growth']:.2f})，盈利/收入高速增長")
            if s['momentum'] >= 0.70:
                reasons.append(f"動量正面({s['momentum']:.2f})，趨勢向上")
            if s['decel'] > -3:
                reasons.append(f"動量減速Decel={s['decel']}（未觸發沽出門檻-3）")
            print(f"  📝 買入原因：")
            for r in reasons:
                print(f"     ✓ {r}")
        else:  # SELL
            reasons = []
            if s['decel'] <= -3:
                reasons.append(f"⚠️ 動量減速Decel={s['decel']} ≤ -3 → 自動觸發強制沽出")
                for dr in s['decel_reasons']:
                    reasons.append(f"   → {dr}")
            if s['signal'] in ('SELL', 'REDUCE'):
                reasons.append(f"信號={s['signal']}（總分{s['score']:.3f} < {'0.35' if s['signal']=='SELL' else '0.45'}）")
            print(f"  📝 沽出原因：")
            for r in reasons:
                print(f"     ✗ {r}")

        # Price determination
        print(f"  💲 價格釐定：")
        print(f"     評分日 = {date_str}（月度再平衡日）")
        print(f"     交易價 = 當日收市價 ${price:.2f}（等同模型評分用的同一日 close）")
        print(f"     原理：每月最後評分 → 次日按收市價執行（簡化為同日close）")
        print(f"     港元等值 = HK${price*USD_HKD:.2f}")

    # Print skips
    print(f"\n{'='*110}")
    print(f" 🚫 被跳過的股票（評分了但沒買入）")
    print(f"{'='*110}")
    for date_str, ticker, reason in SKIPS:
        print(f"  {date_str}  {ticker:<8} {reason}")

    # Also print ALL scores at each rebalance for context
    print(f"\n{'='*110}")
    print(f" 📋 每次再平衡的全部9隻評分排名")
    print(f"{'='*110}")

    rebalance_dates = ['2026-01-02', '2026-02-02', '2026-03-02', '2026-04-01', '2026-05-01', '2026-06-01', '2026-07-01', '2026-07-29']
    for date_str in rebalance_dates:
        eval_date = pd.Timestamp(date_str)
        scores = []
        for t in TICKERS:
            s = score_at_date_full(hist_all[t], info_cache, t, eval_date)
            if s: scores.append(s)
        scores.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n  📅 {date_str}")
        print(f"  {'Rank':<5}{'Ticker':<8}{'Signal':<14}{'Score':>7}{'Q(30%)':>8}{'M(25%)':>8}{'V(20%)':>8}{'G(25%)':>8}{'Decel':>7}{'Price':>9}  動作")
        print(f"  {'─'*105}")

        # Check which were traded
        for i, s in enumerate(scores, 1):
            traded = ''
            for td_date, td_action, td_ticker, td_shares, td_price, td_rb in TRADES:
                if td_date == date_str and td_ticker == s['ticker']:
                    emoji = '🟢買' if td_action == 'BUY' else '🔴賣'
                    traded = f"{emoji} {td_shares}股@${td_price:.0f}"
                    break
            price_str = f"${s['price']:.2f}"
            print(f"  {i:<5}{s['ticker']:<8}{s['signal']:<14}{s['score']:>7.3f}"
                  f"{s['quality']:>8.3f}{s['momentum']:>8.3f}{s['value']:>8.3f}{s['growth']:>8.3f}"
                  f"{s['decel']:>+7}{price_str:>9}  {traded}")

if __name__ == '__main__':
    run()
