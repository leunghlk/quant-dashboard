#!/usr/bin/env python3
"""
4-Factor Quant Model — Full Trading Simulation
================================================
Capital: HKD 50,000 (≈ USD 6,410 @ 7.80)
Period: 2026-01-01 to 2026-07-29
Universe: AMAT, KLAC, LRCX, INTC, AMD, MRVL, CRDO, BE, GLW
Rebalance: Monthly (score at month-end, trade next day open)

Strategy:
  - BUY signal: Deploy capital (equal weight across all BUY+ACCUMULATE stocks)
  - Decel ≤ -3: Force sell (momentum exhaustion override)
  - SELL/REDUCE: Exit immediately
  - HOLD: Keep if already holding, don't buy new
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings, json

warnings.filterwarnings('ignore')

TICKERS = ['AMAT', 'KLAC', 'LRCX', 'INTC', 'AMD', 'MRVL', 'CRDO', 'BE', 'GLW']
USD_HKD = 7.80
INITIAL_CAPITAL_HKD = 50000
INITIAL_CAPITAL_USD = INITIAL_CAPITAL_HKD / USD_HKD

# ============================================================
# FACTOR SCORING (point-in-time)
# ============================================================
def score_at_date(hist_full, info_cache, ticker, eval_date):
    """Score stock using only data up to eval_date."""
    # Handle tz-aware index
    ts = pd.Timestamp(eval_date)
    if hist_full.index.tz is not None:
        if ts.tz is None:
            ed = ts.tz_localize(hist_full.index.tz)
        else:
            ed = ts.tz_convert(hist_full.index.tz)
    else:
        ed = ts.tz_localize(None) if ts.tz else ts
    
    hist = hist_full[hist_full.index <= ed].copy()
    if len(hist) < 60:
        return None
    
    closes = hist['Close']
    n = len(closes)
    price = closes.iloc[-1]
    info = info_cache.get(ticker, {})
    
    # === QUALITY (30%) ===
    qs = {}
    roe = info.get('returnOnEquity')
    if roe is not None and not (isinstance(roe, float) and np.isnan(roe)):
        qs['roe'] = min(1.0, max(0, roe / 0.30))
    gm = info.get('grossMargins')
    if gm is not None and not (isinstance(gm, float) and np.isnan(gm)):
        qs['gm'] = min(1.0, max(0, gm / 0.50))
    om = info.get('operatingMargins')
    if om is not None and not (isinstance(om, float) and np.isnan(om)):
        qs['om'] = min(1.0, max(0, om / 0.25))
    de = info.get('debtToEquity')
    if de is not None and not (isinstance(de, float) and np.isnan(de)):
        qs['de'] = max(0, 1.0 - de / 200.0)
    fcf = info.get('freeCashflow'); rev = info.get('totalRevenue')
    if fcf and rev and rev > 0:
        qs['fcf'] = min(1.0, max(0, (fcf/rev) / 0.15))
    quality = np.mean(list(qs.values())) if qs else 0.5
    
    # === MOMENTUM (25%) — point-in-time ===
    skip = 21
    def mom(days):
        if n > days + skip:
            return closes.iloc[-1-skip] / closes.iloc[-days-skip] - 1
        elif n > skip + 1:
            return closes.iloc[-1-skip] / closes.iloc[0] - 1
        return 0
    
    m12, m6, m3 = mom(252), mom(126), mom(63)
    ms = {'12': np.tanh(m12*2), '6': np.tanh(m6*3), '3': np.tanh(m3*5)}
    if n >= 60:
        ma50 = closes.rolling(50).mean().dropna()
        if len(ma50) >= 10:
            ms['slope'] = np.tanh((ma50.iloc[-1]/ma50.iloc[-10]-1)*10)
    momentum = (np.mean(list(ms.values())) + 1) / 2
    
    # === VALUE (20%) ===
    vs = {}
    pe = info.get('trailingPE') or info.get('priceToEarnings')
    if pe and pe > 0 and pe < 200:
        vs['ey'] = min(1.0, max(0, (1.0/pe - 0.02) / 0.06))
    pb = info.get('priceToBook')
    if pb and pb > 0 and pb < 30:
        vs['pb'] = max(0, 1.0 - pb/10.0)
    ev_ebit = info.get('enterpriseToEbit')
    if ev_ebit and ev_ebit > 0 and ev_ebit < 80:
        vs['ev'] = max(0, 1.0 - (ev_ebit-5)/30.0)
    if fcf and info.get('marketCap') and info['marketCap'] > 0:
        vs['fcfy'] = min(1.0, max(0, (fcf/info['marketCap']) / 0.05))
    value = np.mean(list(vs.values())) if vs else 0.5
    
    # === GROWTH (25%) ===
    gs = {}
    rg = info.get('revenueGrowth')
    if rg is not None and not (isinstance(rg, float) and np.isnan(rg)):
        gs['rev'] = min(1.0, max(0, rg/0.25))
    eg = info.get('earningsGrowth')
    if eg is not None and not (isinstance(eg, float) and np.isnan(eg)):
        gs['earn'] = min(1.0, max(0, eg/0.30))
    eqg = info.get('earningsQuarterlyGrowth')
    if eqg is not None and not (isinstance(eqg, float) and np.isnan(eqg)):
        gs['qtr'] = min(1.0, max(0, eqg/0.40))
    target = info.get('targetMeanPrice')
    if target and price > 0:
        gs['analyst'] = min(1.0, max(0, (target/price-1)/0.25))
    growth = np.mean(list(gs.values())) if gs else 0.5
    
    composite = 0.30*quality + 0.25*momentum + 0.20*value + 0.25*growth
    
    # === DECELERATION ===
    m1m = closes.iloc[-1] / closes.iloc[-22] - 1 if n > 22 else 0
    m5d = closes.iloc[-1] / closes.iloc[-6] - 1 if n > 6 else 0
    decel = 0
    if m12 > 0.20 and m1m < 0: decel -= 2
    elif m12 > 0.20 and m3 < 0: decel -= 1
    if m6 > 0.15 and m1m < -0.03: decel -= 1
    if n >= 50 and closes.iloc[-1] < closes.rolling(50).mean().iloc[-1] and m6 > 0.10: decel -= 1
    ret = closes.pct_change().dropna()
    vol30 = ret.iloc[-30:].std()*np.sqrt(252) if len(ret) >= 30 else 0.30
    vol90 = ret.iloc[-90:].std()*np.sqrt(252) if len(ret) >= 90 else 0.30
    if vol90 > 0 and vol30/vol90 > 1.3: decel -= 1
    if m5d < -0.05: decel -= 1
    
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
        'ticker': ticker, 'price': price, 'score': combined,
        'signal': signal, 'decel': decel,
        'quality': quality, 'momentum': momentum, 'value': value, 'growth': growth,
        'm12': m12, 'm6': m6, 'm3': m3, 'm1m': m1m,
    }

# ============================================================
# TRADING ENGINE
# ============================================================
class Portfolio:
    def __init__(self, capital_usd):
        self.cash = capital_usd
        self.positions = {}  # ticker -> {'shares': int, 'avg_cost': float}
        self.transactions = []
        self.initial_capital = capital_usd
    
    def buy(self, ticker, price, date, max_usd=None):
        available = max_usd if max_usd else self.cash
        if available < price:
            return 0
        shares = int(available / price)
        if shares <= 0:
            return 0
        cost = shares * price
        self.cash -= cost
        if ticker in self.positions:
            old = self.positions[ticker]
            total_shares = old['shares'] + shares
            total_cost = old['shares'] * old['avg_cost'] + cost
            self.positions[ticker] = {'shares': total_shares, 'avg_cost': total_cost/total_shares}
        else:
            self.positions[ticker] = {'shares': shares, 'avg_cost': price}
        self.transactions.append({
            'date': date, 'action': 'BUY', 'ticker': ticker,
            'shares': shares, 'price': price, 'value': cost,
        })
        return shares
    
    def sell(self, ticker, price, date, shares=None):
        if ticker not in self.positions:
            return 0
        pos = self.positions[ticker]
        sell_shares = shares if shares else pos['shares']
        if sell_shares <= 0:
            return 0
        proceeds = sell_shares * price
        self.cash += proceeds
        pnl = (price - pos['avg_cost']) * sell_shares
        self.positions[ticker]['shares'] -= sell_shares
        if self.positions[ticker]['shares'] <= 0:
            del self.positions[ticker]
        self.transactions.append({
            'date': date, 'action': 'SELL', 'ticker': ticker,
            'shares': sell_shares, 'price': price, 'value': proceeds, 'pnl': pnl,
        })
        return sell_shares
    
    def total_value(self, prices):
        val = self.cash
        for ticker, pos in self.positions.items():
            if ticker in prices:
                val += pos['shares'] * prices[ticker]
        return val

def get_trading_day(hist_all, target_date, direction='on_or_after'):
    """Find nearest trading day."""
    all_dates = set()
    for h in hist_all.values():
        for d in h.index:
            all_dates.add(d.normalize())
    sorted_dates = sorted(all_dates)
    target = pd.Timestamp(target_date)
    if sorted_dates[0].tz is not None:
        target = target.tz_localize(sorted_dates[0].tz)
    
    for d in sorted_dates:
        if direction == 'on_or_after' and d >= target:
            return d
        elif direction == 'on_or_before' and d <= target:
            best = d
    if direction == 'on_or_before':
        return best
    return sorted_dates[-1]

def get_price(hist, date):
    """Get close price on or before date."""
    if hist.index.tz is not None:
        d = pd.Timestamp(date).tz_localize(hist.index.tz) if not isinstance(date, pd.Timestamp) else date
    else:
        d = pd.Timestamp(date)
    subset = hist[hist.index <= d]
    if len(subset) > 0:
        return subset['Close'].iloc[-1]
    return None

# ============================================================
# RUN SIMULATION
# ============================================================
def run():
    print("=" * 100)
    print(" 4-FACTOR QUANT MODEL — TRADING SIMULATION")
    print(f" Capital: HKD {INITIAL_CAPITAL_HKD:,.0f} (USD {INITIAL_CAPITAL_USD:,.2f} @ {USD_HKD})")
    print(f" Period: 2026-01-01 → 2026-07-29")
    print(f" Stocks: {', '.join(TICKERS)}")
    print("=" * 100)
    
    # Fetch data
    print("\nFetching data...")
    hist_all = {}
    info_cache = {}
    for t in TICKERS:
        hist_all[t] = yf.Ticker(t).history(start="2024-06-01", end="2026-12-31")
        try:
            info_cache[t] = yf.Ticker(t).info
        except:
            info_cache[t] = {}
        print(f"  {t}: {len(hist_all[t])} bars")
    
    # Rebalance dates (last trading day of each month)
    rebalance_targets = [
        '2026-01-02',  # First trading day of Jan (entry)
        '2026-02-02',  # Feb rebalance
        '2026-03-02',  # Mar
        '2026-04-01',  # Apr
        '2026-05-01',  # May
        '2026-06-01',  # Jun
        '2026-07-01',  # Jul
        '2026-07-29',  # Final valuation
    ]
    
    rebalance_dates = []
    for target in rebalance_targets:
        td = get_trading_day(hist_all, target, 'on_or_after')
        rebalance_dates.append(td)
    
    portfolio = Portfolio(INITIAL_CAPITAL_USD)
    monthly_scores = {}
    
    for ri, rb_date in enumerate(rebalance_dates):
        date_str = rb_date.strftime('%Y-%m-%d')
        is_final = (ri == len(rebalance_dates) - 1)
        
        print(f"\n{'─'*100}")
        if is_final:
            print(f" 📅 FINAL VALUATION: {date_str}")
        else:
            print(f" 📅 REBALANCE #{ri+1}: {date_str}")
        print(f"{'─'*100}")
        
        # Score all stocks at this date
        scores = []
        for ticker in TICKERS:
            s = score_at_date(hist_all[ticker], info_cache, ticker, rb_date)
            if s:
                s['date'] = date_str
                scores.append(s)
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        monthly_scores[date_str] = scores
        
        # Print scores
        print(f"\n  {'Ticker':<8} {'Signal':<14} {'Score':>6} {'Decel':>6} {'Price':>9}  {'Q':>5} {'M':>5} {'V':>5} {'G':>5}")
        print(f"  {'─'*80}")
        for s in scores:
            print(f"  {s['ticker']:<8} {s['signal']:<14} {s['score']:.3f}  {s['decel']:>+5}  ${s['price']:>8.2f}  "
                  f"{s['quality']:.2f} {s['momentum']:.2f} {s['value']:.2f} {s['growth']:.2f}")
        
        if is_final:
            # Just value the portfolio
            current_prices = {s['ticker']: s['price'] for s in scores}
            total_val = portfolio.total_value(current_prices)
            break
        
        # === TRADING LOGIC ===
        
        # Step 1: Check sells (force sell on decel <= -3, or signal SELL/REDUCE)
        to_sell = []
        for s in scores:
            ticker = s['ticker']
            if ticker in portfolio.positions:
                if s['signal'] in ('SELL', 'REDUCE'):
                    to_sell.append(('SIGNAL', s))
                elif s['decel'] <= -3:
                    to_sell.append(('DECEL', s))
        
        for reason, s in to_sell:
            ticker = s['ticker']
            shares = portfolio.sell(ticker, s['price'], date_str)
            print(f"\n  🔴 SELL [{reason}] {ticker}: {shares} shares @ ${s['price']:.2f} = ${shares*s['price']:,.2f}")
            if 'pnl' in portfolio.transactions[-1]:
                print(f"      P&L: ${portfolio.transactions[-1]['pnl']:+,.2f}")
        
        # Step 2: Check buys (BUY or ACCUMULATE)
        buy_candidates = [s for s in scores if s['signal'] in ('BUY', 'ACCUMULATE') and s['decel'] > -3]
        
        if buy_candidates:
            # How much to deploy?
            # If first trade: deploy all cash equally
            # If rebalance: deploy 80% of available cash to new/accumulate positions
            if not portfolio.positions and not portfolio.transactions:
                deploy_pct = 1.0  # Full deployment on first trade
            else:
                deploy_pct = 0.80
            
            available = portfolio.cash * deploy_pct
            per_stock = available / len(buy_candidates) if buy_candidates else 0
            
            print(f"\n  💰 Cash: ${portfolio.cash:,.2f} | Deploy: {deploy_pct*100:.0f}% = ${available:,.2f} | Per stock: ${per_stock:,.2f}")
            
            for s in buy_candidates:
                ticker = s['ticker']
                if ticker in portfolio.positions:
                    # Already holding — check if we should add
                    current_val = portfolio.positions[ticker]['shares'] * s['price']
                    if current_val < per_stock * 0.7:
                        # Underweight — top up
                        add_usd = min(per_stock - current_val, portfolio.cash * 0.9)
                        if add_usd > s['price']:
                            shares = portfolio.buy(ticker, s['price'], date_str, max_usd=add_usd)
                            if shares > 0:
                                print(f"  🔵 ADD {ticker}: +{shares} shares @ ${s['price']:.2f} = ${shares*s['price']:,.2f}")
                else:
                    # New position
                    shares = portfolio.buy(ticker, s['price'], date_str, max_usd=per_stock)
                    if shares > 0:
                        print(f"  🟢 BUY {ticker}: {shares} shares @ ${s['price']:.2f} = ${shares*s['price']:,.2f}")
                    else:
                        print(f"  ⚠ SKIP {ticker}: price ${s['price']:.2f} > allocation ${per_stock:,.2f}")
        
        # Portfolio status
        current_prices = {s['ticker']: s['price'] for s in scores}
        total_val = portfolio.total_value(current_prices)
        pnl_pct = (total_val / INITIAL_CAPITAL_USD - 1) * 100
        positions_str = ', '.join(f"{t}:{p['shares']}sh@${p['avg_cost']:.2f}" for t,p in portfolio.positions.items())
        print(f"\n  📊 Portfolio Value: ${total_val:,.2f} (HKD ${total_val*USD_HKD:,.0f}) | P&L: {pnl_pct:+.1f}% | Cash: ${portfolio.cash:,.2f}")
        if positions_str:
            print(f"  📦 Holdings: {positions_str}")
    
    # === FINAL RESULTS ===
    final_prices = {}
    for t in TICKERS:
        p = get_price(hist_all[t], rebalance_dates[-1])
        if p: final_prices[t] = p
    
    total_val = portfolio.total_value(final_prices)
    total_pnl_usd = total_val - INITIAL_CAPITAL_USD
    total_pnl_hkd = total_pnl_usd * USD_HKD
    pnl_pct = (total_val / INITIAL_CAPITAL_USD - 1) * 100
    
    print(f"\n{'='*100}")
    print(f" FINAL RESULTS (as of 2026-07-29)")
    print(f"{'='*100}")
    print(f" Initial Capital:  USD ${INITIAL_CAPITAL_USD:,.2f} (HKD ${INITIAL_CAPITAL_HKD:,.0f})")
    print(f" Final Value:      USD ${total_val:,.2f} (HKD ${total_val*USD_HKD:,.0f})")
    print(f" P&L:               USD ${total_pnl_usd:+,.2f} (HKD ${total_pnl_hkd:+,.0f})")
    print(f" Return:            {pnl_pct:+.1f}%")
    print(f" Cash remaining:    USD ${portfolio.cash:,.2f}")
    
    if portfolio.positions:
        print(f"\n Open Positions:")
        for ticker, pos in portfolio.positions.items():
            cur_price = final_prices.get(ticker, 0)
            val = pos['shares'] * cur_price
            unrealized = (cur_price - pos['avg_cost']) * pos['shares']
            print(f"  {ticker}: {pos['shares']} shares | cost ${pos['avg_cost']:.2f} | now ${cur_price:.2f} | "
                  f"value ${val:,.2f} | unrealized ${unrealized:+,.2f}")
    
    # === TRANSACTION HISTORY ===
    print(f"\n{'='*100}")
    print(f" TRANSACTION HISTORY ({len(portfolio.transactions)} trades)")
    print(f"{'='*100}")
    print(f"  {'Date':<12} {'Action':<6} {'Ticker':<8} {'Shares':>7} {'Price':>9} {'Value (USD)':>13} {'P&L':>10}")
    print(f"  {'─'*80}")
    
    for t in portfolio.transactions:
        pnl_str = f"${t.get('pnl', 0):+,.2f}" if 'pnl' in t else '—'
        print(f"  {t['date']:<12} {t['action']:<6} {t['ticker']:<8} {t['shares']:>7} ${t['price']:>8.2f} ${t['value']:>12,.2f} {pnl_str:>10}")
    
    # Summary stats
    buys = [t for t in portfolio.transactions if t['action'] == 'BUY']
    sells = [t for t in portfolio.transactions if t['action'] == 'SELL']
    realized_pnl = sum(t.get('pnl', 0) for t in sells)
    total_invested = sum(t['value'] for t in buys)
    total_proceeds = sum(t['value'] for t in sells)
    
    print(f"\n  Trades: {len(buys)} buys, {len(sells)} sells")
    print(f"  Total Invested:  ${total_invested:,.2f}")
    print(f"  Total Proceeds:  ${total_proceeds:,.2f}")
    print(f"  Realized P&L:    ${realized_pnl:+,.2f}")
    
    return portfolio, monthly_scores, total_val

if __name__ == '__main__':
    portfolio, monthly_scores, final_val = run()
