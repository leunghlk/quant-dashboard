#!/usr/bin/env python3
"""
Quant Portfolio Dashboard — Light Theme
Generates a comprehensive dashboard with portfolio, plan, watchlist, transaction log.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import warnings, json, os

warnings.filterwarnings('ignore')

OUT = '/Users/leungkathy/quant_model'

# ============================================================
# CONFIG
# ============================================================
USD_HKD = 7.80
TARGET_HKD = 50000

# Current portfolio (BEFORE planned trades — will show both states)
HOLDINGS = {
    'SOXS':  {'shares': 10, 'cost': 64.55},
    'SQQQ':  {'shares': 10, 'cost': 46.675},
    'SKDD':  {'shares': 20, 'cost': 23.00},
}
CASH = 2016.57

# Planned trades
PLAN = [
    {'action':'SELL','ticker':'SOXS','shares':10,'reason':'3x反向ETF，半導體反彈中，止賺出場'},
    {'action':'SELL','ticker':'SQQQ','shares':10,'reason':'3x反向ETF，納指反彈中，止賺出場'},
    {'action':'SELL','ticker':'SKDD','shares':20,'reason':'止蝕-7.7%，釋放現金'},
    {'action':'BUY', 'ticker':'ADBE','shares':6, 'reason':'Q=0.94 V=0.64 Decel=0 Vol=49%，全場最穩防守股'},
    {'action':'BUY', 'ticker':'DDOG','shares':5, 'reason':'Score=0.623 Decel=0 Vol=45%，唯一高分+零減速+低波幅'},
    {'action':'BUY', 'ticker':'LRCX','shares':2, 'reason':'Q=0.97 G=0.97，目標$375(+48%)，半導體反彈後加碼'},
]

# Watchlist (all 14 stocks)
STOCKS = ['AMAT','KLAC','LRCX','INTC','AMD','MRVL','CRDO','BE','GLW','WDAY','DDOG','ADBE','NOW']

# ============================================================
# SCORE FUNCTION (compact)
# ============================================================
def score_stock(t, hist, info):
    if hist is None or len(hist)<60: return None
    c=hist['Close']; n=len(c); price=c.iloc[-1]
    qs={}
    for k,div in [('returnOnEquity',.3),('grossMargins',.5),('operatingMargins',.25)]:
        v=info.get(k)
        if v is not None and not(isinstance(v,float) and np.isnan(v)): qs[k]=min(1.,max(0,v/div))
    de=info.get('debtToEquity')
    if de is not None and not(isinstance(de,float) and np.isnan(de)): qs['de']=max(0,1.-de/200.)
    fcf=info.get('freeCashflow'); rev=info.get('totalRevenue')
    if fcf and rev and rev>0: qs['fcf']=min(1.,max(0,(fcf/rev)/.15))
    q=np.mean(list(qs.values())) if qs else .5
    sk=21
    def mm(d):
        if n>d+sk: return c.iloc[-1-sk]/c.iloc[-d-sk]-1
        elif n>sk+1: return c.iloc[-1-sk]/c.iloc[0]-1
        return 0
    m12,m6,m3=mm(252),mm(126),mm(63)
    m1m=c.iloc[-1]/c.iloc[-22]-1 if n>22 else 0
    m5d=c.iloc[-1]/c.iloc[-6]-1 if n>6 else 0
    ms={'a':np.tanh(m12*2),'b':np.tanh(m6*3),'c':np.tanh(m3*5)}
    if n>=60:
        ma50=c.rolling(50).mean().dropna()
        if len(ma50)>=10: ms['d']=np.tanh((ma50.iloc[-1]/ma50.iloc[-10]-1)*10)
    m=(np.mean(list(ms.values()))+1)/2
    vs={}
    pe=info.get('trailingPE') or info.get('priceToEarnings')
    if pe and 0<pe<200: vs['ey']=min(1.,max(0,(1./pe-.02)/.06))
    pb=info.get('priceToBook')
    if pb and 0<pb<30: vs['pb']=max(0,1.-pb/10.)
    ev=info.get('enterpriseToEbit')
    if ev and 0<ev<80: vs['ev']=max(0,1.-(ev-5)/30.)
    if fcf and info.get('marketCap') and info['marketCap']>0: vs['fy']=min(1.,max(0,(fcf/info['marketCap'])/.05))
    v=np.mean(list(vs.values())) if vs else .5
    gs={}
    for k,div in [('revenueGrowth',.25),('earningsGrowth',.3),('earningsQuarterlyGrowth',.4)]:
        vv=info.get(k)
        if vv is not None and not(isinstance(vv,float) and np.isnan(vv)): gs[k]=min(1.,max(0,vv/div))
    tg=info.get('targetMeanPrice')
    if tg and price>0: gs['an']=min(1.,max(0,(tg/price-1)/.25))
    g=np.mean(list(gs.values())) if gs else .5
    comp=.30*q+.25*m+.20*v+.25*g
    dec=0; dreasons=[]
    if m12>.2 and m1m<0: dec-=2; dreasons.append(f'12M漲{m12*100:.0f}%但1M跌{m1m*100:.1f}%')
    elif m12>.2 and m3<0: dec-=1; dreasons.append(f'12M漲但3M轉跌')
    if m6>.15 and m1m<-.03: dec-=1; dreasons.append(f'6M漲{m6*100:.0f}%但1M急跌{m1m*100:.1f}%')
    if n>=50 and c.iloc[-1]<c.rolling(50).mean().iloc[-1] and m6>.1: dec-=1; dreasons.append('跌破MA50')
    rt=c.pct_change().dropna()
    v30=rt.iloc[-30:].std()*np.sqrt(252) if len(rt)>=30 else .3
    v90=rt.iloc[-90:].std()*np.sqrt(252) if len(rt)>=90 else .3
    if v90>0 and v30/v90>1.3: dec-=1; dreasons.append('波幅擴大')
    if m5d<-.05: dec-=1; dreasons.append(f'5日跌{m5d*100:.1f}%')
    cb=comp
    if dec<=-3: cb-=.15
    elif dec<=-2: cb-=.08
    cb=max(0,min(1,cb))
    if cb<.35: sig='SELL'
    elif cb<.45: sig='REDUCE'
    elif cb<.55: sig='HOLD'
    elif cb<.70: sig='ACCUMULATE'
    else: sig='BUY'
    er=(tg/price-1) if tg and price>0 else 0
    # sector
    sector_map = {'AMAT':'半導體設備','KLAC':'半導體設備','LRCX':'半導體設備','INTC':'晶片','AMD':'晶片','MRVL':'晶片','CRDO':'通訊晶片','BE':'清潔能源','GLW':'特種材料','WDAY':'軟件/HR','DDOG':'軟件/Cyber','ADBE':'軟件','NOW':'軟件/IT'}
    return {'ticker':t,'price':price,'score':cb,'signal':sig,'decel':dec,'dreasons':dreasons,
            'quality':q,'momentum':m,'value':v,'growth':g,
            'm12':m12,'m6':m6,'m3':m3,'m1m':m1m,'m5d':m5d,
            'vol30':v30,'target':tg or 0,'exp_ret':er,
            'sector':sector_map.get(t,'—'),
            'name':t,}

# ============================================================
# FETCH & SCORE
# ============================================================
print("Fetching data...")
stock_scores = []
for t in STOCKS:
    try:
        hist = yf.Ticker(t).history(period="2y")
        info = yf.Ticker(t).info
        s = score_stock(t, hist, info)
        if s: stock_scores.append(s)
    except: pass

stock_scores.sort(key=lambda x: x['score'], reverse=True)
print(f"Scored {len(stock_scores)} stocks")

# Current prices for holdings
holding_prices = {}
for t in HOLDINGS:
    try:
        h = yf.Ticker(t).history(period="5d")
        if len(h): holding_prices[t] = h['Close'].iloc[-1]
    except: pass

# ============================================================
# GENERATE HTML — LIGHT THEME
# ============================================================
def sig_badge(sig):
    colors = {'SELL':'#ef4444','REDUCE':'#f97316','HOLD':'#eab308','ACCUMULATE':'#3b82f6','BUY':'#22c55e'}
    bg = colors.get(sig, '#94a3b8')
    return f'<span style="background:{bg};color:#fff;padding:3px 10px;border-radius:6px;font-size:11px;font-weight:600">{sig}</span>'

def factor_mini(score):
    pct = score * 100
    color = '#22c55e' if score >= 0.65 else '#3b82f6' if score >= 0.45 else '#f97316' if score >= 0.35 else '#ef4444'
    return f'<div style="background:#e2e8f0;height:5px;border-radius:3px;width:40px;display:inline-block;vertical-align:middle"><div style="background:{color};height:5px;border-radius:3px;width:{pct:.0f}%"></div></div><span style="font-size:11px;color:#64748b;margin-left:4px">{pct:.0f}</span>'

def decel_badge(dec):
    if dec <= -3: return '<span style="background:#fee2e2;color:#dc2626;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600">⚠️ EXHAUSTION</span>'
    elif dec <= -2: return '<span style="background:#ffedd5;color:#ea580c;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600">DECEL</span>'
    elif dec <= -1: return '<span style="background:#fef9c3;color:#ca8a04;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600">CAUTION</span>'
    return '<span style="color:#94a3b8;font-size:11px">—</span>'

# Calculate portfolio values
cur_total = CASH
cur_rows = ""
for t, pos in HOLDINGS.items():
    p = holding_prices.get(t, pos['cost'])
    val = pos['shares'] * p
    pnl = (p - pos['cost']) * pos['shares']
    pnl_pct = (p/pos['cost']-1)*100
    cur_total += val
    pc = '#16a34a' if pnl >= 0 else '#dc2626'
    cur_rows += f"""<tr style="border-bottom:1px solid #e2e8f0">
      <td style="padding:10px;font-weight:600;color:#1e293b">{t}</td>
      <td style="padding:10px;text-align:right;color:#64748b">{pos['shares']}</td>
      <td style="padding:10px;text-align:right;color:#64748b">${pos['cost']:.3f}</td>
      <td style="padding:10px;text-align:right;color:#1e293b;font-weight:500">${p:.2f}</td>
      <td style="padding:10px;text-align:right;color:#1e293b">${val:,.2f}</td>
      <td style="padding:10px;text-align:right;color:{pc};font-weight:600">${pnl:+,.2f} ({pnl_pct:+.1f}%)</td>
    </tr>"""

# After-plan portfolio
after_holdings = {'ADBE':6,'DDOG':5,'LRCX':2}
after_prices = {}
for t in after_holdings:
    s = next((x for x in stock_scores if x['ticker']==t), None)
    if s: after_prices[t] = s['price']

# Plan rows
plan_rows = ""
for p in PLAN:
    s = next((x for x in stock_scores if x['ticker']==p['ticker']), None)
    price = s['price'] if s else holding_prices.get(p['ticker'], 0)
    bg = '#fef2f2' if p['action']=='SELL' else '#f0fdf4'
    ac = '#dc2626' if p['action']=='SELL' else '#16a34a'
    score_col = ""
    if s:
        score_col = f"Score {s['score']:.3f} · Decel={s['decel']} · Q={s['quality']:.2f} M={s['momentum']:.2f} V={s['value']:.2f} G={s['growth']:.2f}"
    plan_rows += f"""<tr style="border-bottom:1px solid #e2e8f0;background:{bg}">
      <td style="padding:10px"><span style="background:{ac};color:#fff;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:600">{p['action']}</span></td>
      <td style="padding:10px;font-weight:600;color:#1e293b">{p['ticker']}</td>
      <td style="padding:10px;text-align:right;color:#64748b">{p['shares']}</td>
      <td style="padding:10px;text-align:right;color:#1e293b">${price:.2f}</td>
      <td style="padding:10px;text-align:right;color:#1e293b;font-weight:500">${p['shares']*price:,.2f}</td>
      <td style="padding:10px;font-size:12px;color:#64748b">{p['reason']}</td>
      <td style="padding:10px;font-size:11px;color:#94a3b8">{score_col}</td>
    </tr>"""

# Watchlist rows
watch_rows = ""
for i, s in enumerate(stock_scores, 1):
    pc12 = '#16a34a' if s['m12']>0.05 else '#dc2626' if s['m12']<-0.05 else '#94a3b8'
    pc6 = '#16a34a' if s['m6']>0.05 else '#dc2626' if s['m6']<-0.05 else '#94a3b8'
    pc3 = '#16a34a' if s['m3']>0.05 else '#dc2626' if s['m3']<-0.05 else '#94a3b8'
    pc1 = '#16a34a' if s['m1m']>0.05 else '#dc2626' if s['m1m']<-0.05 else '#94a3b8'
    watch_rows += f"""<tr style="border-bottom:1px solid #e2e8f0" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background=''">
      <td style="padding:8px;text-align:center;color:#94a3b8;font-weight:600">{i}</td>
      <td style="padding:8px"><div style="font-weight:600;color:#1e293b">{s['ticker']}</div><div style="font-size:11px;color:#94a3b8">{s['sector']}</div></td>
      <td style="padding:8px">{sig_badge(s['signal'])}</td>
      <td style="padding:8px"><div style="font-weight:700;font-size:16px;color:{ '#22c55e' if s['score']>=.65 else '#3b82f6' if s['score']>=.45 else '#f97316' if s['score']>=.35 else '#dc2626'}">{s['score']*100:.0f}</div></td>
      <td style="padding:8px">{factor_mini(s['quality'])}</td>
      <td style="padding:8px">{factor_mini(s['momentum'])}</td>
      <td style="padding:8px">{factor_mini(s['value'])}</td>
      <td style="padding:8px">{factor_mini(s['growth'])}</td>
      <td style="padding:8px;text-align:center">{decel_badge(s['decel'])}</td>
      <td style="padding:8px;text-align:right;font-size:12px"><span style="color:{pc12}">{s['m12']*100:+.0f}%</span></td>
      <td style="padding:8px;text-align:right;font-size:12px"><span style="color:{pc6}">{s['m6']*100:+.0f}%</span></td>
      <td style="padding:8px;text-align:right;font-size:12px"><span style="color:{pc3}">{s['m3']*100:+.0f}%</span></td>
      <td style="padding:8px;text-align:right;font-size:12px"><span style="color:{pc1}">{s['m1m']*100:+.0f}%</span></td>
      <td style="padding:8px;text-align:right;color:#1e293b;font-weight:500">${s['price']:.2f}</td>
      <td style="padding:8px;text-align:right;color:#64748b;font-size:12px">{s['vol30']*100:.0f}%</td>
      <td style="padding:8px;text-align:right"><span style="color:{ '#16a34a' if s['exp_ret']>0 else '#dc2626'};font-weight:600">{s['exp_ret']*100:+.0f}%</span></td>
    </tr>"""

now = datetime.now().strftime('%Y-%m-%d %H:%M')
gap_to_target = TARGET_HKD - cur_total * USD_HKD

html = f"""<!DOCTYPE html>
<html lang="zh-HK"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Portfolio Dashboard — Kathy Leung</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box }}
  body {{ background:#f0f4f8; color:#1e293b; font-family:'PingFang HK',-apple-system,'Helvetica Neue',sans-serif; padding:20px }}
  .container {{ max-width:1300px; margin:0 auto }}
  .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:20px }}
  .header h1 {{ font-size:24px; color:#1e293b }}
  .header .sub {{ font-size:13px; color:#64748b; margin-top:2px }}
  .timestamp {{ background:#fff; padding:8px 16px; border-radius:20px; box-shadow:0 1px 3px rgba(0,0,0,.08); font-size:13px; color:#64748b }}
  .card {{ background:#fff; border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:20px; overflow:hidden }}
  .card-header {{ padding:16px 20px; border-bottom:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center }}
  .card-title {{ font-size:16px; font-weight:700; color:#1e293b }}
  .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:0 }}
  .summary-item {{ padding:20px; text-align:center; border-right:1px solid #e2e8f0 }}
  .summary-item:last-child {{ border-right:none }}
  .summary-val {{ font-size:28px; font-weight:700 }}
  .summary-label {{ font-size:12px; color:#64748b; margin-top:4px }}
  .summary-sub {{ font-size:11px; color:#94a3b8; margin-top:2px }}
  table {{ width:100%; border-collapse:collapse }}
  th {{ background:#f8fafc; color:#475569; font-size:11px; font-weight:600; text-transform:uppercase; padding:10px 8px; text-align:left; letter-spacing:.5px }}
  th.right {{ text-align:right }}
  th.center {{ text-align:center }}
  .progress-bar {{ background:#e2e8f0; height:8px; border-radius:4px; overflow:hidden; margin-top:8px }}
  .progress-fill {{ height:8px; border-radius:4px; transition:width .3s }}
  .tab {{ background:#f1f5f9; color:#64748b; border:none; padding:8px 20px; border-radius:6px; cursor:pointer; font-size:13px; font-weight:600 }}
  .tab.active {{ background:#2563eb; color:#fff }}
  .badge {{ padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600 }}
  .tx-form {{ padding:20px }}
  .tx-form input, .tx-form select {{ padding:8px 12px; border:1px solid #cbd5e1; border-radius:6px; font-size:13px }}
  .tx-form button {{ padding:8px 20px; background:#2563eb; color:#fff; border:none; border-radius:6px; cursor:pointer; font-size:13px; font-weight:600 }}
</style>
</head><body><div class="container">

<!-- HEADER -->
<div class="header">
  <div>
    <h1>📈 Quant Portfolio Dashboard</h1>
    <div class="sub">4-Factor Model · Quality 30% · Momentum 25% · Value 20% · Growth 25% · Kathy Leung 梁凱菱</div>
  </div>
  <div class="timestamp">📅 {now}</div>
</div>

<!-- SUMMARY CARDS -->
<div class="card">
  <div class="summary-grid">
    <div class="summary-item">
      <div class="summary-val" style="color:#1e293b">${cur_total:,.2f}</div>
      <div class="summary-label">當前組合總值</div>
      <div class="summary-sub">HKD ${cur_total*USD_HKD:,.0f}</div>
    </div>
    <div class="summary-item">
      <div class="summary-val" style="color:#dc2626">HKD ${gap_to_target:,.0f}</div>
      <div class="summary-label">距回本差距</div>
      <div class="summary-sub">目標 HKD $50,000</div>
    </div>
    <div class="summary-item">
      <div class="summary-val" style="color:#2563eb">{(cur_total*USD_HKD/TARGET_HKD)*100:.0f}%</div>
      <div class="summary-label">回本進度</div>
      <div class="progress-bar" style="margin-top:8px"><div class="progress-fill" style="background:#2563eb;width:{(cur_total*USD_HKD/TARGET_HKD)*100:.0f}%"></div></div>
    </div>
    <div class="summary-item">
      <div class="summary-val" style="color:#64748b">4-6 個月</div>
      <div class="summary-label">理性回本預期</div>
      <div class="summary-sub">每月目標 +8-10%</div>
    </div>
    <div class="summary-item">
      <div class="summary-val" style="color:#f97316">⚠️ 反彈中</div>
      <div class="summary-label">半導體板塊狀態</div>
      <div class="summary-sub">Decel=-5 但超賣反彈</div>
    </div>
  </div>
</div>

<!-- CURRENT PORTFOLIO -->
<div class="card">
  <div class="card-header">
    <span class="card-title">📂 目前持倉</span>
    <span style="color:#94a3b8;font-size:12px">成本基礎 ${sum(p['shares']*p['cost'] for p in HOLDINGS.values())+CASH:,.2f}</span>
  </div>
  <table>
    <thead><tr>
      <th>股票</th><th class="right">股數</th><th class="right">成本</th><th class="right">現價</th>
      <th class="right">市值</th><th class="right">盈虧</th>
    </tr></thead>
    <tbody>{cur_rows}
      <tr style="border-bottom:1px solid #e2e8f0">
        <td style="padding:10px;color:#64748b">現金</td>
        <td></td><td></td><td></td>
        <td style="padding:10px;text-align:right;color:#1e293b">${CASH:,.2f}</td>
        <td></td>
      </tr>
      <tr style="background:#f8fafc;font-weight:700">
        <td style="padding:10px">總計</td><td></td><td></td><td></td>
        <td style="padding:10px;text-align:right;color:#1e293b">${cur_total:,.2f}</td>
        <td style="padding:10px;text-align:right;color:{'#16a34a' if cur_total>sum(p['shares']*p['cost'] for p in HOLDINGS.values())+CASH else '#dc2626'}">HKD ${cur_total*USD_HKD:,.0f}</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- ACTION PLAN -->
<div class="card">
  <div class="card-header">
    <span class="card-title">🎯 執行計劃（盤前沽 ETF → 開市買個股）</span>
    <span class="badge" style="background:#fef3c7;color:#92400e">待執行</span>
  </div>
  <div style="padding:12px 20px;background:#fffbeb;font-size:13px;color:#92400e;line-height:1.6">
    ⚡ <b>盤前立即沽</b> SOXS / SQQQ / SKDD（半導體反彈中，3x ETF 每分鐘失血）→ <b>9:30 ET 開市後買入</b> ADBE / DDOG / LRCX（限價單）
  </div>
  <table>
    <thead><tr>
      <th>動作</th><th>股票</th><th class="right">股數</th><th class="right">價位</th>
      <th class="right">金額</th><th>原因</th><th>模型評分</th>
    </tr></thead>
    <tbody>{plan_rows}</tbody>
  </table>
</div>

<!-- WATCHLIST -->
<div class="card">
  <div class="card-header">
    <span class="card-title">📊 觀察名單 — 14 隻股票實時評分</span>
    <span style="color:#94a3b8;font-size:12px">按總分排序 · 每日更新</span>
  </div>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th class="center">#</th><th>股票</th><th>信號</th><th class="center">總分</th>
      <th>Quality</th><th>Momentum</th><th>Value</th><th>Growth</th>
      <th class="center">Decel</th>
      <th class="right">12M</th><th class="right">6M</th><th class="right">3M</th><th class="right">1M</th>
      <th class="right">現價</th><th class="right">Vol30</th><th class="right">上行</th>
    </tr></thead>
    <tbody>{watch_rows}</tbody>
  </table>
  </div>
</div>

<!-- TRANSACTION LOG (input + history) -->
<div class="card">
  <div class="card-header">
    <span class="card-title">📝 交易紀錄</span>
    <span style="color:#94a3b8;font-size:12px">手動輸入買入/沽出</span>
  </div>
  <div class="tx-form">
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:end">
      <div><label style="font-size:11px;color:#64748b;display:block;margin-bottom:4px">日期</label><input type="date" id="txDate" value="{datetime.now().strftime('%Y-%m-%d')}"></div>
      <div><label style="font-size:11px;color:#64748b;display:block;margin-bottom:4px">動作</label><select id="txAction"><option>BUY</option><option>SELL</option></select></div>
      <div><label style="font-size:11px;color:#64748b;display:block;margin-bottom:4px">股票</label><input type="text" id="txTicker" placeholder="e.g. ADBE" style="width:80px"></div>
      <div><label style="font-size:11px;color:#64748b;display:block;margin-bottom:4px">股數</label><input type="number" id="txShares" placeholder="0" style="width:70px"></div>
      <div><label style="font-size:11px;color:#64748b;display:block;margin-bottom:4px">價格</label><input type="number" id="txPrice" placeholder="0.00" step="0.01" style="width:90px"></div>
      <div><label style="font-size:11px;color:#64748b;display:block;margin-bottom:4px">備註</label><input type="text" id="txNote" placeholder="原因..." style="width:200px"></div>
      <button onclick="addTx()">+ 新增交易</button>
      <button onclick="exportTx()" style="background:#64748b">匯出 JSON</button>
    </div>
    <div id="txLog" style="margin-top:20px"></div>
  </div>
</div>

<!-- STRATEGY -->
<div class="card">
  <div class="card-header"><span class="card-title">📋 策略概覽</span></div>
  <div style="padding:20px;font-size:13px;line-height:1.8;color:#475569">
    <div style="margin-bottom:12px"><b style="color:#dc2626">階段一（Week 1-2）：清倉 ETF → 建防守倉</b><br>
    盤前沽 SOXS/SQQQ/SKDD，開市買 ADBE(44%) + DDOG(37%) + LRCX(14%)。平均 Vol54%，Decel=0，用低波幅熬過震盪。</div>
    <div style="margin-bottom:12px"><b style="color:#2563eb">階段二（Month 1-3）：反彈加碼</b><br>
    半導體 Decel 從 -5 回升至 ≥-2 時，加碼 LRCX +3股、AMAT 2股、INTC 10股。每月再平衡一次。</div>
    <div style="margin-bottom:12px"><b style="color:#16a34a">階段三（Month 4-6）：收成</b><br>
    目標股到分析員目標價分批套利。Decel≤-3 必沽。預期每月 +8-10%，4-6 個月回本。</div>
    <div style="padding:10px;background:#fef2f2;border-radius:8px;border-left:3px solid #dc2626">
    <b>鐵律</b>：Decel≤-3 必沽 · 單股不超40% · 保留5%現金 · 不用槓桿ETF · 月度再平衡</div>
  </div>
</div>

<div style="text-align:center;color:#94a3b8;font-size:11px;padding:10px">
  Kathy Leung 梁凱菱 · 4-Factor Quant Model · 數據: yfinance · 僅供教學參考
</div>

</div>

<script>
let transactions = [];

function addTx() {{
  const d = document.getElementById('txDate').value;
  const a = document.getElementById('txAction').value;
  const t = document.getElementById('txTicker').value.toUpperCase();
  const s = parseInt(document.getElementById('txShares').value);
  const p = parseFloat(document.getElementById('txPrice').value);
  const n = document.getElementById('txNote').value;
  if (!t || !s || !p) {{ alert('請填寫股票、股數、價格'); return; }}
  transactions.push({{date:d, action:a, ticker:t, shares:s, price:p, note:n}});
  renderTx();
  // Clear
  document.getElementById('txTicker').value = '';
  document.getElementById('txShares').value = '';
  document.getElementById('txPrice').value = '';
  document.getElementById('txNote').value = '';
}}

function delTx(i) {{
  transactions.splice(i, 1);
  renderTx();
}}

function renderTx() {{
  const log = document.getElementById('txLog');
  if (!transactions.length) {{
    log.innerHTML = '<div style="color:#94a3b8;font-size:13px;text-align:center;padding:20px">尚無交易紀錄。執行買賣後在此輸入。</div>';
    return;
  }}
  let html = '<table style="width:100%;border-collapse:collapse"><thead><tr>' +
    '<th style="padding:8px">日期</th><th>動作</th><th>股票</th><th class="right">股數</th>' +
    '<th class="right">價格</th><th class="right">金額</th><th>備註</th><th></th></tr></thead><tbody>';
  transactions.forEach((t, i) => {{
    const ac = t.action === 'BUY' ? '#16a34a' : '#dc2626';
    const val = t.shares * t.price;
    html += '<tr style="border-bottom:1px solid #e2e8f0">' +
      '<td style="padding:8px;font-size:13px">'+t.date+'</td>' +
      '<td style="padding:8px"><span style="background:'+ac+';color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">'+t.action+'</span></td>' +
      '<td style="padding:8px;font-weight:600">'+t.ticker+'</td>' +
      '<td style="padding:8px;text-align:right">'+t.shares+'</td>' +
      '<td style="padding:8px;text-align:right">$'+t.price.toFixed(2)+'</td>' +
      '<td style="padding:8px;text-align:right;font-weight:600">$'+val.toLocaleString(undefined,{{minimumFractionDigits:2}})+'</td>' +
      '<td style="padding:8px;font-size:12px;color:#64748b">'+(t.note||'')+'</td>' +
      '<td style="padding:8px"><button onclick="delTx('+i+')" style="background:none;border:none;color:#dc2626;cursor:pointer;font-size:16px">×</button></td>' +
      '</tr>';
  }});
  html += '</tbody></table>';
  log.innerHTML = html;
}}

function exportTx() {{
  if (!transactions.length) {{ alert('沒有交易可匯出'); return; }}
  const blob = new Blob([JSON.stringify(transactions, null, 2)], {{type:'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'transactions.json'; a.click();
}}

renderTx();
</script>
</body></html>"""

path = os.path.join(OUT, 'portfolio_dashboard.html')
with open(path, 'w') as f:
    f.write(html)
print(f"Dashboard: {path}")
