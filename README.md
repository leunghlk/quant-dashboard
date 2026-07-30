# Quant Dashboard

4-Factor Quant Model Dashboard — PIMCO-style systematic equity selection

🔗 **Live Dashboard**: https://leunghlk.github.io/quant-dashboard/

## Model
- **Quality (30%)**: ROE, Gross Margin, Operating Margin, Leverage, FCF Margin
- **Momentum (25%)**: 12-1M / 6-1M / 3-1M residual momentum + MA50 slope
- **Value (20%)**: Earnings Yield, P/B, EV/EBIT, FCF Yield
- **Growth (25%)**: Revenue Growth, Earnings Growth, Quarterly EPS, Analyst Target

## Strategy
- Monthly rebalance
- Decel ≤ -3 → Force sell (momentum exhaustion)
- Target: +8-10%/month, 4-6 month recovery

Kathy Leung 梁凱菱 · 數據: yfinance · 僅供教學參考
