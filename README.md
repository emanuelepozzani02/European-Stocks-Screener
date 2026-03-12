# European-Stocks-Screener
# 🇪🇺 European Stock Screener & Portfolio Backtest

A walk-forward stock screener and portfolio backtesting engine built on a STOXX Europe 600 representative universe. Screens ~120 large-cap European equities each quarter using seven risk-adjusted metrics, then constructs and compares three portfolio strategies against a passive benchmark — after realistic transaction costs.


---

## Results at a Glance

Performance over the full walk-forward period (2019–2026), after 10 bps one-way transaction costs:

| Strategy | CAGR | Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| **Equal Weight** | 29.89% | 17.19% | 1.44 | -27.18% | 1.10 |
| **Min Variance** | 26.48% | 14.45% | 1.49 | -23.39% | 1.13 |
| **Max Sharpe** | 31.08% | 16.63% | 1.53 | -24.06% | 1.29 |
| Benchmark (STOXX 600 ETF) | 13.76% | 21.15% | 0.57 | -38.49% | 0.36 |

All three strategies outperform the benchmark on every risk-adjusted metric. Min Variance achieves the lowest drawdown (-23.39%) and best Sharpe among the optimised strategies — consistent with its defensive design.

---

## How It Works

### 1. Universe
~120 large-cap stocks across Germany, France, UK, Switzerland, Netherlands, Italy, Spain, and the Nordics — broadly representative of the STOXX Europe 600.

### 2. Walk-forward screening (no look-ahead bias)
At each quarterly rebalancing date, the screener evaluates every stock using **only data available up to that point**. No future information leaks into the selection process. Each stock is scored on seven factors:

| Factor | Direction |
|---|---|
| CAGR | ↑ higher is better |
| Sharpe ratio | ↑ |
| Sortino ratio | ↑ |
| Calmar ratio | ↑ |
| Momentum 12-1 | ↑ (12-month return, skipping last month) |
| Volatility | ↓ lower is better |
| Max Drawdown | ↓ |

Factors are z-scored and averaged into a composite score. The top 35 stocks are selected each quarter.

### 3. Portfolio construction
Three weighting strategies are compared:

- **Equal Weight (EW)** — 1/N, no assumptions, maximum naive diversification
- **Min Variance** — minimises portfolio volatility using the historical covariance matrix
- **Max Sharpe (Markowitz)** — maximises the risk-adjusted return on the efficient frontier

Both optimised strategies are subject to a **5% maximum weight per stock** to prevent concentration and improve out-of-sample robustness.

### 4. Transaction costs
At every rebalancing, the model computes the exact portfolio turnover:
- Stocks entering or exiting: full weight traded
- Stocks remaining: `|Δw|` traded

A **10 bps one-way cost** is applied as a NAV drag on the rebalancing day.

### 5. Benchmark
Buy-and-hold of the **iShares STOXX Europe 600 UCITS ETF (EXW1.DE)**, normalised to the same starting capital, used as the passive reference throughout.

---

## Project Structure

```
European_Stocks_Screener.py    # main script
European_Stocks_Screener.pdf   # latest backtest report
requirements.txt               # dependencies
```

---

## Getting Started

```bash
pip install -r requirements.txt
python European_Stocks_Screener.py
```

The script downloads live price data via `yfinance` and outputs a multi-page PDF report to the same directory. Runtime is approximately 2–4 minutes depending on connection speed.

---

## Requirements

```
yfinance
numpy
pandas
matplotlib
seaborn
scipy
```

---

## Known Limitations

- **Covariance estimation** — the sample covariance matrix is used without shrinkage (e.g. Ledoit-Wolf). With ~35 assets this is generally stable, but shrinkage would improve out-of-sample optimiser behaviour.
- **Survivorship bias** — the universe is fixed at today's constituents. Companies delisted or bankrupt since 2019 are excluded, which modestly flatters results.
- **Transaction cost model** — 10 bps is a reasonable approximation for liquid large-caps, but does not model bid-ask spread or market impact for larger position sizes.
- **Single time period** — the backtest covers one macro cycle (2019–2026). Results over different regimes (e.g. 2000–2010) may differ materially.

---

## Author: Emanuele Pozzani

Built as part of a personal research project on quantitative European equity strategies.
