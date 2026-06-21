# European Stock Screener & Portfolio Backtester

A walk-forward quantitative screening and portfolio construction pipeline for
European equities, benchmarked against the iShares STOXX Europe 600 ETF
(EXW1.DE). Built to practice systematic stock selection and backtesting
methodology for asset management.

## What it does

At every quarterly rebalancing date, the model:

1. Screens a ~120-stock European universe (Germany, France, UK, Switzerland,
   Netherlands, Italy, Spain, Nordics) using **only price history available
   up to that date**
2. Ranks stocks on a composite score across seven factors: CAGR, Sharpe,
   Sortino, Calmar, 12-1 Momentum, Volatility, Max Drawdown
3. Builds three portfolios from the top 35 picks — Equal Weight, Minimum
   Variance, and Max Sharpe (Markowitz, capped at 5% per position) — and
   applies the weights **forward** until the next rebalancing date
4. Deducts 10 bps one-way transaction costs on every weight change
5. Outputs an 11-page PDF report: cover, latest screening table, metrics
   heatmap, equity curves, active return, drawdown, performance summary,
   strategy comparison, rolling Sharpe, turnover, and a closing
   methodology/limitations page

## Results (2019-01-01 → 2026-06-21, after 10 bps TC)

| Strategy  | CAGR % | Vol % | Sharpe | Sortino | Max DD % | Calmar |
|-----------|-------:|------:|-------:|--------:|---------:|-------:|
| EW        |  11.29 | 18.66 |  0.577 |   0.034 |   -40.14 |  0.281 |
| MinVar    |   9.12 | 16.57 |  0.509 |   0.029 |   -37.95 |  0.240 |
| MaxSharpe |  11.66 | 18.22 |  0.604 |   0.035 |   -36.80 |  0.317 |
| Benchmark |  15.44 | 22.96 |  0.663 |   0.046 |   -38.49 |  0.401 |

The benchmark outperforms all three strategies on a risk-adjusted basis.
This is the honest result after fixing a look-ahead bias in the original
implementation (see below) — and it's broadly consistent with the active
management literature: a fixed-cost, quarterly-rebalanced multi-factor
screen on a concentrated, correlated large-cap universe doesn't reliably
beat a cheap, diversified passive benchmark.

## A note on methodology — a bug found during review

An earlier version of this backtest reported CAGR/Sharpe roughly 2-3x
higher than the numbers above. While reviewing the walk-forward loop, a
look-ahead bias was identified: the stock selection made at each
rebalancing date (using data up to and including that date) was being
applied to the quarter that had **just ended** — the same window that had
determined the selection — instead of to the quarter ahead. This let the
backtest retroactively "pick" each quarter's winners after already knowing
the outcome, which inflated every performance metric and artificially
smoothed drawdowns around market stress periods (e.g. the COVID crash).

The fix (`run_backtest()`): weights computed at each rebalancing date are
now applied to the period from that date to the *next* rebalancing date,
not backward to the period that just ended. Verified with an isolated
synthetic test: a stock that underperforms before a given quarter and
spikes only during it is now correctly excluded from that quarter's
simulated return, and only enters the portfolio's realized returns in
subsequent periods. The numbers above reflect the corrected logic.

## Other methodology notes

- **Risk-free rate**: stepped year-by-year approximation (`RF_BY_YEAR`,
  roughly 0% in 2019-21 rising through the 2022-23 hiking cycle) used in
  all Sharpe/Sortino calculations, instead of one fixed rate across a
  period that spans both NIRP and a 4%+ hiking cycle.
- **Known delistings**: explicitly hard-coded names (e.g. Wirecard, WDI.DE,
  insolvency June 2020) are kept in the dataset and marked down to a
  terminal value rather than silently dropped by the data-quality filter —
  a partial fix for survivorship bias.

## Known limitations

- **Residual survivorship bias** — the ~120-ticker starting universe was
  selected from companies that exist today; a fully point-in-time universe
  would require historical index constituent data (e.g. iShares daily
  holdings files), which wasn't available for this project.
- **Price-based factors only** — the original intent was to include
  quality/value factors (EBITDA, ROE, ROIC, P/E). Reliable fundamental data
  for European tickers isn't freely available (`yfinance`'s fundamentals
  are inconsistent and frequently missing outside US large caps); the
  screen is built entirely from price/return-derived metrics instead.
- **Flat transaction costs** — 10 bps applied uniformly; less liquid
  small/mid-cap names likely carry higher real market impact.
- **No capacity constraints** — assumes full notional is tradeable at the
  closing price with no slippage beyond the flat cost.
- **Single market regime** — the backtest period is dominated by a
  post-COVID bull market; performance hasn't been isolated across a full
  bear cycle.

## Tech stack

Python · pandas · numpy · yfinance · matplotlib · seaborn · scipy

## Usage

```bash
pip install pandas numpy yfinance matplotlib seaborn scipy
python european_stock_screener.py
```

Requires internet access to download price data via `yfinance`. Output is
a PDF saved alongside the script.

## Author

Emanuele Pozzani
