# =============================================================================
#  EUROPEAN STOCK SCREENER & PORTFOLIO BACKTEST  —  v4  (walk-forward)
#  Strategies : Equal Weight | Min Variance | Max Sharpe (Markowitz)
#  Benchmark  : iShares STOXX Europe 600 ETF (EXW1.DE)
#  Costs      : 10 bps one-way per trade (applied at each rebalancing)
#  Selection  : Rolling walk-forward — top N stocks re-screened each quarter
#               using ONLY data available up to that point (no look-ahead bias)
#  Constraints: Markowitz optimisers capped at 5% max weight per stock
#
#  Screening factors:
#    CAGR · Volatility · Sharpe · Sortino · Max Drawdown · Calmar · Momentum 12-1
#
#  Author : [Your Name]
#  GitHub : [Your GitHub Link]
# =============================================================================

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime

# =============================================================================
#  0. CONFIGURATION
# =============================================================================

CONFIG = {
    "start_date"        : "2019-01-01",
    "end_date"          : datetime.today().strftime("%Y-%m-%d"),
    "rebalance_freq"    : "Q",       # Q=quarterly  M=monthly
    "top_n_stocks"      : 35,        # stocks selected each rebalancing date
    "risk_free_rate"    : 0.03,      # annual
    "min_history_days"  : 252,       # minimum days of history to screen a stock
    "min_history_frac"  : 0.85,      # drop ticker if >15% data missing overall
    "momentum_window"   : 252,       # trading days for momentum (12 months)
    "momentum_skip"     : 21,        # skip last month (avoid short-term reversal)
    "transaction_cost"  : 0.0010,    # 10 bps one-way per trade (per stock)
    "benchmark_ticker"  : "EXW1.DE", # iShares STOXX Europe 600 UCITS ETF
    "output_pdf"        : os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "CWW_Screener_Backtest_v4.pdf"),
}

# =============================================================================
#  1. UNIVERSE
# =============================================================================

TICKERS = [
    # Germany
    "SAP.DE","SIE.DE","ALV.DE","MBG.DE","BMW.DE","BAS.DE",
    "MUV2.DE","DTE.DE","BAYN.DE","DBK.DE","ADS.DE","VOW3.DE",
    "RWE.DE","IFX.DE","CON.DE","HEI.DE","MTX.DE","DHER.DE",
    # France
    "MC.PA","TTE.PA","AI.PA","BNP.PA","OR.PA","SU.PA",
    "AIR.PA","DG.PA","RMS.PA","CAP.PA","ACA.PA","GLE.PA",
    "SGO.PA","LR.PA","DSY.PA","VIE.PA",
    # United Kingdom
    "HSBA.L","BP.L","GSK.L","AZN.L","ULVR.L","RIO.L",
    "SHEL.L","VOD.L","LLOY.L","BARC.L","NG.L","REL.L",
    "NWG.L","LGEN.L","IMB.L","BA.L","CRH.L","EXPN.L",
    # Switzerland
    "NOVN.SW","NESN.SW","ROG.SW","UBSG.SW","ABBN.SW",
    "ZURN.SW","LONN.SW","SREN.SW","PGHN.SW",
    # Netherlands
    "ASML.AS","HEIA.AS","UNA.AS","PHIA.AS","INGA.AS",
    "ABN.AS","NN.AS","WKL.AS","AD.AS","ADYEN.AS",
    # Italy
    "ENEL.MI","ENI.MI","ISP.MI","UCG.MI","G.MI",
    "MB.MI","RACE.MI","PRY.MI","A2A.MI",
    # Spain
    "IBE.MC","SAN.MC","BBVA.MC","ITX.MC","REP.MC",
    "CABK.MC","ELE.MC","TEF.MC","AMS.MC",
    # Nordics
    "NOVO-B.CO","MAERSK-B.CO","ORSTED.CO","CARL-B.CO",
    "VOLV-B.ST","ERIC-B.ST","HM-B.ST","SEB-A.ST","ATCO-A.ST",
    "DNB.OL","EQNR.OL","YAR.OL",
    "SAMPO.HE","FORTUM.HE","KNEBV.HE",
]

# =============================================================================
#  2. DATA DOWNLOAD
# =============================================================================

def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    print(f"\n[1/5] Downloading price data for {len(tickers)} tickers …")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    threshold = int(len(raw) * CONFIG["min_history_frac"])
    raw = raw.dropna(axis=1, thresh=threshold)
    raw = raw.ffill()
    print(f"       → {raw.shape[1]} tickers retained after quality filter.")
    return raw


def download_benchmark(start: str, end: str) -> pd.Series:
    """Download benchmark ETF and return a clean price series."""
    ticker = CONFIG["benchmark_ticker"]
    print(f"       Downloading benchmark ({ticker}) …")
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.squeeze()
    raw = raw.ffill().dropna()
    raw.name = "Benchmark"
    return raw

# =============================================================================
#  3. SCREENING METRICS  (on a slice of history)
# =============================================================================

def compute_metrics(prices_slice: pd.DataFrame,
                    rf: float = CONFIG["risk_free_rate"]) -> pd.DataFrame:
    """
    Compute per-ticker metrics on prices_slice.
    Only tickers with enough history are scored.
    Returns DataFrame sorted by composite Score (descending).
    """
    min_days  = CONFIG["min_history_days"]
    mom_win   = CONFIG["momentum_window"]
    mom_skip  = CONFIG["momentum_skip"]
    daily_rf  = (1 + rf) ** (1 / 252) - 1

    rets = prices_slice.pct_change().dropna()
    records = []

    for tkr in rets.columns:
        r = rets[tkr].dropna()
        p = prices_slice[tkr].dropna()

        if len(r) < min_days:
            continue

        n_years   = len(r) / 252
        total_ret = (p.iloc[-1] / p.iloc[0]) - 1
        cagr      = (1 + total_ret) ** (1 / n_years) - 1
        vol       = r.std() * np.sqrt(252)
        excess    = r - daily_rf
        sharpe    = excess.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

        downside    = r[r < daily_rf] - daily_rf
        sortino_den = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
        sortino     = excess.mean() * np.sqrt(252) / sortino_den if sortino_den > 0 else np.nan

        roll_max = p.cummax()
        drawdown = (p - roll_max) / roll_max
        max_dd   = drawdown.min()
        calmar   = cagr / abs(max_dd) if max_dd != 0 else np.nan

        # Momentum 12-1: return from 252 days ago to 21 days ago
        if len(p) >= mom_win + mom_skip:
            mom = (p.iloc[-(mom_skip + 1)] / p.iloc[-(mom_win + mom_skip)]) - 1
        else:
            mom = np.nan

        records.append({
            "Ticker"   : tkr,
            "CAGR"     : round(cagr  * 100, 2),
            "Vol"      : round(vol   * 100, 2),
            "Sharpe"   : round(sharpe,       3),
            "Sortino"  : round(sortino,       3),
            "MaxDD"    : round(max_dd * 100,  2),
            "Calmar"   : round(calmar,         3),
            "Mom12_1"  : round(mom   * 100,   2),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("Ticker")

    # composite score: z-score each factor, average them
    to_score = pd.DataFrame(index=df.index)
    to_score["CAGR"]    = df["CAGR"]
    to_score["Sharpe"]  = df["Sharpe"]
    to_score["Sortino"] = df["Sortino"]
    to_score["Calmar"]  = df["Calmar"]
    to_score["Mom"]     = df["Mom12_1"]
    to_score["Vol_inv"] = -df["Vol"]
    to_score["DD_inv"]  = -df["MaxDD"]

    to_score = to_score.dropna()
    z = (to_score - to_score.mean()) / (to_score.std() + 1e-9)
    df["Score"] = z.mean(axis=1).round(3)
    df = df.sort_values("Score", ascending=False)
    return df

# =============================================================================
#  4. PORTFOLIO OPTIMISATION HELPERS
# =============================================================================

def ew_weights(n: int) -> np.ndarray:
    return np.ones(n) / n


def min_variance(cov: np.ndarray, max_w: float = 0.05) -> np.ndarray:
    n  = cov.shape[0]
    w0 = np.ones(n) / n
    res = minimize(
        lambda w: w @ cov @ w, w0,
        method="SLSQP",
        bounds=[(0, max_w)] * n,      # 5% max weight per stock
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    return res.x if res.success else w0


def max_sharpe(mean_rets: np.ndarray, cov: np.ndarray, rf: float,
               max_w: float = 0.05) -> np.ndarray:
    n  = cov.shape[0]
    w0 = np.ones(n) / n
    def neg_sharpe(w):
        p_ret = w @ mean_rets
        p_vol = np.sqrt(w @ cov @ w)
        return -(p_ret - rf) / p_vol if p_vol > 1e-8 else 1e8
    res = minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=[(0, max_w)] * n,      # 5% max weight per stock
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    return res.x if res.success else w0

# =============================================================================
#  5. TRANSACTION COST MODEL
# =============================================================================

def compute_turnover_cost(prev_weights: np.ndarray | None,
                          new_weights: np.ndarray,
                          prev_tickers: list,
                          new_tickers: list,
                          tc: float = CONFIG["transaction_cost"]) -> float:
    """
    Estimate one-way transaction cost as a fraction of portfolio value.

    Logic:
      - Stocks that enter or exit: full weight traded  → cost = w * tc
      - Stocks that remain but change weight: |Δw| traded → cost = |Δw| * tc
      - tc = 10 bps = 0.0010 one-way

    Returns the total cost as a fraction of NAV (applied as a drag on day 0
    of the new period).
    """
    tc_total = 0.0

    if prev_weights is None:
        # First rebalancing: buy everything from cash
        tc_total = np.sum(new_weights) * tc
        return tc_total

    prev_map = dict(zip(prev_tickers, prev_weights))
    new_map  = dict(zip(new_tickers,  new_weights))

    all_tickers = set(prev_tickers) | set(new_tickers)
    for tkr in all_tickers:
        w_old = prev_map.get(tkr, 0.0)
        w_new = new_map.get(tkr, 0.0)
        tc_total += abs(w_new - w_old) * tc

    return tc_total

# =============================================================================
#  6. WALK-FORWARD BACKTEST  (rolling screening + rolling optimisation)
# =============================================================================

def run_backtest(prices_all: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 rf: float = CONFIG["risk_free_rate"]) -> dict:
    """
    At each rebalancing date:
      1. Screen universe using ONLY historical data up to that date → top N
      2. Estimate weights (EW / MinVar / MaxSharpe) on same historical window
      3. Deduct transaction costs (10 bps one-way) on weight changes
      4. Apply weights forward until next rebalancing date

    Benchmark (buy & hold ETF) is simulated in parallel.
    """
    print("\n[3/5] Running walk-forward backtest …")
    freq    = CONFIG["rebalance_freq"]
    top_n   = CONFIG["top_n_stocks"]
    tc      = CONFIG["transaction_cost"]
    daily_rf = (1 + rf) ** (1 / 252) - 1

    rets_all     = prices_all.pct_change().dropna()
    rebal_dates  = rets_all.resample(freq).last().index.tolist()

    strategies   = ["EW", "MinVar", "MaxSharpe"]
    port_values  = {s: [1.0] for s in strategies}
    dates_out    = [rets_all.index[0]]
    selection_log = []
    tc_log        = {s: [] for s in strategies}   # store cost per rebalancing

    # previous weights (for turnover cost calculation)
    prev_weights  = {s: None for s in strategies}
    prev_tickers  = {s: []   for s in strategies}

    prev_date = rets_all.index[0]

    for rd in rebal_dates:
        hist_prices = prices_all.loc[:rd]
        hist_rets   = rets_all.loc[:rd]

        # --- STEP 1: walk-forward screening ---
        metrics = compute_metrics(hist_prices, rf)
        if metrics.empty or len(metrics) < 5:
            prev_date = rd
            continue

        top_tickers = metrics.head(top_n).index.tolist()
        selection_log.append({"date": rd, "tickers": top_tickers,
                               "metrics": metrics.head(top_n)})

        period_rets = rets_all.loc[prev_date:rd][top_tickers].iloc[1:]
        if period_rets.empty:
            prev_date = rd
            continue

        # --- STEP 2: estimate weights on historical data ---
        hist_top   = hist_rets[top_tickers].dropna()
        mean_ann   = hist_top.mean() * 252
        cov_ann    = hist_top.cov()  * 252

        new_weights_map = {
            "EW"      : ew_weights(len(top_tickers)),
            "MinVar"  : min_variance(cov_ann.values),
            "MaxSharpe": max_sharpe(mean_ann.values, cov_ann.values, rf),
        }

        # --- STEP 3: deduct transaction costs ---
        for strat in strategies:
            cost = compute_turnover_cost(
                prev_weights[strat], new_weights_map[strat],
                prev_tickers[strat], top_tickers, tc
            )
            # apply cost drag: NAV decreases by cost fraction on rebalancing day
            port_values[strat][-1] *= (1 - cost)
            tc_log[strat].append({"date": rd, "cost_bps": round(cost * 10_000, 2)})

        # --- STEP 4: simulate period ---
        for day, row in period_rets.iterrows():
            for strat in strategies:
                p_ret = new_weights_map[strat] @ row.values
                port_values[strat].append(port_values[strat][-1] * (1 + p_ret))
            dates_out.append(day)

        # update previous weights
        for strat in strategies:
            prev_weights[strat] = new_weights_map[strat]
            prev_tickers[strat] = top_tickers[:]

        prev_date = rd
        cost_ew = tc_log["EW"][-1]["cost_bps"] if tc_log["EW"] else 0
        print(f"       Rebalanced {rd.date()} → top {len(top_tickers)} stocks  "
              f"| TC (EW) = {cost_ew:.1f} bps")

    nav = pd.DataFrame(port_values,
                       index=dates_out[:len(port_values["EW"])])
    nav = nav[~nav.index.duplicated(keep="last")].sort_index()

    # --- Benchmark: reindex to match nav, normalise to 1 ---
    bm = benchmark_prices.reindex(nav.index, method="ffill").dropna()
    bm = bm / bm.iloc[0]
    nav["Benchmark"] = bm

    return {
        "nav"          : nav,
        "selection_log": selection_log,
        "tc_log"       : tc_log,
    }

# =============================================================================
#  7. PERFORMANCE METRICS
# =============================================================================

def portfolio_metrics(nav: pd.Series, rf: float = CONFIG["risk_free_rate"]) -> dict:
    r = nav.pct_change().dropna()
    daily_rf  = (1 + rf) ** (1 / 252) - 1
    n_years   = len(r) / 252
    cagr      = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1
    vol       = r.std() * np.sqrt(252)
    excess    = r - daily_rf
    sharpe    = excess.mean() / r.std() * np.sqrt(252)
    downside  = r[r < daily_rf] - daily_rf
    sortino_d = np.sqrt((downside**2).mean()) * np.sqrt(252)
    sortino   = excess.mean() * np.sqrt(252) / sortino_d if sortino_d > 0 else np.nan
    roll_max  = nav.cummax()
    drawdown  = (nav - roll_max) / roll_max
    max_dd    = drawdown.min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "CAGR %"   : round(cagr  * 100, 2),
        "Vol %"    : round(vol   * 100, 2),
        "Sharpe"   : round(sharpe,       3),
        "Sortino"  : round(sortino,       3),
        "Max DD %" : round(max_dd * 100,  2),
        "Calmar"   : round(calmar,         3),
    }

# =============================================================================
#  8. VISUALISATION
# =============================================================================

PALETTE = {
    "EW"       : "#2E86AB",
    "MinVar"   : "#E84855",
    "MaxSharpe": "#3BB273",
    "Benchmark": "#FF9F1C",   # orange — benchmark always stands out
}
DARK = "#1A1A2E"
GREY = "#F4F4F8"

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.facecolor"  : GREY,
    "figure.facecolor": "white",
    "axes.edgecolor"  : "#CCCCCC",
    "grid.color"      : "white",
    "grid.linewidth"  : 1.2,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 10,
})


def fig_cover() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.axis("off")
    ax.text(0.5, 0.70, "European Stock Screener",
            ha="center", va="center", fontsize=34, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.58, "& Portfolio Backtesting",
            ha="center", va="center", fontsize=34, fontweight="bold",
            color="#3BB273", transform=ax.transAxes)
    ax.text(0.5, 0.44,
            f"Walk-forward selection · EW  |  Min Variance  |  Max Sharpe\n"
            f"Benchmark: iShares STOXX Europe 600 (EXW1.DE)  |  TC: 10 bps one-way  |  Max weight: 5%\n"
            f"Factors: CAGR · Sharpe · Sortino · Calmar · Momentum 12-1 · Vol · MaxDD\n"
            f"{CONFIG['start_date']}  →  {CONFIG['end_date']}",
            ha="center", va="center", fontsize=13, color="#AAAAAA",
            transform=ax.transAxes)
    ax.text(0.5, 0.18,
            "STOXX Europe 600 representative universe",
            ha="center", va="center", fontsize=11, color="#666688",
            transform=ax.transAxes)
    plt.tight_layout()
    return fig


def fig_last_screening(selection_log: list, top_n: int) -> plt.Figure:
    last = selection_log[-1]
    df   = last["metrics"].reset_index()
    date_str = last["date"].strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(11.7, 9.5))
    ax.axis("off")

    # title placed well above the table using axes coordinates
    ax.text(0.5, 0.97,
            f"Latest Screening Results — Top {top_n} stocks  ({date_str})",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color=DARK, transform=ax.transAxes)

    col_labels = df.columns.tolist()
    # bbox: [left, bottom, width, height] in axes fraction — leaves top 8% for title
    table = ax.table(cellText=df.values.tolist(), colLabels=col_labels,
                     cellLoc="center",
                     bbox=[0, 0, 1, 0.90])
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor(DARK)
        table[0, j].set_text_props(color="white", fontweight="bold")

    score_col = col_labels.index("Score")
    scores    = df["Score"].values
    s_min, s_max = scores.min(), scores.max()
    cmap = LinearSegmentedColormap.from_list("rg", ["#FFB3B3", "#B3FFB3"])

    for i in range(1, len(df) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if j == score_col:
                norm = (scores[i-1] - s_min) / (s_max - s_min + 1e-9)
                cell.set_facecolor(cmap(norm))
            else:
                cell.set_facecolor("#EEEEF5" if i % 2 == 0 else "white")

    plt.tight_layout()
    return fig


def fig_turnover(selection_log: list, tc_log: dict) -> plt.Figure:
    """Bar chart: new stocks each quarter + cost overlay."""
    dates, turnover = [], []
    for i in range(1, len(selection_log)):
        prev = set(selection_log[i-1]["tickers"])
        curr = set(selection_log[i]["tickers"])
        dates.append(selection_log[i]["date"])
        turnover.append(len(curr - prev))

    tc_dates = [e["date"] for e in tc_log["EW"]]
    tc_bps   = [e["cost_bps"] for e in tc_log["EW"]]

    fig, ax1 = plt.subplots(figsize=(11.7, 4.5))
    ax2 = ax1.twinx()

    ax1.bar(dates, turnover, color="#2E86AB", width=50, edgecolor="white",
            alpha=0.85, label="# stocks replaced (left)")
    ax2.plot(tc_dates, tc_bps, color="#E84855", linewidth=2, marker="o",
             markersize=5, label="TC drag EW (bps, right)")

    ax1.set_title("Portfolio Turnover & Transaction Cost Drag (EW strategy)",
                  fontweight="bold", color=DARK)
    ax1.set_ylabel("# stocks replaced", color="#2E86AB")
    ax2.set_ylabel("TC drag (bps)", color="#E84855")
    ax1.tick_params(axis="y", labelcolor="#2E86AB")
    ax2.tick_params(axis="y", labelcolor="#E84855")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def fig_equity_curves(nav: pd.DataFrame) -> plt.Figure:
    """Equity curves for all strategies + benchmark."""
    fig, ax = plt.subplots(figsize=(11.7, 6))
    for col in ["EW", "MinVar", "MaxSharpe", "Benchmark"]:
        lw    = 1.6 if col == "Benchmark" else 2.2
        ls    = "--" if col == "Benchmark" else "-"
        label = f"{col} (STOXX 600 ETF)" if col == "Benchmark" else col
        ax.plot(nav.index, nav[col] * 100_000,
                label=label, color=PALETTE[col],
                linewidth=lw, linestyle=ls)

    ax.set_title(
        "Portfolio Equity Curves vs Benchmark — Walk-forward, 10 bps TC  "
        "(starting capital €100,000)",
        fontweight="bold", color=DARK)
    ax.set_ylabel("Portfolio Value (€)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=11)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def fig_drawdown(nav: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.7, 4.5))
    for col in ["EW", "MinVar", "MaxSharpe", "Benchmark"]:
        lw = 1.2 if col == "Benchmark" else 1.5
        ls = "--" if col == "Benchmark" else "-"
        roll_max = nav[col].cummax()
        dd = (nav[col] - roll_max) / roll_max * 100
        ax.fill_between(nav.index, dd, 0,
                        alpha=0.20 if col == "Benchmark" else 0.30,
                        color=PALETTE[col])
        ax.plot(nav.index, dd, color=PALETTE[col],
                linewidth=lw, linestyle=ls,
                label=col)
    ax.set_title("Drawdown over time (%)", fontweight="bold", color=DARK)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(fontsize=11)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def fig_metrics_table(nav: pd.DataFrame, rf: float) -> plt.Figure:
    """Summary table including benchmark row."""
    cols_order = ["EW", "MinVar", "MaxSharpe", "Benchmark"]
    rows = {s: portfolio_metrics(nav[s], rf) for s in cols_order}
    df   = pd.DataFrame(rows).T.reset_index()
    df.rename(columns={"index": "Strategy"}, inplace=True)

    fig, ax = plt.subplots(figsize=(11.7, 3.8))
    ax.axis("off")
    ax.set_title("Performance Summary — Full Walk-forward Period  (after 10 bps TC)",
                 fontsize=14, fontweight="bold", pad=16, color=DARK)

    table = ax.table(cellText=df.values.tolist(), colLabels=df.columns.tolist(),
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    row_colors = {
        "EW"       : "#2E86AB",
        "MinVar"   : "#E84855",
        "MaxSharpe": "#3BB273",
        "Benchmark": "#FF9F1C",
    }
    for j in range(len(df.columns)):
        table[0, j].set_facecolor(DARK)
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, strat in enumerate(cols_order, start=1):
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(
                row_colors[strat] if j == 0 else
                ("#EEEEF5" if i % 2 == 0 else "white"))
            if j == 0:
                table[i, j].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    return fig


def fig_metrics_bars(nav: pd.DataFrame, rf: float) -> plt.Figure:
    """Bar comparison across strategies + benchmark."""
    metrics_list = ["CAGR %", "Sharpe", "Sortino", "Calmar"]
    cols_order   = ["EW", "MinVar", "MaxSharpe", "Benchmark"]
    rows = {s: portfolio_metrics(nav[s], rf) for s in cols_order}
    df   = pd.DataFrame(rows).T[metrics_list]

    fig, axes = plt.subplots(1, 4, figsize=(11.7, 4.5))
    for ax, metric in zip(axes, metrics_list):
        vals   = df[metric]
        colors = [PALETTE[s] for s in vals.index]
        bars   = ax.bar(vals.index, vals.values, color=colors,
                        width=0.55, edgecolor="white")
        ax.set_title(metric, fontweight="bold", color=DARK)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(vals.values).max() * 0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
    fig.suptitle("Strategy Comparison vs Benchmark — Key Metrics  (after 10 bps TC)",
                 fontweight="bold", fontsize=13, color=DARK, y=1.01)
    plt.tight_layout()
    return fig


def fig_rolling_sharpe(nav: pd.DataFrame, rf: float,
                        window: int = 252) -> plt.Figure:
    """Rolling Sharpe for all strategies + benchmark."""
    fig, ax = plt.subplots(figsize=(11.7, 4.5))
    daily_rf = (1 + rf) ** (1 / 252) - 1
    for col in ["EW", "MinVar", "MaxSharpe", "Benchmark"]:
        lw = 1.5 if col == "Benchmark" else 1.8
        ls = "--" if col == "Benchmark" else "-"
        r  = nav[col].pct_change().dropna()
        rs = ((r - daily_rf).rolling(window).mean() /
              r.rolling(window).std() * np.sqrt(252))
        ax.plot(rs.index, rs, label=col,
                color=PALETTE[col], linewidth=lw, linestyle=ls)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {window}-day Sharpe Ratio vs Benchmark",
                 fontweight="bold", color=DARK)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=11)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def fig_alpha_chart(nav: pd.DataFrame) -> plt.Figure:
    """
    Relative performance vs benchmark (active return).
    Active NAV = strategy NAV / benchmark NAV.
    Values above 1 mean outperformance.
    """
    fig, ax = plt.subplots(figsize=(11.7, 4.5))
    bm = nav["Benchmark"]
    for strat in ["EW", "MinVar", "MaxSharpe"]:
        rel = nav[strat] / bm
        ax.plot(rel.index, rel, label=strat,
                color=PALETTE[strat], linewidth=2)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--",
               label="Benchmark (=1)")
    ax.fill_between(nav.index,
                    nav["EW"] / bm, 1.0,
                    where=(nav["EW"] / bm) > 1.0,
                    alpha=0.10, color=PALETTE["EW"])
    ax.set_title("Active Return vs Benchmark  (Strategy NAV / Benchmark NAV)",
                 fontweight="bold", color=DARK)
    ax.set_ylabel("Relative NAV  (1.0 = benchmark)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    ax.legend(fontsize=11)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def fig_screening_heatmap(selection_log: list, top_n: int) -> plt.Figure:
    last = selection_log[-1]
    df   = last["metrics"].head(top_n)[
        ["CAGR", "Sharpe", "Sortino", "Calmar", "Mom12_1", "MaxDD", "Vol"]]

    fig, ax = plt.subplots(figsize=(11.7, max(5, top_n * 0.32)))
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#FF6B6B", "#F8F8F8", "#51CF66"])
    sns.heatmap(df, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.4, linecolor="#DDDDDD",
                ax=ax, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.6})

    # move column labels from bottom to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", length=0, labelsize=9, pad=4)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_title(f"Metrics Heatmap — Latest screening top {top_n} stocks",
                 fontweight="bold", fontsize=13, color=DARK, pad=30)
    plt.tight_layout()
    return fig


# =============================================================================
#  9. MAIN
# =============================================================================

def main():
    rf    = CONFIG["risk_free_rate"]
    top_n = CONFIG["top_n_stocks"]

    # download universe + benchmark
    prices_all = download_prices(TICKERS, CONFIG["start_date"], CONFIG["end_date"])
    benchmark  = download_benchmark(CONFIG["start_date"], CONFIG["end_date"])

    print("\n[2/5] Walk-forward screening will run inside the backtest loop …")

    # walk-forward backtest (screening + optimisation + TC at each rebalancing date)
    result    = run_backtest(prices_all, benchmark, rf)
    nav       = result["nav"]
    sel_log   = result["selection_log"]
    tc_log    = result["tc_log"]

    # generate PDF
    print(f"\n[4/5] Generating PDF → {CONFIG['output_pdf']} …")
    with PdfPages(CONFIG["output_pdf"]) as pdf:
        for fig in [
            fig_cover(),
            fig_last_screening(sel_log, top_n),
            fig_screening_heatmap(sel_log, top_n),
            fig_equity_curves(nav),
            fig_alpha_chart(nav),
            fig_drawdown(nav),
            fig_metrics_table(nav, rf),
            fig_metrics_bars(nav, rf),
            fig_rolling_sharpe(nav, rf),
            fig_turnover(sel_log, tc_log),
        ]:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        d = pdf.infodict()
        d["Title"]   = "European Stock Screener & Portfolio Backtest"
        d["Author"]  = "Emanuele Pozzani"
        d["Subject"] = ("Walk-forward · EW vs MinVar vs MaxSharpe vs STOXX 600 "
                        "· 10 bps TC · 5% max weight")

    print(f"[5/5] Done!  PDF saved as '{CONFIG['output_pdf']}'")


if __name__ == "__main__":
    main()