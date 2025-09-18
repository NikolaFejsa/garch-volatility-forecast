# Auto-GARCH Forecasting Project

# Initialisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os

import warnings
warnings.filterwarnings("ignore")

# Plot Style 
plt.style.use("dark_background")        
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "gray"
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.facecolor"] = "black"
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["savefig.facecolor"] = "black"


#Defining Functions
def fetch_prices(ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol or period.")
    df = df[["Close"]].dropna()
    return df

def make_returns(df: pd.DataFrame) -> pd.Series:
    # percent log returns
    ret = 100 * np.log(df["Close"]).diff()
    ret.name = "ret_pct"
    return ret.dropna()

def suggest_pq(returns: pd.Series, p_max=3, q_max=3) -> tuple[int, int, float]:
    """Grid-search to suggest (p, q) by BIC."""
    best = (None, None, np.inf)
    for p in range(p_max + 1):
        for q in range(1, q_max + 1):  # require at least an ARCH term
            try:
                am = arch_model(returns, mean="Constant", vol="GARCH", p=p, q=q, dist="t")
                res = am.fit(disp="off")
                bic = res.bic
                if bic < best[2]:
                    best = (p, q, bic)
            except Exception:
                continue
    return best

def fit_garch(returns: pd.Series, p: int, q: int):
    am = arch_model(returns, mean="Constant", vol="GARCH", p=p, q=q, dist="t")
    res = am.fit(disp="off")
    return res

def forecast_vol(model_fit, horizon: int = 7) -> pd.Series:
    """
    Returns daily volatility forecast in percent units,
    for the next `horizon` trading days.
    """
    fc = model_fit.forecast(horizon=horizon, reindex=False)
    var_h = np.asarray(fc.variance.values[-1, :], dtype=float)
    sigma = np.sqrt(var_h)  # percent per day

    # Use a public index
    last_date = model_fit.conditional_volatility.index[-1]
    future_idx = pd.bdate_range(start=last_date, periods=horizon + 1, freq="B")[1:]
    return pd.Series(sigma, index=future_idx, name="sigma_pct")

def realized_variance(ret: pd.Series) -> pd.Series:
    """One-day realized variance proxy: r_t^2 (returns are in percent)."""
    return (ret / 100.0) ** 2  # percent -> decimal, then square


def qlike_loss(forecast_var: pd.Series, realized_var: pd.Series) -> pd.Series:
    """
    QLIKE loss: log(sigma^2) + r^2 / sigma^2 (lower is better).
    """
    f = forecast_var.clip(lower=1e-12)
    r = realized_var
    return np.log(f) + (r / f)


def ewma_variance_1dahead(ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    EWMA 1-step-ahead forecast of variance.
    Returns a series indexed by (t+1),
    aligned with realized variance on that next day.
    """
    r = (ret / 100.0).dropna()  # decimal
    if len(r) < 20:
        return pd.Series(dtype=float)

    var = pd.Series(index=r.index, dtype=float)
    # init variance with first 20 obs
    v = r.iloc[:20].var()
    var.iloc[19] = v
    for t in range(20, len(r)):
        v = lam * v + (1 - lam) * (r.iloc[t - 1] ** 2)
        var.iloc[t] = v

    # Forecast made at date t applies to date t+1 -> reindex forward by one day
    next_idx = var.index[1:]
    return pd.Series(var.values[:-1], index=next_idx, name="ewma_var")

def backtest_garch_1d(rets: pd.Series, p: int, q: int, window: int | None = None, dist: str = "t") -> pd.DataFrame:
    """
    Rolling 1-step-ahead backtest of GARCH(p,q) reporting QLIKE only.
    Returns DF with realized_var, garch_var, ewma_var, and their QLIKE.
    """
    rets = rets.dropna()
    n = len(rets)
    min_obs = 40
    if n < min_obs:
        raise ValueError(f"Not enough data for backtest (have {n}, need >= {min_obs}). Try a longer period.")

    # Auto window sized to sample
    if window is None:
        window = max(30, min(int(0.5 * n), n - 15))

    dates = rets.index
    ewma_var_next = ewma_variance_1dahead(rets) # at t+1

    garch_var_next = []
    garch_idx = []

    # At time t (using data up to t) forecast variance for t+1
    for t in range(window, n - 1):
        train = rets.iloc[:t + 1]  
        try:
            am = arch_model(train, mean="Constant", vol="GARCH", p=p, q=q, dist=dist)
            res = am.fit(disp="off")
            f = res.forecast(horizon=1, reindex=False)
            v_next_pct2 = float(f.variance.values[-1, 0])     
            v_next = v_next_pct2 / (100.0 ** 2)                
        except Exception:
            v_next = np.nan

        garch_var_next.append(v_next)
        garch_idx.append(dates[t + 1])  # next day

    garch_var_next = pd.Series(garch_var_next, index=garch_idx, name="garch_var")
    rv_next = realized_variance(rets).reindex(garch_var_next.index)

    out = pd.concat(
        [rv_next.rename("realized_var"),
         garch_var_next,
         ewma_var_next.reindex(garch_var_next.index)],
        axis=1
    ).dropna()

    # Report QLIKE
    out["loss_qlike_garch"] = qlike_loss(out["garch_var"], out["realized_var"])
    out["loss_qlike_ewma"]  = qlike_loss(out["ewma_var"],  out["realized_var"])
    return out

#PDF Building function
def _fmt_pct(x, digits=2):
    try:
        return f"{x:.{digits}f}%"
    except Exception:
        return "—"

def build_pdf_report(
    ticker: str,
    period: str,
    p: int,
    q: int,
    horizon: int,
    cond_vol_last5: pd.Series,
    sigma_forecast: pd.Series,
    mean_qlike_garch: float | None,
    mean_qlike_ewma: float | None,
    have_backtest: bool,
    plot_full_path: str = "plot_vol_full.png",
    plot_clarity_path: str = "plot_vol_clarity.png",
    plot_bt_path: str = "plot_backtest.png",
    out_path: str | None = None,
):
    
    #Includes: Objective, Data, Model, Forecasts, Backtest (if available), Takeaways.
    
    if out_path is None:
        out_path = f"report_{ticker}_garch{p}{q}.pdf"

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.grey))

    story = []

    #Title
    title = f"GARCH Volatility Forecasting Report – {ticker}"
    story.append(Paragraph(title, styles["H1"]))
    story.append(Paragraph(f"Period analyzed: {period} &nbsp;&nbsp;|&nbsp;&nbsp; Model: GARCH({p},{q}) &nbsp;&nbsp;|&nbsp;&nbsp; Forecast horizon: {horizon} trading days", styles["Small"]))
    story.append(Spacer(1, 6))

    #Objective
    story.append(Paragraph("1. Objective", styles["H2"]))
    story.append(Paragraph(
        "Forecast short-term stock return volatility using a GARCH(p,q) model and provide clear visuals for upcoming risk. "
        "Where possible, evaluate forecast quality against a simple EWMA(0.94) benchmark using QLIKE loss (lower is better).",
        styles["Body"]))
    story.append(Spacer(1, 6))

    #Data
    story.append(Paragraph("2. Data", styles["H2"]))
    story.append(Paragraph(
        "Data source: Yahoo Finance (adjusted close). Returns are daily log returns expressed in percent. "
        "Volatility figures shown are daily standard deviations in percent.", styles["Body"]))
    story.append(Spacer(1, 6))

    #Model Selection
    story.append(Paragraph("3. Model Selection", styles["H2"]))
    story.append(Paragraph(
        "Model orders (p,q) guided by ACF/PACF diagnostics of (squared) returns and BIC search. "
        f"Chosen specification: GARCH({p},{q}).", styles["Body"]))
    story.append(Spacer(1, 6))

    #Forecasts 
    story.append(Paragraph("4. Forecasts", styles["H2"]))
    # Last in-sample vols table
    try:
        last_rows = [
            ["Date", "In-sample daily vol (%)"]
        ] + [[str(idx.date()), _fmt_pct(val, 3)] for idx, val in cond_vol_last5.items()]
        t = Table(last_rows, hAlign="LEFT", colWidths=[140, 160])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))
    except Exception:
        pass

    # Forecast head table
    try:
        fc_head = sigma_forecast.head(min(7, len(sigma_forecast)))
        fc_rows = [["Date", "Forecast daily vol (%)"]] + [[str(idx.date()), _fmt_pct(val, 3)] for idx, val in fc_head.items()]
        ft = Table(fc_rows, hAlign="LEFT", colWidths=[140, 160])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(ft)
        story.append(Spacer(1, 8))
    except Exception:
        pass

    # Insert plots 
    if os.path.exists(plot_full_path):
        story.append(Image(plot_full_path, width=520, height=240))
        story.append(Spacer(1, 8))
    if os.path.exists(plot_clarity_path):
        story.append(Image(plot_clarity_path, width=520, height=220))
        story.append(Spacer(1, 8))

    #Backtest
    if have_backtest and (mean_qlike_garch is not None) and (mean_qlike_ewma is not None):
        story.append(Paragraph("5. Backtest (1-day ahead, QLIKE)", styles["H2"]))
        story.append(Paragraph(
            "Rolling re-fit with 1-day-ahead variance forecasts. Benchmark: EWMA(0.94). "
            "Evaluation metric: QLIKE", styles["Body"]))
        story.append(Spacer(1, 4))

        bt_rows = [
            ["Metric", "GARCH", "EWMA(0.94)"],
            ["Average QLIKE", f"{mean_qlike_garch:.6f}", f"{mean_qlike_ewma:.6f}"],
        ]
        bt_table = Table(bt_rows, hAlign="LEFT", colWidths=[180, 120, 120])
        bt_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(bt_table)
        story.append(Spacer(1, 6))

        if os.path.exists(plot_bt_path):
            story.append(Image(plot_bt_path, width=520, height=240))
            story.append(Spacer(1, 8))

    #Takeaways
    story.append(Paragraph("6. Takeaways", styles["H2"]))

    # 1)Most recent predicted volatilities
    fc_head = sigma_forecast.head(min(3, len(sigma_forecast)))
    predicted_str = "; ".join([f"{str(idx.date())}: {_fmt_pct(val,3)}" for idx, val in fc_head.items()])

    # 2) Backtest performance message
    if have_backtest and (mean_qlike_garch is not None) and (mean_qlike_ewma is not None):
        if mean_qlike_garch < mean_qlike_ewma:
            perf_msg = f"GARCH outperformed EWMA on QLIKE ({mean_qlike_garch:.6f} vs {mean_qlike_ewma:.6f})."
        else:
            perf_msg = f"EWMA outperformed GARCH on QLIKE ({mean_qlike_ewma:.6f} vs {mean_qlike_garch:.6f})."
    else:
        perf_msg = "Backtest unavailable for this sample (insufficient data)."

    # 3) Build final bullets
    bullets = (
        f"Predicted volatility (next days): {predicted_str}. "
        "<br/>"
        f"{perf_msg} "
        "<br/>"
        "GARCH provides time-varying daily volatility estimates useful for risk sizing, derivatives, and portfolio optimisation. "
        "Forecasts are in daily %; results depend on asset and lookback window."
        )

    story.append(Paragraph(bullets, styles["Body"]))

    doc.build(story)
    return out_path


#Core Function
def main():
    print("=== Auto-GARCH Forecasting Project ===")
    ticker = input("Enter ticker: ").strip().upper()
    period = input("History period for yfinance (6mo, 1y, 2y, 5y, 10y, ytd, max): ").strip()

    # 1) Download data
    prices = fetch_prices(ticker, period=period)
    rets = make_returns(prices)

    # 2) Inspect ACF/PACF
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Price 
    axes[0].plot(prices.index, prices["Close"], color="cyan", linewidth=1.2)
    axes[0].set_title(f"{ticker} Adjusted Close")
    axes[0].set_xlabel("Date"); axes[0].set_ylabel("Price")

    # ACF (returns)
    plot_acf(rets.dropna(), lags=30, ax=axes[1])
    for l in axes[1].lines:    
        l.set_color("white")
    axes[1].set_title("ACF of Returns")

    # PACF (squared returns) 
    plot_pacf((rets**2).dropna(), lags=30, ax=axes[2], method="ywm")
    for l in axes[2].lines:
        l.set_color("white")
    axes[2].set_title("PACF of Squared Returns")

    fig.suptitle("Use PACF(squared) for ARCH order (q); slow ACF decay ⇒ GARCH component (p).", fontsize=10)
    plt.tight_layout()
    plt.show()

    # 3) Suggest (p,q) but let user decide
    sp, sq, sbic = suggest_pq(rets, p_max=3, q_max=3)
    print(f"\nSuggested (p,q) by BIC: ({sp},{sq})  [BIC={sbic:.2f}]")
    try:
        p = int(input("Choose GARCH p: ").strip())
        q = int(input("Choose ARCH q:").strip())
    except ValueError:
        print("Invalid input. Falling back to suggested (p,q).")
        p, q = sp, sq

    # 4) Fit model
    print(f"\nFitting GARCH({p},{q}) with t-distribution…")
    res = fit_garch(rets, p=p, q=q)
    print(res.summary())

    # 5) Forecast next chosen 'N' business days of volatility
    horizon = int(input("Enter forecast horizon in trading days (e.g. 1,7): ").strip())
    sigma = forecast_vol(res, horizon=horizon)  # in percent per day

    # 6) Plot: in-sample conditional volatility + forecast
    cond_vol = res.conditional_volatility  
    plt.figure(figsize=(12, 4))
    cond_vol.plot(label="In-sample sigma (pct)")
    sigma.plot(label="Forecast sigma (pct)")
    plt.title(f"{ticker} – GARCH({p},{q}) Daily Volatility (percent)")
    plt.ylabel("Volatility (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_vol_full.png", dpi=150, bbox_inches="tight")
    plt.show()
    

    # 7) Plot: close-up of forecast (last K vs next H days)
    K = min(7, len(res.conditional_volatility))
    H = min(len(sigma), max(1, int(horizon)))  # if user typed 1, title/legend say 1

    ins_last = res.conditional_volatility.tail(K)   # daily % (in-sample)
    fc_next  = sigma.iloc[:H]                        # first H forecasted days
    split_date = ins_last.index[-1] if K > 0 else fc_next.index[0]

    plt.figure(figsize=(12, 3.8))
    if K > 0:
        ins_last.plot(color="lime", label=f"In-sample (last {K}d)", linewidth=2, marker="o")
        fc_next.plot(color="red", label=f"Forecast (next {H}d)", linewidth=2, marker="o")
        if K > 0:
            plt.axvline(split_date, linestyle="--", linewidth=1, color="gray", alpha=0.6)

    plt.title(f"{ticker} – Volatility: last {K} days vs next {H} days")
    plt.ylabel("Volatility (%)"); plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("plot_vol_clarity.png", dpi=150, bbox_inches="tight")  
    plt.show()

    # 8) Print last values
    print("\nLast 5 in-sample daily vol (%):")
    print(cond_vol.tail().round(3))
    print("\nForecast daily vol (%):")
    print(sigma.round(3))
    
    # 9) Backtesting
    print("\nRunning rolling 1-day-ahead backtest…")
    bt = None
    mean_qlike_garch = None
    mean_qlike_ewma  = None

    try:
        bt = backtest_garch_1d(rets, p=p, q=q, window=None, dist="t")
        mean_qlike_garch = bt["loss_qlike_garch"].mean()
        mean_qlike_ewma  = bt["loss_qlike_ewma"].mean()

        print("\nBacktest results (QLIKE, lower is better):")
        print(f"  GARCH: {mean_qlike_garch:.6f}")
        print(f"  EWMA : {mean_qlike_ewma:.6f}")
        if mean_qlike_garch < mean_qlike_ewma:
            print("GARCH outperforms EWMA on QLIKE")
        else:
            print("EWMA outperforms GARCH on QLIKE")

    # Plot realized vs forecasts
        plt.figure(figsize=(12, 4))
        (np.sqrt(bt["realized_var"]) * 100).plot(label="Realized sigma (%)", color="white", alpha=0.7)
        (np.sqrt(bt["garch_var"])   * 100).plot(label="GARCH sigma forecast (%)", color="lime")
        (np.sqrt(bt["ewma_var"])    * 100).plot(label="EWMA sigma forecast (%)", color="orange")
        plt.title(f"{ticker} – Rolling 1D Volatility: Realized vs Forecasts (QLIKE evaluation)")
        plt.ylabel("Volatility (%)"); plt.xlabel("Date")
        plt.legend(framealpha=0.2)
        plt.tight_layout()
        plt.savefig("plot_backtest.png", dpi=150, bbox_inches="tight")  
        plt.show()

    except ValueError as e:
        print(f"Backtest skipped: {e}")
    

    # 10) PDF Report
    have_backtest = ('bt' in locals()) and isinstance(bt, pd.DataFrame) and (not bt.empty)

    cond_vol_last5 = res.conditional_volatility.tail(5)

    report_path = build_pdf_report(
    ticker=ticker,
    period=period,
    p=p, q=q,
    horizon=horizon,
    cond_vol_last5=cond_vol_last5,
    sigma_forecast=sigma,         
    mean_qlike_garch=mean_qlike_garch,
    mean_qlike_ewma=mean_qlike_ewma,
    have_backtest=have_backtest,
    plot_full_path="plot_vol_full.png",
    plot_clarity_path="plot_vol_clarity.png",
    plot_bt_path="plot_backtest.png",
)
    print(f"\nSaved PDF report: {report_path}")

if __name__ == "__main__":
    main()
