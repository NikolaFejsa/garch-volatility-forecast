# Auto-GARCH Forecasting Project

# Initialisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

import warnings
warnings.filterwarnings("ignore")

TRADING_DAYS = 252

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

    # Use a public index (not internal attrs)
    last_date = model_fit.conditional_volatility.index[-1]
    future_idx = pd.bdate_range(start=last_date, periods=horizon + 1, freq="B")[1:]
    return pd.Series(sigma, index=future_idx, name="sigma_pct")

def main():
    print("=== Auto-GARCH Forecasting Project ===")
    ticker = input("Enter ticker: ").strip().upper()
    period = input("History period for yfinance (1y/3y/5y/max): ").strip()

    # 1) Download data
    prices = fetch_prices(ticker, period=period)
    rets = make_returns(prices)

    # 2) Inspect ACF/PACF
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(prices.index, prices["Close"])
    axes[0].set_title(f"{ticker} Adjusted Close")
    axes[0].set_xlabel("Date"); axes[0].set_ylabel("Price")

    plot_acf(rets.dropna(), lags=30, ax=axes[1])
    axes[1].set_title("ACF of Returns")

    plot_pacf((rets**2).dropna(), lags=30, ax=axes[2], method="ywm")
    axes[2].set_title("PACF of Squared Returns")
    fig.suptitle("Use PACF(squared) for ARCH order (q); slow ACF decay ⇒ GARCH component (p).", fontsize=10)
    plt.tight_layout()
    plt.show()

    # 3) Suggest (p,q) but let user decide
    sp, sq, sbic = suggest_pq(rets, p_max=3, q_max=3)
    print(f"\nSuggested (p,q) by BIC (grid p<=3,q<=3): ({sp},{sq})  [BIC={sbic:.2f}]")
    try:
        p = int(input("Choose GARCH p (lags of conditional variance, typical 0–3): ").strip())
        q = int(input("Choose ARCH q (lags of squared residuals, typical 1–3): ").strip())
    except ValueError:
        print("Invalid input. Falling back to suggested (p,q).")
        p, q = sp, sq

    # 4) Fit model
    print(f"\nFitting GARCH({p},{q}) with t-distribution…")
    res = fit_garch(rets, p=p, q=q)
    print(res.summary())

    # 5) Forecast next chosen 'N' business days of volatility
    horizon = int(input("Enter forecast horizon in trading days (e.g. 7, 21): ").strip())
    sigma = forecast_vol(res, horizon=horizon)  # in percent per day

    # 6) Plot: in-sample conditional volatility + forecast
    cond_vol = res.conditional_volatility  # also in percent
    plt.figure(figsize=(12, 4))
    cond_vol.plot(label="In-sample sigma (pct)")
    sigma.plot(label="Forecast sigma (pct)")
    plt.title(f"{ticker} – GARCH({p},{q}) Daily Volatility (percent)")
    plt.ylabel("Volatility (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7) Print last values
    print("\nLast 5 in-sample daily vol (%):")
    print(cond_vol.tail().round(3))
    print("\nForecast daily vol (%):")
    print(sigma.round(3))

if __name__ == "__main__":
    main()