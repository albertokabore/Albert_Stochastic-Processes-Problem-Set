"""
Stochastic process models for equity price simulation.

Implements three calibrated stochastic differential equation (SDE) models used
throughout the accompanying notebook / report:

    1. Geometric Brownian Motion (GBM)          -- trend + constant volatility
    2. Ornstein-Uhlenbeck process (OU)           -- mean reversion
    3. Merton Jump-Diffusion (MJD)               -- GBM + compound Poisson jumps

Every "calibrate_*" function estimates parameters from a historical log-return
(or price) series. Every "simulate_*" function produces a vectorized Monte
Carlo array of shape (n_sims, n_steps + 1), column 0 being the known S0.

All simulation functions accept a `rng` (numpy.random.Generator) so that
results are reproducible across the notebook when a fixed seed is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

TRADING_DAYS_PER_YEAR = 252

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
for _d in (DATA_DIR, FIGURES_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Data collection & preprocessing
# --------------------------------------------------------------------------- #

KAGGLE_DATASET = "jillanisofttech/tesla-stock-price"
KAGGLE_RAW_FILE = DATA_DIR / "kaggle_raw" / "Tasla_Stock_Updated_V2.csv"


def load_tesla_price_history(csv_path: Path = KAGGLE_RAW_FILE) -> pd.DataFrame:
    """
    Load daily OHLCV history for Tesla (TSLA) from the Kaggle dataset
    "jillanisofttech/tesla-stock-price" (downloaded via `kagglehub` and
    committed to data/kaggle_raw/ for reproducibility). Covers 2015-01-02
    through 2024-01-16, split-adjusted -- well beyond the assignment's
    one-year minimum. This is the sole data source used in this project;
    no data is fetched from any other API.
    """
    df = pd.read_csv(csv_path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def clean_price_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw OHLCV frame:
      - keep business-day frequency, forward-fill short gaps (holidays already
        excluded by the exchange calendar; this only patches missing rows)
      - drop rows still null after fill (e.g. leading NaNs)
      - drop non-positive or non-finite close prices (bad ticks)
      - report the number of rows affected for transparency
    """
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.asfreq("B")

    n_missing = int(out["Close"].isna().sum())
    out["Close"] = out["Close"].ffill()
    out = out.dropna(subset=["Close"])

    bad = ~np.isfinite(out["Close"]) | (out["Close"] <= 0)
    n_bad = int(bad.sum())
    out = out.loc[~bad]

    if n_missing or n_bad:
        print(f"[clean] filled {n_missing} missing trading-day rows, removed {n_bad} invalid prices")
    return out


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def summary_statistics(close: pd.Series) -> dict:
    """
    Annualized descriptive, distributional, and stationarity statistics for a
    price series. Two formal hypothesis tests supplement the descriptive
    skew/kurtosis numbers:
      - Jarque-Bera (scipy.stats.jarque_bera) tests H0: returns are Normally
        distributed, using sample skewness and kurtosis jointly. A low
        p-value is direct statistical evidence (not just visual/descriptive)
        that GBM's Normal-return assumption is violated -- motivating the
        jump-diffusion model.
      - Ljung-Box on squared returns (statsmodels.stats.diagnostic) tests
        H0: no autocorrelation in squared returns. A low p-value indicates
        volatility clustering (ARCH effects): large moves tend to be
        followed by large moves. None of GBM, OU, or MJD -- all of which
        draw i.i.d. increments each step -- can reproduce this, which is
        flagged explicitly as a shared limitation in the interpretation.
    """
    r = log_returns(close)
    adf_price = adfuller(close.dropna())
    adf_ret = adfuller(r)
    jb_stat, jb_pvalue = spstats.jarque_bera(r)
    lb = acorr_ljungbox(r ** 2, lags=[10], return_df=True)
    return {
        "n_obs": int(len(close)),
        "start": close.index.min(),
        "end": close.index.max(),
        "last_price": float(close.iloc[-1]),
        "ann_mean_return": float(r.mean() * TRADING_DAYS_PER_YEAR),
        "ann_volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "adf_stat_price": float(adf_price[0]),
        "adf_pvalue_price": float(adf_price[1]),
        "adf_stat_returns": float(adf_ret[0]),
        "adf_pvalue_returns": float(adf_ret[1]),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "ljung_box_stat_lag10": float(lb["lb_stat"].iloc[0]),
        "ljung_box_pvalue_lag10": float(lb["lb_pvalue"].iloc[0]),
    }


def gbm_analytic_confidence_interval(S0: float, params: "GBMParams", n_steps: int,
                                      dt: float = 1 / TRADING_DAYS_PER_YEAR,
                                      conf: float = 0.90) -> tuple[float, float, float]:
    """
    Closed-form (analytic) confidence interval for GBM's terminal price,
    derived from the exact log-normal distribution of S_t rather than Monte
    Carlo sampling: log(S_t) ~ Normal(log(S0) + (mu - sigma^2/2)t, sigma^2*t).
    Used as an independent cross-check that the Monte Carlo simulator
    (`simulate_gbm`) is implemented correctly -- the empirical MC percentiles
    should closely match these analytic percentiles for a large n_sims.

    Returns (lower, median, upper) at the given confidence level.
    """
    t = n_steps * dt
    mean_log = np.log(S0) + (params.mu - 0.5 * params.sigma ** 2) * t
    std_log = params.sigma * np.sqrt(t)
    alpha = 1 - conf
    lower = float(np.exp(spstats.norm.ppf(alpha / 2, mean_log, std_log)))
    median = float(np.exp(mean_log))
    upper = float(np.exp(spstats.norm.ppf(1 - alpha / 2, mean_log, std_log)))
    return lower, median, upper


def backtest_model(close: pd.Series, holdout_days: int, calibrate_fn, simulate_fn,
                    n_sims: int = 1000, seed: int = 42) -> dict:
    """
    Out-of-sample validation: fit a model on all data up to `holdout_days`
    before the end of the series, simulate `holdout_days` forward, and check
    where the price actually realized over the holdout period falls within
    the simulated distribution. A well-calibrated model should place the
    realized outcome near the middle of its distribution (percentile rank
    well inside [5, 95]) more often than not; a percentile rank near 0 or 100
    is evidence the model over- or under-estimated risk for that period.

    Returns the realized terminal price, its percentile rank among the
    simulated terminal prices, and whether it fell inside the model's own
    90% confidence interval.
    """
    train = close.iloc[:-holdout_days]
    actual_path = close.iloc[-holdout_days - 1:]
    S0 = float(train.iloc[-1])
    actual_terminal = float(actual_path.iloc[-1])

    params = calibrate_fn(train)
    rng = np.random.default_rng(seed)
    paths = simulate_fn(S0, params, holdout_days, n_sims, rng=rng)
    terminal = paths[:, -1]

    percentile_rank = float((terminal < actual_terminal).mean() * 100)
    p05, p95 = np.percentile(terminal, [5, 95])
    return {
        "train_end": train.index[-1],
        "S0": S0,
        "actual_terminal": actual_terminal,
        "sim_mean_terminal": float(terminal.mean()),
        "sim_p05": float(p05),
        "sim_p95": float(p95),
        "percentile_rank_of_actual": percentile_rank,
        "inside_90pct_interval": bool(p05 <= actual_terminal <= p95),
    }


# --------------------------------------------------------------------------- #
# 2. Geometric Brownian Motion:  dS = mu*S*dt + sigma*S*dW
# --------------------------------------------------------------------------- #

@dataclass
class GBMParams:
    mu: float      # annualized drift
    sigma: float   # annualized volatility


def calibrate_gbm(close: pd.Series) -> GBMParams:
    r = log_returns(close)
    mu = r.mean() * TRADING_DAYS_PER_YEAR + 0.5 * (r.std(ddof=1) ** 2) * TRADING_DAYS_PER_YEAR
    sigma = r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return GBMParams(mu=float(mu), sigma=float(sigma))


def simulate_gbm(S0: float, params: GBMParams, n_steps: int, n_sims: int,
                  dt: float = 1 / TRADING_DAYS_PER_YEAR, rng: np.random.Generator | None = None) -> np.ndarray:
    """Exact-solution Monte Carlo: S_t = S0 * exp((mu - sigma^2/2)t + sigma*W_t)."""
    rng = rng or np.random.default_rng()
    z = rng.standard_normal((n_sims, n_steps))
    increments = (params.mu - 0.5 * params.sigma ** 2) * dt + params.sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_sims, 1)), log_paths]))
    return paths


# --------------------------------------------------------------------------- #
# 3. Ornstein-Uhlenbeck (mean-reverting):  dX = theta*(mu - X)*dt + sigma*dW
# --------------------------------------------------------------------------- #

@dataclass
class OUParams:
    theta: float   # speed of mean reversion (annualized)
    mu: float      # long-run mean level (price)
    sigma: float   # annualized volatility


def calibrate_ou(close: pd.Series, dt: float = 1 / TRADING_DAYS_PER_YEAR) -> OUParams:
    """
    Calibrate via the discretized OU solution, which is a Gaussian AR(1)
    process:  X_{t+1} = a + b*X_t + eps_t,  b = exp(-theta*dt).
    theta, mu and sigma are recovered analytically from the OLS fit (a, b,
    residual std) -- this is the standard Lo (1988) / Vasicek estimator.
    """
    x = close.values
    x_t, x_next = x[:-1], x[1:]
    X = add_constant(x_t)
    fit = OLS(x_next, X).fit()
    a, b = fit.params[0], fit.params[1]
    resid_std = float(np.std(fit.resid, ddof=2))

    b = min(max(b, 1e-6), 1 - 1e-6)  # keep in (0,1) so theta is finite & positive
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = resid_std * np.sqrt(2 * theta / (1 - b ** 2))
    return OUParams(theta=float(theta), mu=float(mu), sigma=float(sigma))


def simulate_ou(X0: float, params: OUParams, n_steps: int, n_sims: int,
                 dt: float = 1 / TRADING_DAYS_PER_YEAR, rng: np.random.Generator | None = None) -> np.ndarray:
    """Exact discretization of the OU transition density (no Euler bias)."""
    rng = rng or np.random.default_rng()
    theta, mu, sigma = params.theta, params.mu, params.sigma
    decay = np.exp(-theta * dt)
    cond_std = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta)) if theta > 0 else sigma * np.sqrt(dt)

    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = X0
    z = rng.standard_normal((n_sims, n_steps))
    for t in range(n_steps):
        paths[:, t + 1] = mu + (paths[:, t] - mu) * decay + cond_std * z[:, t]
    return paths


# --------------------------------------------------------------------------- #
# 4. Merton Jump-Diffusion:  dS/S = (mu - lambda*k)dt + sigma*dW + d(sum jumps)
# --------------------------------------------------------------------------- #

@dataclass
class JumpDiffusionParams:
    mu: float        # annualized diffusive drift (jump-compensated)
    sigma: float      # annualized diffusive volatility (jumps excluded)
    lam: float       # jump intensity, jumps per year
    mu_j: float      # mean log-jump size
    sigma_j: float   # std of log-jump size


def calibrate_jump_diffusion(close: pd.Series, threshold_std: float = 3.0) -> JumpDiffusionParams:
    """
    Simple, transparent method-of-moments calibration:
      1. Flag daily log-returns more than `threshold_std` standard deviations
         from the mean as "jump days".
      2. Estimate the jump distribution (mean, std) from those observations.
      3. Estimate the diffusive part (mu, sigma) from the remaining returns.
      4. Jump intensity lambda = (jump days / total days) * 252.
    """
    r = log_returns(close)
    mean, std = r.mean(), r.std(ddof=1)
    is_jump = (r - mean).abs() > threshold_std * std

    n = len(r)
    n_jumps = int(is_jump.sum())
    lam = (n_jumps / n) * TRADING_DAYS_PER_YEAR if n_jumps > 0 else 0.0

    if n_jumps >= 2:
        mu_j = float(r[is_jump].mean())
        sigma_j = float(r[is_jump].std(ddof=1))
    else:
        mu_j, sigma_j = 0.0, 0.0

    diffusive = r[~is_jump]
    sigma = float(diffusive.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1  # expected relative jump size
    mu = float(diffusive.mean() * TRADING_DAYS_PER_YEAR + 0.5 * (diffusive.std(ddof=1) ** 2) * TRADING_DAYS_PER_YEAR
               - lam * k)
    return JumpDiffusionParams(mu=mu, sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j)


def simulate_jump_diffusion(S0: float, params: JumpDiffusionParams, n_steps: int, n_sims: int,
                             dt: float = 1 / TRADING_DAYS_PER_YEAR,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    k = np.exp(params.mu_j + 0.5 * params.sigma_j ** 2) - 1
    drift = (params.mu - params.lam * k - 0.5 * params.sigma ** 2) * dt

    z = rng.standard_normal((n_sims, n_steps))
    diffusion = drift + params.sigma * np.sqrt(dt) * z

    n_jumps_per_step = rng.poisson(params.lam * dt, size=(n_sims, n_steps))
    max_jumps = int(n_jumps_per_step.max())
    jump_component = np.zeros((n_sims, n_steps))
    if max_jumps > 0 and params.sigma_j >= 0:
        jump_sizes = rng.normal(params.mu_j, max(params.sigma_j, 1e-8), size=(n_sims, n_steps, max_jumps))
        mask = np.arange(max_jumps)[None, None, :] < n_jumps_per_step[:, :, None]
        jump_component = (jump_sizes * mask).sum(axis=2)

    increments = diffusion + jump_component
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_sims, 1)), log_paths]))
    return paths


# --------------------------------------------------------------------------- #
# 5. Monte Carlo path statistics
# --------------------------------------------------------------------------- #

def path_statistics(paths: np.ndarray, S0: float) -> dict:
    """Terminal-value risk/return statistics for a Monte Carlo path array."""
    terminal = paths[:, -1]
    ret = terminal / S0 - 1
    var95 = float(-np.percentile(ret, 5) * 100)
    mean_return_pct = float(ret.mean() * 100)
    return {
        "mean_terminal": float(terminal.mean()),
        "median_terminal": float(np.median(terminal)),
        "std_terminal": float(terminal.std(ddof=1)),
        "p05_terminal": float(np.percentile(terminal, 5)),
        "p95_terminal": float(np.percentile(terminal, 95)),
        "mean_return_pct": mean_return_pct,
        "prob_loss_pct": float((terminal < S0).mean() * 100),
        "VaR_95_pct": var95,  # 95% 1-horizon Value at Risk
        "CVaR_95_pct": float(-ret[ret <= np.percentile(ret, 5)].mean() * 100),
        # Return-to-risk ratio: expected return per unit of downside tail risk (VaR).
        # Higher is better; lets investors compare models/horizons on a risk-adjusted basis
        # rather than on expected return alone.
        "return_to_VaR_ratio": float(mean_return_pct / var95) if var95 > 0 else float("nan"),
    }
