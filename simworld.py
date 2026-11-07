"""
simworld.py  —  A synthetic world with real constituent structure, for validating the machinery.

Each ETF is a buy-and-hold portfolio of K constituents with cap-like weights; constituent returns come from
a market factor with stochastic volatility, an ETF-specific factor and idiosyncratic noise, so breadth,
concentration, entropy and correlation all vary realistically over time. The ETF return is exactly the
start-of-day-weighted sum of constituent returns (simple returns). Holdings snapshots are archived daily
from `snap_start` on, optionally only the top `ksnap` weights (partial coverage, like free data).

Planted effects (theta > 0): the next-day return of every constituent of an ETF is shifted by theta times a signal
computed from the same feature code the engine uses.
  effect='flags'   : +1 after capitulation, -1 after hidden deterioration, +0.5 after broadening strength. These are
                     functions of return, breadth and concentration, so a momentum+breadth+concentration BASELINE can
                     express them — the ablation must NOT credit the extra structure features here.
  effect='entropy' : -1 when contribution entropy is accelerating up, +1 when accelerating down. Carried only by the
                     entropy/transition features, so the full model SHOULD beat the baseline here.
theta = 0 is the null world.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

import config
import features as F


def _weights_path(base_w: np.ndarray, R: np.ndarray, Rc: np.ndarray) -> np.ndarray:
    """Start-of-day weights under buy-and-hold drift: w_{t+1} = w_t (1 + r_t) / (1 + R_t)."""
    n, K = Rc.shape
    W = np.zeros((n, K)); W[0] = base_w
    for t in range(n - 1):
        W[t + 1] = W[t] * (1 + Rc[t]) / (1 + R[t])
    return W


def make_world(n: int = 1400, n_etf: int = 14, k: int = 40, snap_start: int = 500, ksnap: Optional[int] = None,
               theta: float = 0.0, seed: int = 0, effect: str = "flags") -> Dict:
    rng = np.random.default_rng(seed)
    ksnap = k if ksnap is None else ksnap
    dates = pd.bdate_range("2019-01-02", periods=n + 1)
    lv = np.zeros(n)
    for t in range(1, n):
        lv[t] = 0.985 * lv[t - 1] + 0.12 * rng.normal()
    m = 0.008 * np.exp(lv) * rng.normal(size=n)
    cons_ret, base_w, etf_names = [], [], []
    for e in range(n_etf):
        sector = 0.005 * np.exp(0.5 * lv) * rng.normal(size=n) * rng.uniform(0.8, 1.3)
        beta = rng.uniform(0.7, 1.3, k)
        idio_sd = rng.uniform(0.010, 0.020, k)
        # idiosyncratic volatility clusters per constituent AND an ETF-level dispersion regime (breadth/dispersion vary)
        disp_reg = np.exp(0.4 * np.cumsum(0.05 * rng.normal(size=n)) * 0.0 + 0.3 * np.sin(np.arange(n) / rng.uniform(40, 90) + rng.uniform(0, 6)))
        idio = rng.normal(size=(n, k)) * idio_sd[None, :] * disp_reg[:, None]
        Rc = beta[None, :] * (m + sector)[:, None] + idio
        sig_w = rng.uniform(0.4, 1.4)
        w0 = np.exp(sig_w * rng.normal(size=k)); w0 /= w0.sum()
        cons_ret.append(Rc); base_w.append(w0); etf_names.append("E%02d" % e)

    def etf_series(Rc, w0):
        R = np.zeros(Rc.shape[0]); W = np.zeros_like(Rc); W[0] = w0
        for t in range(Rc.shape[0]):
            R[t] = (W[t] * Rc[t]).sum()
            if t < Rc.shape[0] - 1:
                W[t + 1] = W[t] * (1 + Rc[t]) / (1 + R[t])
        return R, W

    if theta != 0.0:
        shifted = []
        for e in range(n_etf):
            R, W = etf_series(cons_ret[e], base_w[e])
            raw = F.constituent_features(W, cons_ret[e], R)
            der = F.derived_features(raw, R)
            if effect == "entropy":
                # an effect carried ONLY by contribution-entropy acceleration (not expressible by return, breadth or
                # concentration): entropy accelerating up -> weaker next day, accelerating down -> stronger
                za = np.nan_to_num(der["z_hacc"])
                sig = np.where(za > 1.0, -1.0, np.where(za < -1.0, 1.0, 0.0))
            else:
                fl = F.transition_flags(der)
                sig = -1.0 * fl["hidden_deterioration"] + 1.0 * fl["capitulation"] + 0.5 * fl["broadening_strength"]
            delta = np.concatenate([[0.0], theta * sig[:-1]])                  # shift applies to day t+1
            shifted.append(cons_ret[e] + delta[:, None])
        cons_ret = shifted

    etf_ret, W_true = [], []
    for e in range(n_etf):
        R, W = etf_series(cons_ret[e], base_w[e])
        etf_ret.append(R); W_true.append(W)

    prices = {name: 100 * np.concatenate([[1.0], np.cumprod(1 + R)]) for name, R in zip(etf_names, etf_ret)}
    etf_prices = pd.DataFrame(prices, index=dates)
    tick_cols, rows = {}, []
    for e, name in enumerate(etf_names):
        for j in range(k):
            tick = "%s_C%02d" % (name, j)
            tick_cols[tick] = 100 * np.concatenate([[1.0], np.cumprod(1 + cons_ret[e][:, j])])
    cons_prices = pd.DataFrame(tick_cols, index=dates)
    for e, name in enumerate(etf_names):
        W = W_true[e]
        top = np.argsort(-W, axis=1)[:, :ksnap]
        for t in range(snap_start, n):
            for j in top[t]:
                rows.append((dates[t], name, "%s_C%02d" % (name, j), float(W[t, j]), "sim"))
    holdings = pd.DataFrame(rows, columns=["asof", "etf", "ticker", "weight", "source"])
    return {"etf_prices": etf_prices, "cons_prices": cons_prices, "holdings": holdings, "W_true": dict(zip(etf_names, W_true)),
            "dates": dates, "snap_start": snap_start}
