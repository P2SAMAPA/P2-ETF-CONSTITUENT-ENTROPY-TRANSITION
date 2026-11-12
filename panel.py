"""
panel.py  —  From archived holdings snapshots to a model-ready, point-in-time panel.

Weights. A day's contribution decomposition uses the weights a constituent had at the START of that day,
so day t uses the latest snapshot taken strictly BEFORE day t. Dates before the first snapshot are
BACKFILLED by rolling the first snapshot's weights backward through returns (start(j) = start(j+1) *
(1+R_etf,j) / (1+r_i,j), the inverse of buy-and-hold drift). That is an approximation (it lists today's
constituents in the past) and is flagged pit=False: it may be used as training data, never as evidence.

Eligibility. An ETF is eligible on a date if its constituents describe it: enough of them, enough of the
fund's weight, and a constituent-implied return that tracks the ETF (rolling R^2). Bond, gold and
foreign-listed funds, whose holdings do not map to the downloadable US tickers, drop out by themselves.

Targets (all known only after day t): y_abs = next-day return; y_rel = next-day return minus the equal-weight
eligible-universe next-day return; reversal = the next-day relative return has the opposite sign of today's
after a large relative move.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import config
import features as F


def snapshot_matrix(snaps: pd.DataFrame, tickers: List[str]):
    """(asof datetimes (m,), S (m,K)) from long-format snapshots of ONE ETF."""
    snaps = snaps.copy()
    snaps["asof"] = pd.to_datetime(snaps["asof"])
    asof = np.sort(snaps["asof"].unique())
    col = {t: j for j, t in enumerate(tickers)}
    S = np.zeros((len(asof), len(tickers)))
    pos = {a: i for i, a in enumerate(asof)}
    for a, t, w in zip(snaps["asof"].values, snaps["ticker"].values, snaps["weight"].values):
        if t in col:
            S[pos[a], col[t]] = float(w)
    return asof, S


def pit_weights(asof: np.ndarray, S: np.ndarray, dates: pd.DatetimeIndex):
    """Start-of-day weights from the latest snapshot taken strictly before each date. Returns W (n,K), pit (n,)."""
    idx = np.searchsorted(asof, dates.values, side="left") - 1
    pit = idx >= 0
    W = np.zeros((len(dates), S.shape[1]))
    W[pit] = S[idx[pit]]
    return W, pit


def backfill_weights(W: np.ndarray, pit: np.ndarray, Rm: np.ndarray, etf_ret: np.ndarray, S0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Roll the first snapshot's weights backward through the days that have no snapshot. A snapshot taken after the close
    of day D is the start-of-day weights of day D+1, so it can seed the roll even when D+1 does not exist yet (the
    day-one case: a single snapshot taken today). `S0` is that first snapshot's weight vector.
    """
    W = W.copy()
    n = W.shape[0]
    if pit.all():
        return W
    i0 = int(np.argmax(pit)) if pit.any() else n
    nxt = W[i0].copy() if i0 < n else (S0.copy() if S0 is not None else None)
    if nxt is None:
        return W
    for j in range(i0 - 1, -1, -1):
        r = Rm[j]
        Rj = etf_ret[j] if np.isfinite(etf_ret[j]) else 0.0
        rj = np.where(np.isfinite(r), r, Rj)
        W[j] = nxt * (1.0 + Rj) / (1.0 + rj)
        nxt = W[j]
    return W


def eligibility(etf_ret: np.ndarray, implied: np.ndarray, coverage: np.ndarray, nconst: np.ndarray):
    s = pd.Series(etf_ret)
    r2 = (s.rolling(config.ELIG_WINDOW, min_periods=config.ELIG_WINDOW // 2).corr(pd.Series(implied)) ** 2).values
    elig = (nconst >= config.MIN_CONSTITUENTS) & (coverage >= config.MIN_COVERAGE) & (np.nan_to_num(r2, nan=0.0) >= config.MIN_TRACKING_R2)
    return elig, r2


def build_etf(etf: str, dates: pd.DatetimeIndex, etf_ret: np.ndarray, snaps: pd.DataFrame, cons_ret: pd.DataFrame,
              backfill: Optional[bool] = None) -> Optional[Dict]:
    backfill = config.BACKFILL_FROM_FIRST_SNAPSHOT if backfill is None else backfill
    if snaps is None or snaps.empty:
        return None
    wmean = snaps.groupby("ticker")["weight"].mean().sort_values(ascending=False)
    tickers = [t for t in wmean.index if t in cons_ret.columns]
    if len(tickers) < config.MIN_CONSTITUENTS:
        return None
    Rm = cons_ret[tickers].reindex(dates).values
    asof, S = snapshot_matrix(snaps, tickers)
    W, pit = pit_weights(asof, S, dates)
    if backfill:
        W = backfill_weights(W, pit, Rm, etf_ret, S0=S[0])
    have = pit | backfill
    W[~have] = 0.0
    raw = F.constituent_features(W, Rm, etf_ret)
    der = F.derived_features(raw, etf_ret)
    flags = F.transition_flags(der)
    valid = np.isfinite(Rm) & (W > 0)
    coverage = np.where(valid, W, 0.0).sum(axis=1)
    elig, r2 = eligibility(etf_ret, raw["implied"], coverage, raw["nconst"])
    elig &= have
    return {"etf": etf, "tickers": tickers, "W": W, "pit": pit, "have": have, "feats": der, "flags": flags, "elig": elig, "r2": r2,
            "coverage": coverage, "nconst": raw["nconst"], "first_snapshot": str(pd.Timestamp(asof[0]).date())}


def cs_standardize(X: np.ndarray, univ: np.ndarray) -> np.ndarray:
    """Per-date z-score across the universe ETFs; NaN outside the universe or on dates with too few ETFs."""
    n, T, Fn = X.shape
    out = np.full_like(X, np.nan)
    cnt = univ.sum(axis=1)
    for t in np.where(cnt >= config.MIN_ETFS_PER_DAY)[0]:
        m = univ[t]
        blk = X[t][m]
        mu, sd = blk.mean(axis=0), blk.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        out[t][m] = (blk - mu) / sd
    return out


def build_universe(etf_prices: pd.DataFrame, cons_prices: pd.DataFrame, holdings: pd.DataFrame,
                   etfs: Optional[List[str]] = None, backfill: Optional[bool] = None) -> Dict:
    dates = etf_prices.index
    etf_ret_df = etf_prices.pct_change()
    cons_ret = cons_prices.pct_change() if cons_prices is not None and len(cons_prices.columns) else pd.DataFrame(index=dates)
    etfs = list(etf_prices.columns) if etfs is None else [e for e in etfs if e in etf_prices.columns]
    built, skipped = {}, {}
    for e in etfs:
        sn = holdings[holdings["etf"] == e] if holdings is not None and len(holdings) else None
        b = build_etf(e, dates, etf_ret_df[e].values, sn, cons_ret, backfill)
        if b is None:
            skipped[e] = "no usable holdings snapshots"
        else:
            built[e] = b
    names = list(built.keys())
    n, T = len(dates), len(names)
    Fn = len(F.ALL_FEATURES)
    X = np.full((n, T, Fn), np.nan)
    ret = np.full((n, T), np.nan)
    pit = np.zeros((n, T), bool); elig = np.zeros((n, T), bool)
    flags = {k: np.zeros((n, T), bool) for k in ["broadening_strength", "hidden_deterioration", "capitulation", "leadership_rotation"]}
    for j, e in enumerate(names):
        b = built[e]
        for k, name in enumerate(F.ALL_FEATURES):
            X[:, j, k] = b["feats"][name]
        ret[:, j] = etf_ret_df[e].values
        pit[:, j], elig[:, j] = b["pit"], b["elig"]
        for k in flags:
            flags[k][:, j] = b["flags"][k]
    finite_all = np.isfinite(X).all(axis=2) & np.isfinite(ret)
    univ = elig & finite_all
    Xcs = cs_standardize(X, univ)
    ok_day = univ.sum(axis=1) >= config.MIN_ETFS_PER_DAY
    univ &= ok_day[:, None]

    nxt = np.vstack([ret[1:], np.full((1, T), np.nan)])
    y_abs = np.where(univ, nxt, np.nan)
    with np.errstate(invalid="ignore"):
        um = np.where(univ & np.isfinite(nxt), nxt, np.nan)
        mean_next = np.nanmean(um, axis=1, keepdims=True)
        y_rel = np.where(univ & np.isfinite(nxt), nxt - mean_next, np.nan)
        rel_today = np.where(univ, ret - np.nanmean(np.where(univ, ret, np.nan), axis=1, keepdims=True), np.nan)
    rel_sd = pd.DataFrame(rel_today).rolling(63, min_periods=30).std().shift(1).values
    big = univ & np.isfinite(rel_sd) & (np.abs(rel_today) >= config.REVERSAL_K * rel_sd) & np.isfinite(y_rel)
    y_rev = np.where(big, (rel_today * y_rel < 0).astype(float), np.nan)
    pit_day = (univ & pit).sum(axis=1) >= config.MIN_ETFS_PER_DAY
    meta = []
    for j, e in enumerate(names):
        b = built[e]
        meta.append({"etf": e, "n_constituents": int(len(b["tickers"])), "coverage_last": float(b["coverage"][-1]),
                     "tracking_r2_last": None if not np.isfinite(b["r2"][-1]) else float(b["r2"][-1]),
                     "eligible_now": bool(b["elig"][-1]), "eligible_days": int(b["elig"].sum()), "pit_days": int(b["pit"].sum()),
                     "first_snapshot": b["first_snapshot"]})
    return {"dates": dates, "etfs": names, "X": Xcs, "X_unstd": X, "ret": ret, "pit": pit, "elig": elig, "univ": univ, "flags": flags,
            "y_abs": y_abs, "y_rel": y_rel, "rel_today": rel_today, "y_rev": y_rev, "big": big, "pit_day": pit_day, "meta": meta,
            "skipped": skipped, "built": built}
