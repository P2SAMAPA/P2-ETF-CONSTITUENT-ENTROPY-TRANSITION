"""
features.py  —  The internal structure of an ETF's day, from its constituents.

For ETF e on day t, with constituent weights w_i (renormalized to sum to one over the
constituents that have a return that day) and returns r_i, the contribution of constituent i is
c_i = w_i * r_i and the constituent-implied ETF return is R = sum c_i.

  Contribution entropy   positive and negative contributions are each turned into a distribution
                         p_i = c_i^+ / sum c^+ (and likewise for decliners); H = -sum p ln p;
                         effective contributors = exp(H) — "how many constituents carried the move".
  Breadth                advancing / declining share of constituents; weight-weighted breadth.
  Synchronization        mean and dispersion of pairwise correlations over a rolling window.
  Dispersion             weight-weighted cross-sectional standard deviation of constituent returns.
  Concentration          share of total absolute contribution from the top 1, 3 and 5 constituents.
  Divergence             ETF return minus constituent-implied return (what the listed constituents
                         do NOT explain — with partial coverage, the rest of the fund).
  Leadership rotation    rank correlation of today's contribution vector with yesterday's, and the
                         turnover of the top-3 contributors.
  Transition             velocity and acceleration of entropy; acceleration of breadth.

Everything the models see is a WITHIN-ETF rolling z-score or change (never a raw level: raw levels
mostly identify the ETF and its long-run drift), then standardized across the eligible ETFs each day.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

import config

BASE_VARS = ["hpos", "hneg", "effpos", "effneg", "adv", "dec", "wbreadth", "avgcorr", "corrdisp", "disp",
             "top1", "top3", "top5", "gap", "rankcorr", "top3turn"]

FEATURE_SETS = {
    "MOM": ["ret1", "ret5", "ret21", "vol21"],
    "BREADTH": ["z_adv", "z_dec", "z_wbreadth", "d1_z_wbreadth"],
    "CONC": ["z_top1", "z_top3", "z_top5", "d1_z_top3"],
    "ENTROPY": ["z_hpos", "z_hneg", "z_effpos", "z_effneg", "d1_z_hpos"],
    "SYNC": ["z_avgcorr", "d1_z_avgcorr", "z_corrdisp"],
    "DISP": ["z_disp", "d1_z_disp"],
    "TRANS": ["z_hvel1", "z_hvel5", "z_hacc", "z_wbacc"],
    "DIVERG": ["z_gap", "z_gap5"],
    "ROT": ["z_rankcorr", "z_top3turn"],
}
COMBO = FEATURE_SETS["MOM"] + FEATURE_SETS["BREADTH"] + FEATURE_SETS["CONC"]
FULL = COMBO + sum((FEATURE_SETS[k] for k in ["ENTROPY", "SYNC", "DISP", "TRANS", "DIVERG", "ROT"]), [])
MODEL_FEATURE_SETS = {"MOM": FEATURE_SETS["MOM"], "BREADTH": FEATURE_SETS["BREADTH"], "CONC": FEATURE_SETS["CONC"],
                      "COMBO": COMBO, "FULL": FULL}
ALL_FEATURES = FULL


def _entropy(part: np.ndarray):
    """
    part >= 0, shape (n,K). Returns H, normalized H, effective count exp(H).
    When NOBODY contributes on that side (every listed holding moved the other way) the entropy is 0 and the effective
    count is 0 — a defined, informative state. A first version returned NaN there, which silently deleted the ETF-day:
    with a top-10 view that happened on ~21% of ETF-days (34-38% for utilities, semis and energy), exactly the broad,
    one-directional days this hypothesis is about.
    """
    s = part.sum(axis=1)
    p = np.divide(part, s[:, None], out=np.zeros_like(part), where=s[:, None] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(p > 0, np.log(p), 0.0)
    H = np.where(s > 0, -(p * lp).sum(axis=1), 0.0)
    npos = (part > 0).sum(axis=1)
    Hn = np.where(npos > 1, H / np.log(np.maximum(npos, 2)), 0.0)
    eff = np.where(s > 0, np.exp(H), 0.0)
    return H, Hn, eff


def _rolling_corr_stats(Rc: np.ndarray, window: int):
    """Mean and dispersion (sd) of off-diagonal pairwise correlations over a trailing window, (n,) each."""
    n, K = Rc.shape
    if K < 3:
        return np.full(n, np.nan), np.full(n, np.nan)
    c = np.concatenate([np.zeros((1, K)), np.cumsum(Rc, axis=0)], axis=0)
    s1 = c[window:] - c[:-window]
    s2 = np.cumsum(np.concatenate([np.zeros((1, K)), Rc ** 2], axis=0), axis=0)
    s2 = s2[window:] - s2[:-window]
    cross = np.einsum("ti,tj->tij", Rc, Rc)
    cc = np.concatenate([np.zeros((1, K * K)), np.cumsum(cross.reshape(n, K * K), axis=0)], axis=0)
    s12 = (cc[window:] - cc[:-window]).reshape(-1, K, K)
    m1 = s1 / window
    var = np.maximum(s2 / window - m1 ** 2, 1e-14)
    cov = s12 / window - m1[:, :, None] * m1[:, None, :]
    sd = np.sqrt(var)
    corr = cov / np.maximum(sd[:, :, None] * sd[:, None, :], 1e-14)
    off = ~np.eye(K, dtype=bool)
    vals = corr[:, off]
    mean = np.full(n, np.nan); disp = np.full(n, np.nan)
    mean[window - 1:] = vals.mean(axis=1)
    disp[window - 1:] = vals.std(axis=1)
    return mean, disp


def constituent_features(W: np.ndarray, Rm: np.ndarray, etf_ret: np.ndarray) -> Dict[str, np.ndarray]:
    """Raw daily structure features for ONE ETF. W (n,K) raw weights >= 0, Rm (n,K) returns (NaN = missing), etf_ret (n,)."""
    n, K = W.shape
    valid = np.isfinite(Rm) & (W > 0)
    Wv = np.where(valid, W, 0.0)
    sw = Wv.sum(axis=1)
    Wn = np.divide(Wv, sw[:, None], out=np.zeros_like(Wv), where=sw[:, None] > 0)
    r = np.where(valid, Rm, 0.0)
    c = Wn * r
    R = c.sum(axis=1)
    nv = valid.sum(axis=1)
    out: Dict[str, np.ndarray] = {}

    hp, _, ep = _entropy(np.clip(c, 0, None))
    hn, _, en = _entropy(np.clip(-c, 0, None))
    out.update({"hpos": hp, "hneg": hn, "effpos": ep, "effneg": en})

    with np.errstate(invalid="ignore", divide="ignore"):
        adv = np.where(nv > 0, ((r > 0) & valid).sum(axis=1) / nv, np.nan)
        dec = np.where(nv > 0, ((r < 0) & valid).sum(axis=1) / nv, np.nan)
    wb = (Wn * (r > 0)).sum(axis=1) - (Wn * (r < 0)).sum(axis=1)
    out.update({"adv": adv, "dec": dec, "wbreadth": np.where(nv > 0, wb, np.nan)})
    out["disp"] = np.where(nv > 1, np.sqrt((Wn * (r - R[:, None]) ** 2).sum(axis=1)), np.nan)

    a = np.abs(c)
    sa = a.sum(axis=1)
    srt = -np.sort(-a, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["top1"] = np.where(sa > 0, srt[:, 0] / sa, np.nan)
        out["top3"] = np.where(sa > 0, srt[:, :3].sum(axis=1) / sa, np.nan)
        out["top5"] = np.where(sa > 0, srt[:, :5].sum(axis=1) / sa, np.nan)
    out["gap"] = etf_ret - np.where(nv > 0, R, np.nan)
    out["implied"] = np.where(nv > 0, R, np.nan)

    cnan = np.where(valid, c, np.nan)
    # Spearman on the constituents valid on BOTH days, re-ranked within that common set (ranking each day over its own
    # set and then intersecting gave 0.74 where the exact value was 0.80 when a return was missing on one of the days)
    m = np.isfinite(cnan[1:]) & np.isfinite(cnan[:-1])
    A = pd.DataFrame(np.where(m, cnan[1:], np.nan)).rank(axis=1).values
    B = pd.DataFrame(np.where(m, cnan[:-1], np.nan)).rank(axis=1).values
    cnt = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(m, A, 0).sum(axis=1) / cnt
        mb = np.where(m, B, 0).sum(axis=1) / cnt
        da = np.where(m, A - ma[:, None], 0.0)
        db = np.where(m, B - mb[:, None], 0.0)
        rc = (da * db).sum(axis=1) / np.sqrt((da ** 2).sum(axis=1) * (db ** 2).sum(axis=1))
    rc = np.where(cnt >= 3, rc, np.nan)
    out["rankcorr"] = np.concatenate([[np.nan], rc])
    idx3 = np.argsort(-np.where(valid, c, -np.inf), axis=1)[:, :3]
    if idx3.shape[1] == 3:
        inter = (idx3[1:, :, None] == idx3[:-1, None, :]).any(axis=2).sum(axis=1)
        tt = 1.0 - inter / 3.0
    else:
        tt = np.full(n - 1, np.nan)
    out["top3turn"] = np.concatenate([[np.nan], tt])

    avg_w = Wn.mean(axis=0)
    cols = np.argsort(-avg_w)[: min(config.MAX_CORR_CONSTITUENTS, K)]
    ac, cd = _rolling_corr_stats(np.nan_to_num(Rm[:, cols]), config.CORR_WINDOW)
    out["avgcorr"], out["corrdisp"] = ac, cd
    out["nconst"] = nv.astype(float)
    return out


def _z(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    mu = s.rolling(config.Z_WINDOW, min_periods=config.Z_MIN).mean().shift(1)     # history only: today is not in its own baseline
    sd = s.rolling(config.Z_WINDOW, min_periods=config.Z_MIN).std().shift(1)
    # sd is floored (not masked): a feature that is numerically constant — e.g. divergence when the listed
    # constituents are 100% of the fund — has z ~ 0 ("no divergence"), and must not turn into NaN and
    # delete every row that needs it
    return ((s - mu) / sd.clip(lower=1e-8)).values


def derived_features(raw: Dict[str, np.ndarray], etf_ret: np.ndarray) -> Dict[str, np.ndarray]:
    """Within-ETF z-scores, changes, transition features, momentum features and the raw ingredients of the transition flags."""
    f: Dict[str, np.ndarray] = {}
    for v in BASE_VARS:
        f["z_" + v] = _z(raw[v])
    d1 = lambda a: pd.Series(a).diff(1).values
    for v in ["wbreadth", "top3", "hpos", "avgcorr", "disp"]:
        f["d1_z_" + v] = d1(f["z_" + v])
    hp = pd.Series(raw["hpos"])
    v1 = hp.diff(1)
    f["z_hvel1"] = _z(v1.values)
    f["z_hvel5"] = _z(hp.diff(5).values)
    f["z_hacc"] = _z(v1.diff(1).values)
    wb = pd.Series(raw["wbreadth"])
    f["z_wbacc"] = _z(wb.diff(1).diff(1).values)
    f["z_gap5"] = _z(pd.Series(raw["gap"]).rolling(5, min_periods=5).sum().values)
    s = pd.Series(etf_ret)
    f["ret1"] = etf_ret
    f["ret5"] = s.rolling(5, min_periods=5).sum().values
    f["ret21"] = s.rolling(21, min_periods=21).sum().values
    f["vol21"] = np.log(np.maximum(s.rolling(21, min_periods=21).std().values, 1e-9))
    L = config.TRANSITION_TREND_DAYS
    wb5 = wb.rolling(L, min_periods=L).mean()
    t5 = pd.Series(raw["top5"]).rolling(L, min_periods=L).mean()
    f["breadth_trend"] = (wb5 - wb5.shift(L)).values
    f["conc_trend"] = (t5 - t5.shift(L)).values
    return f


def transition_flags(f: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The four pre-registered transition types for ONE ETF, each an (n,) boolean."""
    ok = lambda *a: np.all([np.isfinite(x) for x in a], axis=0)
    r5, r1 = f["ret5"], f["ret1"]
    bt, ct = f["breadth_trend"], f["conc_trend"]
    with np.errstate(invalid="ignore"):
        broad = ok(r5, bt, ct) & (r5 > 0) & (bt > 0) & (ct < 0)
        hidden = ok(r5, bt, ct) & (r5 >= 0) & (bt < 0) & (ct > 0)
        capit = ok(r1, f["z_dec"], f["z_effneg"]) & (r1 < 0) & (f["z_dec"] >= config.CAPITULATION_Z_DEC) & \
            (f["z_effneg"] >= config.CAPITULATION_Z_EFFNEG)
        rot = ok(f["z_rankcorr"], f["z_top3turn"]) & (f["z_rankcorr"] <= config.ROTATION_Z_RANKCORR) & \
            (f["z_top3turn"] >= config.ROTATION_Z_TURNOVER)
    return {"broadening_strength": broad, "hidden_deterioration": hidden, "capitulation": capit, "leadership_rotation": rot}

CODE_VERSION = "2026-11-12.4"
