"""
evaluate.py  —  What does constituent structure add, and when?

Every comparison is made on IDENTICAL (date, ETF) rows (the universe mask is set by the FULL feature set), so a
baseline can never win or lose because it was scored on different data. The credit given to the structure is the
DIFFERENCE between the full model and the COMBO baseline (momentum + breadth + concentration) of the SAME model
family, on the same days, with a paired t-statistic over days.

Metrics: daily cross-sectional rank IC against the next-day relative return; top-3 vs universe on realized
next-day returns, gross and net of costs (cost = fraction of picks replaced x round trip); reversal AUC.
"""

import numpy as np
from scipy.stats import rankdata
from typing import Dict, List, Optional

import config
import features as F
import models as M


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def daily_ic(score: np.ndarray, y: np.ndarray, min_etfs: int = None) -> np.ndarray:
    min_etfs = config.MIN_ETFS_PER_DAY if min_etfs is None else min_etfs
    out = np.full(score.shape[0], np.nan)
    ok = np.isfinite(score) & np.isfinite(y)
    for t in np.where(ok.sum(axis=1) >= min_etfs)[0]:
        m = ok[t]
        a, b = rankdata(score[t, m]), rankdata(y[t, m])
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        out[t] = (a * b).sum() / den if den > 0 else np.nan
    return out


def topn_series(score: np.ndarray, ret_next: np.ndarray, univ: np.ndarray, top_n: int = None, cost_bps: float = None):
    top_n = config.TOP_N if top_n is None else top_n
    cost_bps = config.TRADING_COST_BPS if cost_bps is None else cost_bps
    n = score.shape[0]
    gross, net = np.full(n, np.nan), np.full(n, np.nan)
    prev = None
    for t in range(n):
        m = univ[t] & np.isfinite(score[t]) & np.isfinite(ret_next[t])
        if m.sum() < config.MIN_ETFS_PER_DAY:
            prev = None
            continue
        idx = np.where(m)[0]
        order = idx[np.argsort(-score[t, idx])][:top_n]
        g = ret_next[t, order].mean() - ret_next[t, idx].mean()
        repl = (top_n - len(set(order.tolist()) & prev)) / top_n if prev is not None else 0.0
        gross[t], net[t] = g, g - repl * cost_bps / 1e4
        prev = set(order.tolist())
    return gross, net


def auc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def day_sets(valid_days: np.ndarray, pit_day: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Evaluation days: all, selection, holdout, PIT-only.
    The primary model is SELECTED on the backfilled (non-PIT) days and JUDGED on the PIT days, so the two never overlap. A
    first version selected on the first 70% of all days; in a mature archive that segment contains PIT days, so the model
    chosen for looking good on some days was then graded partly on those same days, which inflated false positives.
    Until there are enough backfilled days (60) it falls back to a 70/30 split by time.
    """
    idx = np.where(valid_days)[0]
    sets = {k: np.zeros_like(valid_days) for k in ("all", "selection", "holdout", "pit")}
    sets["all"][idx] = True
    sets["pit"][idx] = pit_day[idx]
    non_pit = idx[~pit_day[idx]]
    if len(non_pit) >= 60:
        sets["selection"][non_pit] = True
        sets["holdout"][idx[pit_day[idx]]] = True
    else:
        cut = int(len(idx) * config.SELECTION_FRACTION)
        sets["selection"][idx[:cut]] = True
        sets["holdout"][idx[cut:]] = True
    return sets


def summarize(ic: np.ndarray, gross: np.ndarray, net: np.ndarray, mask: np.ndarray) -> Dict:
    i, g, nt = ic[mask], gross[mask], net[mask]
    return {"n_days": int(np.isfinite(i).sum()), "ic_mean": float(np.nanmean(i)) if np.isfinite(i).any() else None, "ic_t": tstat(i),
            "gross_bps": float(np.nanmean(g) * 1e4) if np.isfinite(g).any() else None,
            "net_bps": float(np.nanmean(nt) * 1e4) if np.isfinite(nt).any() else None, "net_t": tstat(nt)}


def paired(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Dict:
    d = (a - b)[mask]
    d = d[np.isfinite(d)]
    return {"n_days": int(len(d)), "mean": float(d.mean()) if len(d) else None, "t": tstat(d)}


def run_experiments(U: Dict, fast: bool = False, specs: Optional[List] = None) -> Dict:
    names = F.ALL_FEATURES
    idx = {k: [names.index(f) for f in v] for k, v in F.MODEL_FEATURE_SETS.items()}
    X, rows = U["X"], U["univ"]
    rd = config.REFIT_DAYS * (2 if fast else 1)
    if specs is None:
        specs = [("ridge", "MOM"), ("ridge", "BREADTH"), ("ridge", "CONC"), ("ridge", "COMBO"), ("ridge", "FULL"),
                 ("lightgbm", "COMBO"), ("lightgbm", "FULL")] + ([] if fast else [("gam", "FULL")])
    res = {"rel": {}, "abs": {}, "cls": {}, "rev": {}, "fits": {}}
    for kind, fs in specs:
        wf = M.walk_forward(X, U["y_rel"], rows, idx[fs], names, kind, refit_days=rd * (config.GAM_REFIT_MULT if kind == "gam" else 1))
        res["rel"]["%s_%s" % (kind, fs)] = wf["score"]
        res["fits"]["rel_%s_%s" % (kind, fs)] = (wf["fits"], wf["fails"])
    for kind, fs in [s for s in specs if s[1] == "FULL"]:
        wf = M.walk_forward(X, U["y_abs"], rows, idx[fs], names, kind, refit_days=rd * (config.GAM_REFIT_MULT if kind == "gam" else 1))
        res["abs"]["%s_%s" % (kind, fs)] = wf["score"]
    # "will ETF i outperform the eligible universe tomorrow?" — a probability, from a classifier
    y_cls = np.where(np.isfinite(U["y_rel"]), (U["y_rel"] > 0).astype(float), np.nan)
    for kind in ("logit", "lightgbm_clf"):
        wf = M.walk_forward(X, y_cls, rows, idx["FULL"], names, kind, refit_days=rd)
        res["cls"]["%s_FULL" % kind] = wf["score"]
    rev_specs = [("logit", k) for k in ["MOM", "BREADTH", "CONC", "COMBO", "FULL"]] + [("lightgbm_clf", "COMBO"), ("lightgbm_clf", "FULL")]
    for kind, fs in rev_specs:
        wf = M.walk_forward(X, U["y_rev"], U["big"], idx[fs], names, kind, refit_days=rd, min_rows=800, min_etfs=1)
        res["rev"]["%s_%s" % (kind, fs)] = wf["score"]
    return res


def evaluate_all(U: Dict, res: Dict) -> Dict:
    ret_next = np.vstack([U["ret"][1:], np.full((1, U["ret"].shape[1]), np.nan)])
    out = {"rel": {}, "abs": {}, "cls": {}, "series": {}, "rev": {}}
    first = next(iter(res["rel"].values()))
    valid_days = (np.isfinite(first) & np.isfinite(U["y_rel"])).sum(axis=1) >= config.MIN_ETFS_PER_DAY
    sets = day_sets(valid_days, U["pit_day"])
    out["day_counts"] = {k: int(v.sum()) for k, v in sets.items()}
    ser = {}
    for grp in ("rel", "abs", "cls"):
        for name, sc in res[grp].items():
            ic = daily_ic(sc, U["y_rel"])
            gr, nt = topn_series(sc, ret_next, U["univ"])
            ser[(grp, name)] = (ic, gr, nt)
            out[grp][name] = {k: summarize(ic, gr, nt, m) for k, m in sets.items()}
            if grp == "cls":
                yc = (U["y_rel"] > 0).astype(float)
                okr = np.isfinite(sc) & np.isfinite(U["y_rel"])
                out[grp][name]["auc_all"] = auc(yc[okr], sc[okr])
                okp = okr & sets["pit"][:, None]
                out[grp][name]["auc_pit"] = auc(yc[okp], sc[okp])
    # the ablation: structure vs the COMBO baseline of the SAME family
    abl = {}
    for fam in ("ridge", "lightgbm"):
        a, b = ser.get(("rel", "%s_FULL" % fam)), ser.get(("rel", "%s_COMBO" % fam))
        if a and b:
            abl[fam] = {k: {"d_ic": paired(a[0], b[0], m), "d_net": paired(a[2], b[2], m)} for k, m in sets.items()}
    out["ablation"] = abl
    for name, sc in res["rev"].items():
        m_rows = U["big"] & np.isfinite(sc)
        out["rev"][name] = {"auc_all": auc(U["y_rev"][m_rows], sc[m_rows]), "n_rows": int(m_rows.sum()),
                            "auc_pit": auc(U["y_rev"][m_rows & sets["pit"][:, None]], sc[m_rows & sets["pit"][:, None]]),
                            "n_rows_pit": int((m_rows & sets["pit"][:, None]).sum())}
    return {"summary": out, "series": ser, "sets": sets, "ret_next": ret_next}


def transition_table(U: Dict, mask_days: Optional[np.ndarray] = None) -> Dict:
    """Next-day relative-return distribution after each transition type, vs the rest of the universe-days."""
    ok = U["univ"] & np.isfinite(U["y_rel"])
    if mask_days is not None:
        ok = ok & mask_days[:, None]
    sign_flip = np.where(ok, (U["rel_today"] * U["y_rel"] < 0).astype(float), np.nan)
    out = {}
    base_rows = ok
    for name, fl in list(U["flags"].items()) + [("none_of_the_four", ~(U["flags"]["broadening_strength"] | U["flags"]["hidden_deterioration"] | U["flags"]["capitulation"] | U["flags"]["leadership_rotation"]))]:
        rows = ok & fl
        if rows.sum() < 30:
            out[name] = {"n_rows": int(rows.sum()), "n_days": int(rows.any(axis=1).sum())}
            continue
        per_day = np.array([U["y_rel"][t, rows[t]].mean() for t in np.where(rows.any(axis=1))[0]])
        out[name] = {"n_rows": int(rows.sum()), "n_days": int(len(per_day)), "share_of_rows": float(rows.sum() / base_rows.sum()),
                     "mean_next_rel_bps": float(U["y_rel"][rows].mean() * 1e4), "t_by_day": tstat(per_day),
                     "hit_rate": float((U["y_rel"][rows] > 0).mean()), "reversal_rate": float(np.nanmean(sign_flip[rows])),
                     "next_rel_sd_bps": float(U["y_rel"][rows].std() * 1e4)}
    return out


def pooled_spearman(score: np.ndarray, y: np.ndarray, rows: np.ndarray, block: int = 10, reps: int = 300, seed: int = 0) -> Dict:
    """Spearman over ALL rows (a type's rows are too sparse per day for a daily IC) with a date-block bootstrap interval."""
    ok = rows & np.isfinite(score) & np.isfinite(y)
    di, ei = np.where(ok)
    if len(di) < 100:
        return {"n_rows": int(len(di)), "rho": None}
    s, t = score[ok], y[ok]
    rho = float(np.corrcoef(rankdata(s), rankdata(t))[0, 1])
    days = np.unique(di)
    nb = int(np.ceil(len(days) / block))
    starts = np.arange(0, len(days), block)
    rng = np.random.default_rng(seed)
    pos = {d: np.where(di == d)[0] for d in days}
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(starts), size=nb)
        sel = np.concatenate([pos[d] for st in starts[pick] for d in days[st: st + block]])
        vals.append(np.corrcoef(rankdata(s[sel]), rankdata(t[sel]))[0, 1])
    return {"n_rows": int(len(di)), "rho": rho, "lo": float(np.percentile(vals, 5)), "hi": float(np.percentile(vals, 95))}


def per_type_models(U: Dict, pooled_score: np.ndarray, fast: bool = False) -> Dict:
    """Train a ridge on each transition type's rows alone and compare with the pooled model on the SAME rows."""
    names = F.ALL_FEATURES
    idx = [names.index(f) for f in F.FULL]
    out = {}
    for name, fl in U["flags"].items():
        rows = U["univ"] & fl
        if rows.sum() < 400:
            out[name] = {"n_rows": int(rows.sum()), "note": "too few rows for a separate model"}
            continue
        wf = M.walk_forward(U["X"], U["y_rel"], rows, idx, names, "ridge", refit_days=config.REFIT_DAYS * (2 if fast else 1),
                            min_rows=250, min_days=60, min_etfs=1)
        out[name] = {"separate": pooled_spearman(wf["score"], U["y_rel"], rows),
                     "pooled": pooled_spearman(np.where(rows, pooled_score, np.nan), U["y_rel"], rows & np.isfinite(wf["score"]))}
    return out


FAMILY_BASELINE = {"ridge": "ridge_COMBO", "lightgbm": "lightgbm_COMBO", "gam": "ridge_COMBO"}


def select_primary(summary: Dict) -> Optional[str]:
    """The FULL model with the best SELECTION-segment IC t-statistic (the first 70% of evaluation days)."""
    cands = [k for k in ("ridge_FULL", "lightgbm_FULL", "gam_FULL") if k in summary["rel"]]
    cands = [k for k in cands if summary["rel"][k]["selection"].get("n_days", 0) >= 30]
    if not cands:
        return None
    return max(cands, key=lambda k: summary["rel"][k]["selection"]["ic_t"])


def verdict_for(model: str, ev: Dict, series: Dict, sets: Dict) -> Dict:
    """
    Judged on PIT days only (the days with true point-in-time holdings):
      Collecting data : fewer than MIN_PIT_DAYS such days
      Supported       : IC t >= 1.65, structure beats the same-family COMBO baseline (paired t >= 1.65), AND the top-3 is
                        profitable after costs
      Suggestive      : IC t >= 1.0 and the structure beats the baseline (t >= 1.0)
      Not supported   : otherwise
    """
    n_pit = ev["day_counts"]["pit"]
    out = {"model": model, "n_pit_days": n_pit, "min_pit_days": config.MIN_PIT_DAYS}
    if n_pit < config.MIN_PIT_DAYS:
        out.update({"verdict": "Collecting data", "reason": "%d of %d point-in-time days collected; nothing is validated yet." % (n_pit, config.MIN_PIT_DAYS)})
        return out
    s = ev["rel"][model]["pit"]
    base = FAMILY_BASELINE[model.split("_")[0]]
    a, b = series[("rel", model)], series[("rel", base)]
    d_ic = paired(a[0], b[0], sets["pit"])
    d_net = paired(a[2], b[2], sets["pit"])
    out.update({"ic_mean": s["ic_mean"], "ic_t": s["ic_t"], "net_bps": s["net_bps"], "d_ic": d_ic, "d_net": d_net, "baseline": base})
    if s["ic_t"] >= config.IC_T_SUPPORTED and d_ic["t"] >= config.IC_T_SUPPORTED and (s["net_bps"] or 0) > 0:
        v, why = "Supported", "Significant IC, beats the baseline, and the top-3 pays after costs."
    elif s["ic_t"] >= config.IC_T_SUGGESTIVE and d_ic["t"] >= config.IC_T_SUGGESTIVE:
        v, why = "Suggestive", "Some ranking information beyond the baseline"
        if (s["net_bps"] or 0) <= 0:
            why += ", but the daily top-3 does not pay after costs (%.1f bps/day)." % (s["net_bps"] or 0)
        else:
            why += "; not yet significant enough."
    else:
        v, why = "Not supported", "On point-in-time days the structure does not rank ETFs better than the baseline (IC t = %.2f, ΔIC t = %.2f)." % (s["ic_t"], d_ic["t"])
    out.update({"verdict": v, "reason": why})
    return out

CODE_VERSION = "2026-11-12.4"
