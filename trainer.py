"""
trainer.py  —  P2 CONSTITUENT-ENTROPY-TRANSITION

Each run:
  1. archive today's holdings snapshot for every ETF (point-in-time history is built forward, one day at a time);
  2. refresh constituent prices;
  3. build the panel (point-in-time weights where snapshots exist, backfilled before the first one), the feature
     tensor and the next-day targets;
  4. walk-forward ridge, LightGBM and GAM, plus momentum / breadth / concentration baselines, an outperform-
     probability classifier and reversal-risk classifiers — all scored on identical rows;
  5. judge on point-in-time days only; show picks with the status their evidence earns;
  6. transition-type distributions and per-type models.

`run_pipeline` is pure (data in, results out) so it is testable offline; `run_trainer` wraps it with the network and
the archive.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import features as F
import panel as P
import evaluate as E
import data_layer as D

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d):
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return _f(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


def top_picks(score_row: np.ndarray, univ_row: np.ndarray, etfs: List[str], n: int = None) -> List[Dict]:
    n = config.TOP_N if n is None else n
    m = univ_row & np.isfinite(score_row)
    if m.sum() < config.MIN_ETFS_PER_DAY:
        return []
    idx = np.where(m)[0]
    order = idx[np.argsort(-score_row[idx])]
    return [{"etf": etfs[j], "score": float(score_row[j]), "rank": int(r + 1)} for r, j in enumerate(order[:n])]


STATUS_TEXT = {"Collecting data": "Provisional — point-in-time evidence not yet collected; do not trade",
               "Not supported": "Not supported on point-in-time days — shown for reference only",
               "Suggestive": "Suggestive — weak evidence, small size at most",
               "Supported": "Supported on point-in-time days"}


def run_pipeline(etf_prices: pd.DataFrame, cons_prices: pd.DataFrame, holdings: pd.DataFrame, info: Optional[Dict] = None,
                 fast: bool = False) -> Dict:
    info = info or {}
    etf_prices = etf_prices.iloc[-config.MAX_HISTORY_DAYS:]
    U = P.build_universe(etf_prices, cons_prices, holdings)
    results: Dict = {"last_data_date": str(etf_prices.index[-1].date()), "data": {"info": info, "skipped": U["skipped"], "etfs": U["meta"]}}
    n_univ = int(U["univ"].sum())
    results["data"].update({"n_etfs_with_holdings": len(U["etfs"]), "universe_etf_days": n_univ, "pit_days": int(U["pit_day"].sum()),
                            "usable_days": int((U["univ"].sum(axis=1) >= config.MIN_ETFS_PER_DAY).sum())})
    if n_univ == 0 or results["data"]["usable_days"] < config.MIN_TRAIN_DAYS + 60:
        results["status"] = "insufficient_data"
        results["message"] = "Not enough eligible ETF-days to train: %d usable days (need > %d)." % (results["data"]["usable_days"], config.MIN_TRAIN_DAYS + 60)
        return results

    res = E.run_experiments(U, fast=fast)
    ev = E.evaluate_all(U, res)
    summ = ev["summary"]
    results["status"] = "ok"
    results["models"] = summ
    results["fits"] = res["fits"]
    prim = E.select_primary(summ)
    results["primary"] = prim
    verdict = E.verdict_for(prim, summ, ev["series"], ev["sets"]) if prim else {"verdict": "Collecting data", "reason": "No model had enough selection-segment days."}
    results["verdict"] = verdict
    results["status_text"] = STATUS_TEXT.get(verdict["verdict"], "")

    L = len(U["dates"]) - 1
    etfs = U["etfs"]
    picks = {"primary_rel": top_picks(res["rel"][prim][L], U["univ"][L], etfs) if prim else []}
    picks["by_model_rel"] = {k: top_picks(v[L], U["univ"][L], etfs) for k, v in res["rel"].items() if k.endswith("FULL")}
    picks["abs"] = {k: top_picks(v[L], U["univ"][L], etfs) for k, v in res["abs"].items()}
    picks["outperform_probability"] = {k: top_picks(v[L], U["univ"][L], etfs) for k, v in res["cls"].items()}
    picks["baseline_rel"] = {k: top_picks(v[L], U["univ"][L], etfs) for k, v in res["rel"].items() if "COMBO" in k or k.endswith("MOM")}
    results["picks"] = picks

    # today's table: every eligible ETF with its score, outperform probability and which transition types it is in
    today = []
    sc = res["rel"][prim][L] if prim else np.full(len(etfs), np.nan)
    cl = res["cls"].get("lightgbm_clf_FULL", res["cls"].get("logit_FULL"))
    names = F.ALL_FEATURES
    for j, e in enumerate(etfs):
        if not U["univ"][L, j]:
            continue
        today.append({"etf": e, "score": _f(sc[j]), "p_outperform": _f(cl[L, j]) if cl is not None else None,
                      "transitions": [k for k, v in U["flags"].items() if v[L, j]],
                      "pit_today": bool(U["pit"][L, j]),
                      "z_hpos": _f(U["X_unstd"][L, j, names.index("z_hpos")]), "z_adv": _f(U["X_unstd"][L, j, names.index("z_adv")]),
                      "z_top3": _f(U["X_unstd"][L, j, names.index("z_top3")]), "z_disp": _f(U["X_unstd"][L, j, names.index("z_disp")])})
    results["today"] = sorted(today, key=lambda r: -(r["score"] if r["score"] is not None else -9))

    results["transitions"] = {"all_days": E.transition_table(U), "pit_days": E.transition_table(U, ev["sets"]["pit"])}
    results["per_type_models"] = E.per_type_models(U, res["rel"]["ridge_FULL"], fast=fast)

    # chart series: cumulative net top-3 of the primary and its baseline, rolling IC
    if prim:
        base = E.FAMILY_BASELINE[prim.split("_")[0]]
        a, b = ev["series"][("rel", prim)], ev["series"][("rel", base)]
        ok = np.isfinite(a[2])
        idx = np.where(ok)[0][:: config.SERIES_STEP]
        cum = lambda x: np.nancumsum(np.where(np.isfinite(x), x, 0.0))
        ca, cb = cum(a[2]), cum(b[2])
        pit_start = int(np.argmax(U["pit_day"])) if U["pit_day"].any() else None
        results["series"] = {"dates": [str(U["dates"][i].date()) for i in idx], "cum_net_primary": [float(ca[i]) * 100 for i in idx],
                             "cum_net_baseline": [float(cb[i]) * 100 for i in idx],
                             "ic_roll60": [_f(pd.Series(a[0]).rolling(60, min_periods=30).mean().values[i]) for i in idx],
                             "pit_start_date": str(U["dates"][pit_start].date()) if pit_start is not None else None}
    return _clean(results)


def run_trainer(fetch: Callable = D.fetch_top_holdings, download: Callable = D.download_prices, archive: Optional[D.Archive] = None,
                prices_loader: Optional[Callable] = None, push: bool = True) -> Dict:
    logger.info("🔄 Loading ETF prices...")
    try:
        if prices_loader is None:
            from data_manager import load_master_data, validate_data
            prices_df, macro_df = load_master_data()
            validate_data(prices_df, macro_df)
        else:
            prices_df = prices_loader()
    except Exception as e:
        import traceback
        logger.error(f"Failed to load data: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return {}
    etf_prices = prices_df[[e for e in config.ETFS if e in prices_df.columns]].dropna(how="all")
    run_date = datetime.now().strftime("%Y-%m-%d")
    asof = etf_prices.index[-1].strftime("%Y-%m-%d")

    arch = archive or D.Archive().load(config.ARCHIVE_REPO if push else None, config.HF_TOKEN if push else None)
    logger.info(f"Archive status: {arch.status} (writable={arch.writable})")
    snap = arch.snapshot_today(list(etf_prices.columns), asof, fetch=fetch)
    logger.info(f"Holdings snapshot as of {asof}: {snap['added']} added, {snap['already']} already archived, {len(snap['failed'])} failed")
    hold = arch.effective_holdings()
    tickers = sorted(set(hold["ticker"])) if len(hold) else []
    prep = arch.update_prices(tickers, etf_prices.index[0].strftime("%Y-%m-%d"), download=download) if tickers else {}
    save_msg = arch.save(config.ARCHIVE_REPO if push else None, config.HF_TOKEN if push else None)
    logger.info(f"Archive save: {save_msg}")

    info = {"asof": asof, "archive_status": arch.status, "archive_detail": arch.detail, "snapshot_report": snap, "price_report": prep,
            "archive_save": save_msg, "n_snapshot_dates": int(hold["asof"].nunique()) if len(hold) else 0,
            "n_constituent_tickers": len(tickers), "first_snapshot": str(hold["asof"].min()) if len(hold) else None}
    logger.info("📊 Building panel and running walk-forward models...")
    results = run_pipeline(etf_prices, arch.prices, hold, info)
    results.update({"run_date": run_date, "algorithm": "CONSTITUENT-ENTROPY-TRANSITION"})
    vpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_results.json")
    if os.path.exists(vpath):
        try:
            results["validation"] = json.load(open(vpath))
        except Exception:
            pass
    out = f"entropy_results_{run_date}.json"
    with open(out, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"💾 Saved: {out} | status {results.get('status')} | verdict {(results.get('verdict') or {}).get('verdict')}")
    if push:
        try:
            from push_results import upload_results
            upload_results(out, hf_token=config.HF_TOKEN)
        except Exception as e:
            logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    out = run_trainer()
    if not out:                      # a failed run must be a RED run, not a green one with no output
        logger.error("Run produced no results.")
        sys.exit(1)
