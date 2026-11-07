"""
validation.py  —  Does the engine find structure that is there, and stay quiet when it is not?

Three synthetic worlds with real constituent structure (see simworld.py), several seeds each, run through the
SAME panel, walk-forward models, ablation and verdict logic as live data:
  null                   no relation between structure and the next day
  baseline-expressible   an effect that return + breadth + concentration can already express (the ablation must NOT
                         credit the extra structure features)
  entropy-only           an effect carried only by contribution-entropy acceleration (the full model SHOULD win)
Reported per world: how often each verdict is issued, mean IC and mean structure-vs-baseline gain, and the share of
runs in which each planted transition type shows its planted next-day effect in the transition table.
"""

import sys
import json
import time
import numpy as np
from typing import Dict, List

import config
import panel as P
import simworld as S
import evaluate as E

WORLDS = [("null", 0.0, "flags"), ("baseline-expressible effect", 0.004, "flags"), ("entropy-only effect", 0.004, "entropy")]


def one_run(theta: float, effect: str, seed: int, n: int = 900, fast: bool = True) -> Dict:
    w = S.make_world(n=n, n_etf=12, k=40, snap_start=n // 2, ksnap=10, theta=theta, seed=seed, effect=effect)
    U = P.build_universe(w["etf_prices"], w["cons_prices"], w["holdings"])
    res = E.run_experiments(U, fast=fast)
    ev = E.evaluate_all(U, res)
    prim = E.select_primary(ev["summary"])
    v = E.verdict_for(prim, ev["summary"], ev["series"], ev["sets"]) if prim else {"verdict": "No model", "model": None}
    fam = {}
    for f in ("ridge", "lightgbm"):
        a, b = ev["summary"]["rel"]["%s_FULL" % f], ev["summary"]["rel"]["%s_COMBO" % f]
        d = ev["summary"]["ablation"][f]
        fam[f] = {"ic_full": a["all"]["ic_mean"], "ic_combo": b["all"]["ic_mean"], "d_ic_all": d["all"]["d_ic"]["mean"], "d_ic_t_all": d["all"]["d_ic"]["t"],
                  "d_ic_pit": d["pit"]["d_ic"]["mean"], "d_ic_t_pit": d["pit"]["d_ic"]["t"], "net_bps_all": a["all"]["net_bps"]}
    tt = E.transition_table(U)
    return {"seed": seed, "verdict": v["verdict"], "primary": prim, "families": fam,
            "transitions": {k: {"mean_bps": x.get("mean_next_rel_bps"), "t": x.get("t_by_day")} for k, x in tt.items()}}


def run_validation(seeds: int = 4, n: int = 900) -> Dict:
    out = {"n_days": n, "n_etfs": 12, "constituents": 40, "snapshots": "top-10 only (partial coverage)", "worlds": []}
    for label, theta, effect in WORLDS:
        runs = [one_run(theta, effect, 100 + s, n) for s in range(seeds)]
        verdicts = [r["verdict"] for r in runs]
        w = {"label": label, "theta": theta, "effect": effect, "n_runs": seeds,
             "verdicts": {v: verdicts.count(v) for v in sorted(set(verdicts))}}
        for f in ("ridge", "lightgbm"):
            for k in ("ic_full", "ic_combo", "d_ic_all", "d_ic_t_all", "d_ic_pit", "d_ic_t_pit", "net_bps_all"):
                vals = [r["families"][f][k] for r in runs if r["families"][f][k] is not None]
                w["%s_%s" % (f, k)] = float(np.mean(vals)) if vals else None
        for tname in ["broadening_strength", "hidden_deterioration", "capitulation", "leadership_rotation"]:
            vals = [r["transitions"][tname]["mean_bps"] for r in runs if r["transitions"].get(tname, {}).get("mean_bps") is not None]
            w["transition_%s_mean_bps" % tname] = float(np.mean(vals)) if vals else None
        out["worlds"].append(w)
        print(label, w["verdicts"], flush=True)
    return out


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t0 = time.time()
    res = run_validation(seeds=seeds)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open("validation_results.json", "w"), indent=1)
    print("saved validation_results.json in %ds" % res["seconds"])
