# P2-ETF-CONSTITUENT-ENTROPY-TRANSITION

Two ETFs can post the same daily return with very different insides: one lifted by most of its
constituents, the other by two or three dominant holdings. This engine measures that internal
structure — and how it is *changing* — and tests whether it says anything about **tomorrow**:
which ETFs will outperform the eligible universe next session, and which large moves are likely to
reverse. It does not just report entropy values; it ranks the universe, and it reports honestly
how much that ranking is worth.

## Read this first: the data constraint

The test you specified needs **point-in-time ETF holdings**. Free sources do not provide them:
Yahoo (via `yfinance`'s `funds_data.top_holdings`) returns only the **top ~10 holdings, as of today,
with no history**. Complete holdings and their history are paid-vendor products. So:

* **Point-in-time history is built forward.** Every run archives today's holdings snapshot to an HF
  dataset and never rewrites the past. Only days covered by a snapshot taken *at the time* count as
  evidence ("PIT days"). The first verdict needs `MIN_PIT_DAYS = 60` of them (about three months of
  daily runs); a trustworthy one needs much more.
* **Before that, history is backfilled**: the first snapshot's weights are rolled backward through
  returns (`start(j) = start(j+1)·(1+R_etf)/(1+r_i)`). It is exact for a buy-and-hold basket (tested
  to 1e-15) but it lists *today's* constituents in the past, so it is used **only as training data,
  never as evidence**. All verdicts are computed on PIT days.
* **Partial view.** With top-10 holdings, "breadth" is the advancing share of ten stocks and the
  concentration shares are within those ten. An ETF is **eligible** only if its listed constituents
  describe it (≥5 constituents, ≥30% of the fund's weight, and a constituent-implied return that
  tracks the ETF, rolling R² ≥ 0.5). Bond, gold and foreign-listed funds drop out on their own.
* **Vendor holdings can be dropped in** as `manual_holdings.csv` (`asof,etf,ticker,weight`) in the
  archive dataset; for any (asof, etf) present there they replace the archived rows. This is the
  route to the test as you specified it.
* **Yahoo's holdings may only refresh monthly or quarterly**, so consecutive daily snapshots can be
  identical. The archive records what was published when — point-in-time for free data — but not the
  fund's true daily book.

Until PIT days accumulate, any ranking shown is labelled *Provisional — do not trade*.

## What is measured (per ETF, per day, from its constituents)

With weights renormalized over constituents that have a return, contribution `c_i = w_i·r_i`,
implied return `R = Σc_i`:

| group | features |
|---|---|
| Contribution entropy | positive and negative entropy `H = −Σ p ln p` of the contribution distribution; effective contributors `exp(H)` |
| Breadth | advancing %, declining %, weight-weighted breadth |
| Synchronization | mean and dispersion of pairwise correlations (20d) |
| Dispersion | weight-weighted cross-sectional sd of constituent returns |
| Transition | entropy velocity (1d, 5d), acceleration, breadth acceleration |
| Concentration | top-1, top-3, top-5 shares of absolute contribution |
| Price divergence | ETF return − constituent-implied return (with partial coverage: the rest of the fund) |
| Leadership rotation | rank correlation of today's vs yesterday's contributions (re-ranked on the common set), top-3 contributor turnover |

Models see **within-ETF rolling z-scores (60d, baseline excludes the current day) and changes** —
never raw levels, which would just identify the ETF and its long-run drift — standardized across
the eligible ETFs each day. Every measure was checked against an independent direct calculation
(max difference 2e-16) and for causality (editing the future changes nothing earlier).
A real bug was found and fixed this way: ranking each day over its own constituents before
intersecting gave 0.74 where exact Spearman on the common set is 0.80.

## Targets and models

* **Primary target — relative return**: `R_{i,t+1} − mean_j R_{j,t+1}` over the eligible universe.
  **Secondary — absolute return**. Also an **outperform probability** (classifier) and
  **reversal risk** (after a relative move ≥ 1 sd, does the next day flip sign? AUC).
* **Three independent families, compared and never combined**: regularized linear (ridge),
  LightGBM, and a GAM (a spline per feature). Strict walk-forward: refit every 42 days on rows whose
  targets were realized before the prediction date.
* **Baselines on identical rows**: MOM (return/vol), BREADTH, CONC, and **COMBO = MOM+BREADTH+CONC**.
  The credit given to the structure is **FULL − COMBO of the same family**, paired over days.
* **Transition types** (defined in advance from information through day t): *broadening strength*
  (5d return up, breadth rising, concentration falling), *hidden deterioration* (return not down,
  breadth falling, concentration rising), *capitulation* (down today, unusually high declining share
  and many effective negative contributors), *leadership rotation* (contribution ranking and top-3
  unusually changed). For each: next-day relative-return distribution (mean, t by day, hit rate,
  reversal rate), and a separate model per type vs the pooled model on the same rows.

## Verdict (PIT days only)

`Collecting data` (< 60 PIT days) → `Supported` (IC t ≥ 1.65, structure beats the same-family COMBO
baseline with paired t ≥ 1.65, **and the daily top-3 pays after costs**) → `Suggestive` (IC t ≥ 1 and
beats the baseline t ≥ 1) → `Not supported`. Costs: fraction of picks replaced × 15 bps round trip.
Rebalancing a top-3 every day is expensive, which is why a strong IC can still be "Suggestive".

## Validation on synthetic worlds (re-runnable: `python validation.py`)

Twelve ETFs × 40 constituents, 900 days, snapshots of the top 10 only, four seeds per world, run
through the same panel, models, ablation and verdict code:

| world | verdicts | FULL − COMBO IC on PIT days (ridge / LightGBM) | top-3 net |
|---|---|---|---|
| no relation between structure and next day | Not supported ×4 | −0.006 (t −0.5) / −0.004 (t −0.3) | −9.6 bps/day |
| effect that return+breadth+concentration can already express | Not supported ×4 | −0.007 (t −0.5) / +0.010 (t +0.6) | −5.6 |
| effect carried only by entropy acceleration | Suggestive ×3, Supported ×1 | **+0.037 (t +2.6) / +0.046 (t +2.9)** | −2.8 |

So the engine stays quiet when there is nothing there, does **not** credit entropy features for
what breadth and concentration already carry, and does credit them when only they carry it. The
transition tables recover planted effects (hidden deterioration −15.8 bps, broadening +9.0,
rotation ≈ 0 where nothing was planted). Reversal AUCs sit at chance where no reversal structure
exists. Mature-archive end-to-end test: IC +0.10 (t 5.6), beats COMBO by +0.069 (t 4.5) — still only
"Suggestive" because the top-3 lost 2 bps/day after costs. Two findings from testing: a degenerate
feature (divergence when the constituents are 100% of the fund) must z-score to 0, not NaN, or it
deletes every row; and the day-one case (a single snapshot taken today) needed the backfill to seed
from the snapshot itself.

## What is NOT verified

* **The network providers** (`yfinance` top holdings and bulk price download) could not be run in
  the build sandbox. They are written against the documented API and isolated behind injected
  functions; the first Actions run is their first real test. The run log reports every ETF whose
  holdings could not be fetched, and the dashboard's *Data* tab lists them.
* **Nothing is validated on real data.** The synthetic worlds show the machinery is honest, not
  that real constituent structure predicts anything.
* The simulation's effects are planted; real effects, if any, are likely smaller and messier.
* The archive guard: "missing" is decided by explicit `repo_exists`/`file_exists`, and if the stored
  archive cannot be confirmed the run **does not save or upload** (so an outage cannot overwrite history).

## Repo structure

```
config.py          universe, eligibility, feature, model, evaluation and archive settings
data_layer.py      the holdings/price archive (HF-synced, safe), providers, manual vendor override
panel.py           PIT weights, backfill, eligibility, feature tensor, targets
features.py        the constituent-structure measures, z-scores, transition flags
models.py          ridge / LightGBM / GAM / classifiers and the walk-forward engine
evaluate.py        IC, top-3 net of costs, AUC, ablation, transition tables, per-type models, verdict
simworld.py        synthetic constituent worlds with optional planted effects
validation.py      multi-seed validation -> validation_results.json
trainer.py         archive -> panel -> models -> verdict -> results JSON -> HF
streamlit_app.py   Today / Evidence / Transitions / Data / Validation
.github/workflows/daily_run.yml   weekday cron after the US close (GitHub Actions)
```

## Running (GitHub Actions)

`python trainer.py` with the `HF_TOKEN` secret; it needs permission to create/write
`P2SAMAPA/p2-etf-constituent-entropy-transition-data` (the archive) and
`P2SAMAPA/p2-etf-constituent-entropy-transition-results` (create them first if the token cannot).
**Always replace all files together.** Every module carries a `CODE_VERSION`, and `trainer.py` refuses to run if any file is from a different release (it says which). Tickers are normalized the way Yahoo expects (`BRK.B` → `BRK-B`, but exchange suffixes keep their dot: `ABX.TO`, `HSBA.L`, `8306.T`); symbols archived by an earlier version are repaired on load.

History is capped at 2,200 days, GAM trains on the most recent 40,000 rows and refits half as
often, to keep a run well inside the 60-minute limit. This is not financial advice.
