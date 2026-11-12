"""
config.py  —  Configuration for P2 CONSTITUENT-ENTROPY-TRANSITION

Hypothesis: whether an ETF's move is carried by many constituents or by a few
dominant ones — and how that is CHANGING — says something about the next
session. Two ETFs with the same return can have very different internals.

The binding constraint is data, not modelling. Free sources provide only the
TOP ~10 holdings of an ETF and only as of today (Yahoo's funds_data.top_holdings);
complete holdings, and any HISTORY of them, are paid-vendor products. A
point-in-time test therefore cannot be run on the past — it has to be BUILT
FORWARD, by archiving a holdings snapshot every day. This repo does that, and is
strict about the difference between:
  * PIT days   — dates covered by an archived snapshot taken at the time;
  * backfilled — dates before the first snapshot, filled by rolling the first
                 snapshot's weights backward through constituent returns. Usable as
                 TRAINING data (an approximation, with look-ahead in WHICH
                 constituents are listed), never as evidence.
Verdicts are computed on PIT days only. Until MIN_PIT_DAYS of them exist the engine
reports "collecting data" and any ranking it shows is labelled provisional.
Complete holdings from a vendor can be dropped in as manual_holdings.csv
(asof, etf, ticker, weight) and take precedence.
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
ARCHIVE_REPO = "P2SAMAPA/p2-etf-constituent-entropy-transition-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-constituent-entropy-transition-results"
ARCHIVE_DIR = "archive"
HOLDINGS_FILE = "holdings_snapshots.parquet"
PRICES_FILE = "constituent_prices.parquet"
MANUAL_FILE = "manual_holdings.csv"

ETFS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]

# ---------------------------------------------------------------------------
# Eligibility. An ETF is in the eligible universe on a date only if its archived
# constituents describe it: enough constituents, enough of the fund's weight, and a
# constituent-implied return that actually tracks the ETF (so bond, gold and
# foreign-listed funds, whose holdings do not map to US tickers, drop out).
# ---------------------------------------------------------------------------
MIN_CONSTITUENTS = 5
MIN_COVERAGE = 0.30
MIN_TRACKING_R2 = 0.50
ELIG_WINDOW = 252
MIN_ETFS_PER_DAY = 8

# Features: contribution c_i = w_i * r_i with weights renormalized over constituents
# that have a return that day. All structural features are used as WITHIN-ETF rolling
# z-scores and changes (never raw levels, which would just identify the ETF and its
# long-run drift), then standardized across the eligible ETFs each day.
Z_WINDOW = 60
Z_MIN = 30
CORR_WINDOW = 20
MAX_CORR_CONSTITUENTS = 30

# Transition types, defined with information through day t (pre-registered):
#   broadening strength   : 5d ETF return > 0, breadth trend > 0, concentration trend < 0
#   hidden deterioration  : 5d ETF return >= 0, breadth trend < 0, concentration trend > 0
#   capitulation          : ETF return < 0 today, z(declining share) >= 1.5, z(effective negative contributors) >= 1.0
#   leadership rotation   : z(contribution rank correlation) <= -1.0 and z(top-3 turnover) >= 0.5
TRANSITION_TREND_DAYS = 5
CAPITULATION_Z_DEC = 1.5
CAPITULATION_Z_EFFNEG = 1.0
ROTATION_Z_RANKCORR = -1.0
ROTATION_Z_TURNOVER = 0.5

# ---------------------------------------------------------------------------
# Models and walk-forward. Targets: relative next-day return (primary: the ETF's
# next-day return minus the equal-weight eligible-universe return) and absolute
# next-day return (secondary). Features at day t, target day t+1; models refit every
# REFIT_DAYS on rows whose targets were realized before the prediction date.
# Three model families are compared independently (not an ensemble):
#   ridge (regularized linear), LightGBM, GAM.
# Feature sets for the ablation: MOM, BREADTH, CONC, COMBO = MOM+BREADTH+CONC (the
# baseline the structure must beat), FULL = COMBO + entropy/sync/dispersion/
# transition/divergence/rotation.
# ---------------------------------------------------------------------------
REFIT_DAYS = 42            # refit cost matters: ~30 ETFs x thousands of days x 3 families must fit in an Actions run
MAX_HISTORY_DAYS = 2200    # the pipeline uses the most recent ~8.7 years
GAM_MAX_ROWS = 40000       # GAM trains on the most recent rows only (it is the slow family)
GAM_REFIT_MULT = 2         # and is refit half as often
MIN_TRAIN_DAYS = 250
MIN_TRAIN_ROWS = 2000
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
LGBM_PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=200,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=2)
GAM_FEATURES = ["z_hpos", "z_hneg", "z_adv", "z_top3", "z_disp", "z_avgcorr", "z_gap", "z_rankcorr"]
GAM_SPLINES = 6
GAM_LAM = 10.0

# Reversal risk: among ETF-days with a large relative move (|rel_t| >= REVERSAL_K * its
# trailing 63-day sd), does the next-day relative return have the opposite sign?
REVERSAL_K = 1.0

# Evaluation. Next-day top-3 rebalanced daily is expensive: cost = fraction of picks replaced
# x TRADING_COST_BPS round trip (the sibling engines' convention). The selection segment (first
# 70% of evaluation days) picks the model; the rest is its test.
TOP_N = 3
TRADING_COST_BPS = 15
SELECTION_FRACTION = 0.70
MIN_PIT_DAYS = 60
IC_T_SUPPORTED = 1.65
IC_T_SUGGESTIVE = 1.0
BOOT_BLOCK = 10
BOOT_REPS = 1000
BACKFILL_FROM_FIRST_SNAPSHOT = True
SERIES_STEP = 3
