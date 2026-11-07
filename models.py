"""
models.py  —  Three independent model families and a strict walk-forward engine.

ridge     regularized linear regression (alpha chosen by efficient leave-one-out each refit)
lightgbm  gradient-boosted trees, deliberately small and heavily regularized
gam       generalized additive model with a spline per feature (interpretable nonlinear effects)
Classifiers for reversal risk: logistic regression and LightGBM.

They are COMPARED, not combined. Walk-forward: models are refit every REFIT_DAYS on rows whose
next-day targets were realized before the first prediction date of the block; every prediction is made
with a model that has never seen that day or any later one.
"""

import warnings
import numpy as np
from typing import Dict, List, Optional

import config

warnings.filterwarnings("ignore")


class RidgeModel:
    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        self.mu, self.sd = X.mean(axis=0), np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        self.m = RidgeCV(alphas=config.RIDGE_ALPHAS).fit((X - self.mu) / self.sd, y)
        return self

    def predict(self, X):
        return self.m.predict((X - self.mu) / self.sd)


class LgbmModel:
    def fit(self, X, y):
        import lightgbm as lgb
        self.m = lgb.LGBMRegressor(**config.LGBM_PARAMS).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class GamModel:
    def __init__(self, cols: List[int]):
        self.cols = cols

    def fit(self, X, y):
        from pygam import LinearGAM, s
        Xs = X[:, self.cols]
        terms = s(0, n_splines=config.GAM_SPLINES)
        for i in range(1, len(self.cols)):
            terms = terms + s(i, n_splines=config.GAM_SPLINES)
        self.m = LinearGAM(terms, lam=config.GAM_LAM).fit(Xs, y)
        return self

    def predict(self, X):
        return self.m.predict(X[:, self.cols])


class LogitModel:
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = X.mean(axis=0), np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        self.m = LogisticRegression(C=0.1, max_iter=300).fit((X - self.mu) / self.sd, y.astype(int))
        return self

    def predict(self, X):
        return self.m.predict_proba((X - self.mu) / self.sd)[:, 1]


class LgbmClf:
    def fit(self, X, y):
        import lightgbm as lgb
        self.m = lgb.LGBMClassifier(**config.LGBM_PARAMS).fit(X, y.astype(int))
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


def make_model(kind: str, feat_names: List[str]):
    if kind == "ridge":
        return RidgeModel()
    if kind == "lightgbm":
        return LgbmModel()
    if kind == "gam":
        cols = [feat_names.index(f) for f in config.GAM_FEATURES if f in feat_names]
        return GamModel(cols)
    if kind == "logit":
        return LogitModel()
    if kind == "lightgbm_clf":
        return LgbmClf()
    raise ValueError(kind)


def walk_forward(X: np.ndarray, y: np.ndarray, rows: np.ndarray, feat_idx: List[int], feat_names: List[str], kind: str,
                 refit_days: int = None, min_rows: int = None, min_days: int = None, min_etfs: int = None) -> Dict:
    """
    X (n,T,F) standardized features, y (n,T) target (NaN where unknown), rows (n,T) bool = rows to predict.
    Returns score (n,T) (NaN where not predicted) and bookkeeping. The last date may be predicted with y unknown.
    """
    refit_days = config.REFIT_DAYS if refit_days is None else refit_days
    min_rows = config.MIN_TRAIN_ROWS if min_rows is None else min_rows
    min_days = config.MIN_TRAIN_DAYS if min_days is None else min_days
    min_etfs = config.MIN_ETFS_PER_DAY if min_etfs is None else min_etfs
    n, T, _ = X.shape
    score = np.full((n, T), np.nan)
    train_ok = rows & np.isfinite(y)
    test_days = np.where(rows.sum(axis=1) >= min_etfs)[0]
    fits = fails = 0
    names = [feat_names[i] for i in feat_idx]
    for b in range(0, len(test_days), refit_days):
        blk = test_days[b: b + refit_days]
        d0 = blk[0]
        tr = train_ok[:d0]
        if tr.sum() < min_rows or tr.any(axis=1).sum() < min_days:
            continue
        Xtr, ytr = X[:d0][tr][:, feat_idx], y[:d0][tr]
        if kind in ("logit", "lightgbm_clf") and len(np.unique(ytr)) < 2:
            continue
        try:
            model = make_model(kind, names).fit(Xtr, ytr)
        except Exception:
            fails += 1
            continue
        fits += 1
        for d in blk:
            m = rows[d]
            score[d, m] = model.predict(X[d][m][:, feat_idx])
    return {"score": score, "fits": fits, "fails": fails}
