"""
data_layer.py  —  The archive that makes a point-in-time test possible.

Free sources give the top ~10 holdings of an ETF as of TODAY and no history, so point-in-time holdings have
to be built forward: every run appends today's snapshot to an archive (an HF dataset) and never rewrites the
past. Complete holdings from a paid vendor can be dropped in as `manual_holdings.csv` (asof, etf, ticker,
weight); for any (asof, etf) present there, the manual rows replace the archived ones.

Files in the archive dataset:
  holdings_snapshots.parquet   asof, etf, ticker, weight, source
  constituent_prices.parquet   adjusted closes, date x ticker (incremental)
  manual_holdings.csv          optional vendor holdings

Safety: "does not exist yet" is decided by explicit repo_exists / file_exists calls, never by exception class
(a network outage must not look like an empty archive and overwrite real history). If the archive cannot be
CONFIRMED, the run does not save or upload it.

The network providers (yfinance) are injected so everything else can be tested offline; they could not be
exercised in the build sandbox, which cannot reach Yahoo.
"""

import os
import json
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Providers (network)
# ---------------------------------------------------------------------------
# Yahoo exchange suffixes keep their dot (ABX.TO, HSBA.L, ASML.AS, 8306.T); a single-letter share class takes a dash (BRK.B -> BRK-B).
# A first version turned EVERY dot into a dash, so all foreign-listed holdings were stored under symbols that do not exist
# (ABX-TO, HSBA-L ...) and failed to download. T, V, L, F are treated as exchanges (Tokyo, TSX Venture, London, Frankfurt).
EXCHANGE_SUFFIXES = {"TO", "V", "L", "AS", "PA", "DE", "MI", "MC", "SW", "ST", "HE", "CO", "OL", "AX", "NZ", "T", "KS", "KQ", "HK", "SS",
                     "SZ", "JO", "SA", "MX", "TW", "TWO", "SI", "BK", "JK", "NS", "BO", "IR", "LS", "BR", "VI", "WA", "IS", "TA", "F"}


def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    for sep in (".", "-"):
        if sep in s:
            base, suf = s.rsplit(sep, 1)
            if suf in EXCHANGE_SUFFIXES:
                return base + "." + suf
            if sep == "." and len(suf) == 1:
                return base + "-" + suf
            return s
    return s


def fetch_top_holdings(etf: str) -> List[Tuple[str, float]]:
    """Top holdings of an ETF from Yahoo via yfinance (funds_data.top_holdings: indexed by Symbol, with a 'Holding Percent' column)."""
    import yfinance as yf
    th = yf.Ticker(etf).funds_data.top_holdings
    if th is None or len(th) == 0:
        return []
    col = [c for c in th.columns if "percent" in str(c).lower()]
    if not col:
        return []
    w = pd.to_numeric(th[col[0]], errors="coerce")
    if w.sum() > 1.5:                       # percent, not fraction
        w = w / 100.0
    out = []
    for sym, wt in zip(th.index, w.values):
        if isinstance(sym, str) and np.isfinite(wt) and wt > 0:
            out.append((normalize_symbol(sym), float(wt)))
    return out


def download_prices(tickers: List[str], start: str) -> pd.DataFrame:
    """Adjusted closes, date x ticker, in chunks (Yahoo throttles large requests)."""
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 80):
        chunk = tickers[i: i + 80]
        try:
            df = yf.download(chunk, start=start, auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
        if isinstance(close, pd.Series):
            close = close.to_frame(chunk[0])
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.loc[:, ~out.columns.duplicated()]


# ---------------------------------------------------------------------------
# Safe HF access
# ---------------------------------------------------------------------------
def hf_fetch(repo: str, filename: str, token: Optional[str]) -> Tuple[Optional[str], str, str]:
    """(local path | None, status 'ok'|'missing'|'error', detail)."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=token)
        if not api.repo_exists(repo_id=repo, repo_type="dataset"):
            return None, "missing", "archive dataset does not exist yet"
        if not api.file_exists(repo_id=repo, filename=filename, repo_type="dataset"):
            return None, "missing", filename + " not in the dataset yet"
        return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", token=token, force_download=True), "ok", ""
    except Exception as e:
        return None, "error", type(e).__name__ + ": " + str(e)[:200]


def hf_push(repo: str, paths: List[str], token: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    for p in paths:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p), repo_id=repo, repo_type="dataset")
    return "pushed %d files" % len(paths)


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------
HOLD_COLS = ["asof", "etf", "ticker", "weight", "source"]


class Archive:
    def __init__(self, local_dir: str = None):
        self.dir = local_dir or config.ARCHIVE_DIR
        os.makedirs(self.dir, exist_ok=True)
        self.holdings = pd.DataFrame(columns=HOLD_COLS)
        self.prices = pd.DataFrame()
        self.manual = pd.DataFrame(columns=["asof", "etf", "ticker", "weight"])
        self.status: Dict[str, str] = {}
        self.detail: Dict[str, str] = {}
        self.writable = True

    def _path(self, name):
        return os.path.join(self.dir, name)

    def load(self, repo: Optional[str] = None, token: Optional[str] = None) -> "Archive":
        sources = {config.HOLDINGS_FILE: "holdings", config.PRICES_FILE: "prices", config.MANUAL_FILE: "manual"}
        for fname, key in sources.items():
            path = self._path(fname)
            if repo and token:
                p, st, det = hf_fetch(repo, fname, token)
                self.status[key], self.detail[key] = st, det
                if st == "ok":
                    path = p
                elif st == "error" and key != "manual":
                    self.writable = False
            else:
                self.status[key] = "local" if os.path.exists(path) else "missing"
            if os.path.exists(path):
                try:
                    if key == "holdings":
                        self.holdings = pd.read_parquet(path)
                        if len(self.holdings):       # repair symbols archived by the first version (ABX-TO -> ABX.TO)
                            self.holdings["ticker"] = self.holdings["ticker"].map(normalize_symbol)
                    elif key == "prices":
                        self.prices = pd.read_parquet(path)
                    else:
                        self.manual = pd.read_csv(path)
                except Exception as e:
                    self.status[key], self.detail[key] = "error", type(e).__name__ + ": " + str(e)[:200]
                    if key != "manual":
                        self.writable = False
        return self

    def effective_holdings(self) -> pd.DataFrame:
        h = self.holdings.copy()
        if len(self.manual):
            m = self.manual.copy()
            m["source"] = "manual"
            keys = set(zip(m["asof"].astype(str), m["etf"]))
            h = h[[(str(a), e) not in keys for a, e in zip(h["asof"], h["etf"])]]
            h = pd.concat([h, m[HOLD_COLS]], ignore_index=True)
        return h

    def snapshot_today(self, etfs: List[str], asof: str, fetch: Callable = fetch_top_holdings) -> Dict:
        have = set(zip(self.holdings["asof"].astype(str), self.holdings["etf"])) if len(self.holdings) else set()
        rows, report = [], {"added": 0, "already": 0, "failed": {}}
        for e in etfs:
            if (asof, e) in have:
                report["already"] += 1
                continue
            try:
                hold = fetch(e)
            except Exception as ex:
                report["failed"][e] = type(ex).__name__ + ": " + str(ex)[:120]
                continue
            if not hold:
                report["failed"][e] = "no holdings returned"
                continue
            for t, w in hold:
                rows.append((asof, e, t, float(w), "yfinance_top10"))
            report["added"] += 1
        if rows:
            self.holdings = pd.concat([self.holdings, pd.DataFrame(rows, columns=HOLD_COLS)], ignore_index=True)
        return report

    def update_prices(self, tickers: List[str], start: str, download: Callable = download_prices) -> Dict:
        tickers = sorted(set(tickers))
        have = set(self.prices.columns) if len(self.prices) else set()
        new = [t for t in tickers if t not in have]
        old = [t for t in tickers if t in have]
        report = {"new_tickers": len(new), "refreshed": len(old), "failed": []}
        parts = []
        if new:
            parts.append(download(new, start))
        if old and len(self.prices):
            since = (self.prices.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            parts.append(download(old, since))
        parts = [p for p in parts if p is not None and len(p)]
        if parts:
            add = pd.concat(parts, axis=1)
            add = add.loc[:, ~add.columns.duplicated()]
            self.prices = add.combine_first(self.prices) if len(self.prices) else add
            self.prices = self.prices.sort_index()
        got = set(self.prices.columns) if len(self.prices) else set()
        report["failed"] = [t for t in new if t not in got]
        return report

    def save(self, repo: Optional[str] = None, token: Optional[str] = None) -> str:
        if not self.writable:
            return "NOT saved: the stored archive could not be confirmed (" + "; ".join("%s: %s" % (k, v) for k, v in self.detail.items() if self.status.get(k) == "error") + ")"
        paths = []
        if len(self.holdings):
            self.holdings["asof"] = self.holdings["asof"].astype(str)
            self.holdings.to_parquet(self._path(config.HOLDINGS_FILE), index=False)
            paths.append(self._path(config.HOLDINGS_FILE))
        if len(self.prices):
            self.prices.to_parquet(self._path(config.PRICES_FILE))
            paths.append(self._path(config.PRICES_FILE))
        if repo and token and paths:
            try:
                return hf_push(repo, paths, token)
            except Exception as e:
                return "push failed: " + type(e).__name__ + ": " + str(e)[:150]
        return "saved locally (%d files)" % len(paths)

CODE_VERSION = "2026-11-12.2"
