import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="P2 Constituent-Entropy-Transition", page_icon="🧬", layout="wide",
                   initial_sidebar_state="collapsed")

PRIMARY = "#0f766e"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
VERDICT_COLOR = {"Supported": "#16a34a", "Suggestive": "#d97706", "Not supported": "#dc2626", "Collecting data": "#64748b"}

CSS = """
<style>
    .stApp { background-color: __PAGE_BG__; }
    #MainMenu, footer {visibility: hidden;}
    .app-title { font-size: 1.9rem; font-weight: 800; color: __INK__; margin: 0; }
    .app-subtitle { color: __SUBTLE__; font-size: 0.95rem; margin-bottom: 1.1rem; }
    .kpi-card { background: #ffffff; border: 1px solid __BORDER__; border-radius: 12px; padding: 0.8rem 1rem; text-align: center; }
    .kpi-value { font-size: 1.3rem; font-weight: 800; color: __INK__; }
    .kpi-label { font-size: 0.7rem; color: __SUBTLE__; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }
    .note { font-size: 0.82rem; color: __SUBTLE__; margin: 0.4rem 0 0.9rem 0; }
    .banner { background: #ffffff; border: 1px solid __BORDER__; border-left: 6px solid var(--accent);
        border-radius: 10px; padding: 0.8rem 1.1rem; font-size: 0.92rem; color: __INK__; margin: 0.5rem 0 0.9rem 0; }
    .pick-card { background: #ffffff; border: 1px solid __BORDER__; border-radius: 14px; padding: 1rem 1.2rem; text-align: center; }
    .pick-rank { font-size: 0.72rem; color: __SUBTLE__; text-transform: uppercase; letter-spacing: 0.05em; }
    .pick-ticker { font-size: 1.9rem; font-weight: 800; color: __INK__; }
    .pick-sub { font-size: 0.78rem; color: __SUBTLE__; }
    div[data-testid="stDataFrame"] { border: 1px solid __BORDER__; border-radius: 10px; overflow: hidden; }
</style>
"""
CSS = (CSS.replace("__PAGE_BG__", PAGE_BG).replace("__INK__", INK).replace("__SUBTLE__", SUBTLE).replace("__BORDER__", CARD_BORDER))
st.markdown(CSS, unsafe_allow_html=True)

RESULTS_REPO = "P2SAMAPA/p2-etf-constituent-entropy-transition-results"


def _find_latest_hf_result(repo_id):
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("entropy_results_") and f.endswith(".json"))
        if not names:
            return None, "No entropy_results_*.json files found in " + repo_id + "."
        return names[-1], None
    except Exception as e:
        return None, "Could not list files in " + repo_id + ": " + str(e)


@st.cache_data(show_spinner="Loading results...")
def load_data(_cache_key=""):
    notes = []
    local = glob.glob("entropy_results_*.json")
    if local:
        latest = sorted(local)[-1]
        try:
            with open(latest, "r") as f:
                return json.load(f), ["Loaded local file: " + latest]
        except Exception as e:
            notes.append("Found local file " + latest + " but couldn't parse it: " + str(e))
    name, err = _find_latest_hf_result(RESULTS_REPO)
    if err:
        notes.append(err)
    else:
        try:
            url = "https://huggingface.co/datasets/" + RESULTS_REPO + "/resolve/main/" + name
            headers = {}
            token = os.environ.get("HF_TOKEN")
            if token:
                headers["Authorization"] = "Bearer " + token
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json(), ["Loaded from HF dataset: " + name]
            notes.append("Fetching " + name + " returned HTTP " + str(resp.status_code) + ".")
        except Exception as e:
            notes.append("Fetching " + name + " failed: " + str(e))
    return None, notes


def kpi(label, value):
    return ('<div class="kpi-card"><div class="kpi-value">' + value + '</div><div class="kpi-label">' + label + '</div></div>')


def fnum(x, fmt="%.2f", na="n/a"):
    return na if x is None else (fmt % x)


def pick_cards(picks, sub_label):
    if not picks:
        st.info("No ranking today (too few eligible ETFs with complete features).")
        return
    cols = st.columns(len(picks))
    for c, p in zip(cols, picks):
        with c:
            st.markdown('<div class="pick-card"><div class="pick-rank">#' + str(p["rank"]) + '</div><div class="pick-ticker">' + p["etf"]
                        + '</div><div class="pick-sub">' + sub_label + ' ' + fnum(p["score"], "%+.5f") + '</div></div>', unsafe_allow_html=True)


def model_table(block, scopes=("all", "pit", "holdout")):
    rows = []
    for name, d in block.items():
        row = {"Model": name}
        for sc in scopes:
            s = d.get(sc, {})
            row["IC (" + sc + ")"] = s.get("ic_mean")
            row["t (" + sc + ")"] = s.get("ic_t")
        row["Top-3 gross bps/day (pit)"] = d.get("pit", {}).get("gross_bps")
        row["Top-3 net bps/day (pit)"] = d.get("pit", {}).get("net_bps")
        row["Top-3 net bps/day (all)"] = d.get("all", {}).get("net_bps")
        rows.append(row)
    return pd.DataFrame(rows)


def render_today(data):
    v = data.get("verdict") or {}
    verdict = v.get("verdict", "Collecting data")
    accent = VERDICT_COLOR.get(verdict, "#64748b")
    d = data.get("data", {})
    st.markdown('<div class="banner" style="--accent: ' + accent + ';"><b>' + verdict + '.</b> ' + str(data.get("status_text", "")) + ' ' + str(v.get("reason", ""))
                + '</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    n_pit = v.get("n_pit_days", d.get("pit_days", 0))
    for c, (lab, val) in zip(cols, [("Point-in-time days", str(n_pit) + " / " + str(v.get("min_pit_days", 60))), ("Eligible ETFs today", str(len(data.get("today", [])))),
                                    ("Primary model", str(data.get("primary") or "n/a")), ("Data as of", str(data.get("last_data_date", "n/a")))]):
        with c:
            st.markdown(kpi(lab, val), unsafe_allow_html=True)
    st.progress(min(1.0, float(n_pit) / float(v.get("min_pit_days", 60) or 60)), text="Point-in-time evidence collected")
    picks = data.get("picks", {})
    st.markdown("###### Next-session top 3 — relative return (primary target)")
    pick_cards(picks.get("primary_rel", []), "expected vs universe")
    st.caption("Ranked by the model chosen on the selection segment. The status above says how much that ranking is worth: until point-in-time "
               "days accumulate, it rests on backfilled holdings (an approximation) and is not validated.")
    with st.expander("Other targets and models (each compared independently, never combined)"):
        ob = picks.get("outperform_probability", {})
        for k, pk in ob.items():
            st.write("**" + k + "** — most likely to outperform the universe: " + ", ".join(p["etf"] + " (" + fnum(p["score"], "%.3f") + ")" for p in pk))
        for k, pk in picks.get("by_model_rel", {}).items():
            st.write("**" + k + "** (relative): " + ", ".join(p["etf"] for p in pk))
        for k, pk in picks.get("abs", {}).items():
            st.write("**" + k + "** (absolute return): " + ", ".join(p["etf"] for p in pk))
        for k, pk in picks.get("baseline_rel", {}).items():
            st.write("**" + k + "** (baseline): " + ", ".join(p["etf"] for p in pk))
    today = data.get("today", [])
    if today:
        st.markdown("###### Every eligible ETF today")
        df = pd.DataFrame(today)
        df["transitions"] = df["transitions"].apply(lambda x: ", ".join(x) if x else "-")
        df = df.rename(columns={"etf": "ETF", "score": "Score", "p_outperform": "P(outperform)", "transitions": "Transition today",
                                "pit_today": "PIT weights", "z_hpos": "z entropy (+)", "z_adv": "z advancing", "z_top3": "z top-3 share", "z_disp": "z dispersion"})
        st.dataframe(df.style.format({"Score": "{:+.5f}", "P(outperform)": "{:.3f}", "z entropy (+)": "{:+.2f}", "z advancing": "{:+.2f}",
                                      "z top-3 share": "{:+.2f}", "z dispersion": "{:+.2f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)


def render_evidence(data):
    m = data.get("models", {})
    st.markdown('<div class="note">Every model is scored on identical (date, ETF) rows. "pit" = days with true point-in-time holdings (the only evidence); '
                '"all" includes backfilled days (training-quality data, not evidence). The credit given to the structure is FULL minus COMBO '
                '(momentum + breadth + concentration) of the same family, with a paired t over days.</div>', unsafe_allow_html=True)
    st.markdown("###### Relative return — ranking quality")
    df = model_table(m.get("rel", {}))
    if not df.empty:
        st.dataframe(df.style.format({c: "{:+.4f}" for c in df.columns if c.startswith("IC")} | {c: "{:+.2f}" for c in df.columns if c.startswith("t (")}
                                     | {c: "{:+.1f}" for c in df.columns if "bps" in c}, na_rep="n/a"), use_container_width=True, hide_index=True)
    abl = m.get("ablation", {})
    if abl:
        rows = []
        for fam, sc in abl.items():
            for scope, d in sc.items():
                rows.append({"Family": fam, "Days": scope, "n days": d["d_ic"]["n_days"], "FULL − COMBO IC": d["d_ic"]["mean"], "t": d["d_ic"]["t"],
                             "FULL − COMBO net bps/day": (d["d_net"]["mean"] or 0) * 1e4, "t (net)": d["d_net"]["t"]})
        st.markdown("###### Does constituent structure add anything beyond momentum + breadth + concentration?")
        st.dataframe(pd.DataFrame(rows).style.format({"FULL − COMBO IC": "{:+.4f}", "t": "{:+.2f}", "FULL − COMBO net bps/day": "{:+.2f}", "t (net)": "{:+.2f}"}, na_rep="n/a"),
                     use_container_width=True, hide_index=True)
    ser = data.get("series")
    if ser:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ser["dates"], y=ser["cum_net_primary"], mode="lines", name="Primary model top-3 (net, cumulative %)", line=dict(color=PRIMARY, width=2)))
        fig.add_trace(go.Scatter(x=ser["dates"], y=ser["cum_net_baseline"], mode="lines", name="Baseline top-3 (net, cumulative %)", line=dict(color="#94a3b8", width=1.8)))
        if ser.get("pit_start_date"):
            fig.add_vline(x=ser["pit_start_date"], line_dash="dash", line_color="#334155", annotation_text="point-in-time begins")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
                          yaxis=dict(title="Cumulative net spread vs universe (%)", gridcolor="#eef2f7"), xaxis=dict(gridcolor="#eef2f7"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True, key="cum")
        st.caption("Left of the dashed line the holdings are backfilled (an approximation that lists today's constituents in the past). Daily rebalancing "
                   "is costly: the gap between a strong IC and a flat or negative net line is the cost of turnover.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###### Absolute return (secondary target)")
        df = model_table(m.get("abs", {}), scopes=("all", "pit"))
        if not df.empty:
            st.dataframe(df.style.format({c: "{:+.4f}" for c in df.columns if c.startswith("IC")} | {c: "{:+.2f}" for c in df.columns if c.startswith("t (")}
                                         | {c: "{:+.1f}" for c in df.columns if "bps" in c}, na_rep="n/a"), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("###### Will it outperform tomorrow? (classifier AUC, 0.5 = chance)")
        rows = [{"Model": k, "AUC (all)": d.get("auc_all"), "AUC (pit)": d.get("auc_pit"), "IC (pit)": d.get("pit", {}).get("ic_mean")} for k, d in m.get("cls", {}).items()]
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"AUC (all)": "{:.3f}", "AUC (pit)": "{:.3f}", "IC (pit)": "{:+.4f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)
    rev = m.get("rev", {})
    if rev:
        st.markdown("###### Reversal risk: after a large relative move, does the next day reverse? (AUC, 0.5 = chance)")
        rows = [{"Model": k, "AUC (all)": d["auc_all"], "rows": d["n_rows"], "AUC (pit)": d["auc_pit"], "rows (pit)": d["n_rows_pit"]} for k, d in rev.items()]
        st.dataframe(pd.DataFrame(rows).style.format({"AUC (all)": "{:.3f}", "AUC (pit)": "{:.3f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)


def render_transitions(data):
    st.markdown('<div class="note">The four transition types are defined in advance from information through day t: <b>broadening strength</b> '
                '(5-day return up, breadth rising, concentration falling), <b>hidden deterioration</b> (return not down, breadth falling, concentration rising), '
                '<b>capitulation</b> (down today with an unusually high declining share and many effective negative contributors) and <b>leadership rotation</b> '
                '(contribution ranking and top-3 contributors unusually changed). The table shows the next-day return relative to the universe after each.</div>',
                unsafe_allow_html=True)
    for label, key in (("All days (includes backfilled history)", "all_days"), ("Point-in-time days only", "pit_days")):
        tt = data.get("transitions", {}).get(key, {})
        rows = []
        for k, x in tt.items():
            if "mean_next_rel_bps" in x:
                rows.append({"Type": k, "ETF-days": x["n_rows"], "Days": x["n_days"], "Share of ETF-days": x["share_of_rows"], "Mean next-day rel. return (bps)": x["mean_next_rel_bps"],
                             "t (by day)": x["t_by_day"], "Hit rate": x["hit_rate"], "Reversal rate": x["reversal_rate"], "Next-day sd (bps)": x["next_rel_sd_bps"]})
            else:
                rows.append({"Type": k, "ETF-days": x.get("n_rows", 0), "Days": x.get("n_days", 0)})
        st.markdown("###### " + label)
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"Share of ETF-days": "{:.1%}", "Mean next-day rel. return (bps)": "{:+.1f}", "t (by day)": "{:+.2f}",
                                                          "Hit rate": "{:.3f}", "Reversal rate": "{:.3f}", "Next-day sd (bps)": "{:.0f}"}, na_rep="-"),
                         use_container_width=True, hide_index=True)
    pt = data.get("per_type_models", {})
    rows = []
    for k, x in pt.items():
        if x.get("separate", {}).get("rho") is not None:
            rows.append({"Type": k, "Rows": x["separate"]["n_rows"], "Separate model rho": x["separate"]["rho"], "lo": x["separate"]["lo"], "hi": x["separate"]["hi"],
                         "Pooled model rho (same rows)": x["pooled"].get("rho")})
        else:
            rows.append({"Type": k, "Rows": x.get("n_rows", 0)})
    if rows:
        st.markdown("###### A separate model per transition type vs the pooled model on the same rows")
        st.dataframe(pd.DataFrame(rows).style.format({"Separate model rho": "{:+.3f}", "lo": "{:+.3f}", "hi": "{:+.3f}", "Pooled model rho (same rows)": "{:+.3f}"}, na_rep="-"),
                     use_container_width=True, hide_index=True)
        st.caption("Spearman over all of a type's ETF-days with a 90% date-block bootstrap interval (lo, hi). A type that is not significantly better "
                   "modelled on its own does not justify its own model.")


def render_data(data):
    d = data.get("data", {})
    info = d.get("info", {})
    st.markdown('<div class="note">Free data gives only the top ~10 holdings of an ETF, as of today, with no history. Point-in-time holdings are therefore '
                'built forward: each run archives a snapshot, and only days covered by a snapshot taken at the time count as evidence. Dates before the first '
                'snapshot are backfilled by rolling the first snapshot backward through returns (training data only). Vendor holdings can be added as '
                'manual_holdings.csv (asof, etf, ticker, weight).</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for c, (lab, val) in zip(cols, [("Snapshot dates archived", str(info.get("n_snapshot_dates", 0))), ("First snapshot", str(info.get("first_snapshot") or "n/a")),
                                    ("Constituent tickers", str(info.get("n_constituent_tickers", 0))), ("PIT days (universe)", str(d.get("pit_days", 0)))]):
        with c:
            st.markdown(kpi(lab, val), unsafe_allow_html=True)
    rep = info.get("snapshot_report", {})
    if rep:
        st.write("Today's snapshot: " + str(rep.get("added", 0)) + " ETFs added, " + str(rep.get("already", 0)) + " already archived, " + str(len(rep.get("failed", {}))) + " failed.")
        if rep.get("failed"):
            with st.expander("ETFs whose holdings could not be fetched"):
                st.dataframe(pd.DataFrame([{"ETF": k, "Reason": v} for k, v in rep["failed"].items()]), use_container_width=True, hide_index=True)
    st.write("Archive: " + json.dumps(info.get("archive_status", {})) + " — " + str(info.get("archive_save", "")))
    meta = d.get("etfs", [])
    if meta:
        st.markdown("###### Which ETFs the constituents describe well enough to rank")
        df = pd.DataFrame(meta).rename(columns={"etf": "ETF", "n_constituents": "Constituents", "coverage_last": "Weight covered", "tracking_r2_last": "Tracking R²",
                                               "eligible_now": "Eligible now", "eligible_days": "Eligible days", "pit_days": "PIT days", "first_snapshot": "First snapshot"})
        st.dataframe(df.style.format({"Weight covered": "{:.0%}", "Tracking R²": "{:.2f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)
        st.caption("Eligible = at least " + "5 constituents, 30% of the fund's weight, and a constituent-implied return that tracks the ETF (R² >= 0.5). "
                   "Bond, gold and foreign-listed funds usually drop out because their holdings do not map to downloadable US tickers.")
    if d.get("skipped"):
        with st.expander("ETFs with no usable holdings"):
            st.dataframe(pd.DataFrame([{"ETF": k, "Reason": v} for k, v in d["skipped"].items()]), use_container_width=True, hide_index=True)


def render_validation(data):
    v = data.get("validation")
    if not v:
        st.info("No validation_results.json shipped with this run. Run `python validation.py` and commit the file.")
        return
    st.markdown('<div class="note">Synthetic worlds with real constituent structure (' + str(v.get("n_etfs")) + ' ETFs x ' + str(v.get("constituents")) + ' constituents, '
                + str(v.get("n_days")) + ' days, snapshots: ' + str(v.get("snapshots")) + ') run through the same panel, models, ablation and verdict logic as live data. '
                'It checks that the engine stays quiet when there is nothing there, does not credit entropy features for what breadth and concentration already carry, '
                'and does credit them when only they carry it.</div>', unsafe_allow_html=True)
    rows = []
    for w in v.get("worlds", []):
        rows.append({"World": w["label"], "Runs": w["n_runs"], "Verdicts": json.dumps(w["verdicts"]),
                     "Ridge: FULL IC": w.get("ridge_ic_full"), "Ridge: COMBO IC": w.get("ridge_ic_combo"), "Ridge: FULL-COMBO (pit)": w.get("ridge_d_ic_pit"), "t": w.get("ridge_d_ic_t_pit"),
                     "LightGBM: FULL-COMBO (pit)": w.get("lightgbm_d_ic_pit"), "t ": w.get("lightgbm_d_ic_t_pit"),
                     "Top-3 net bps/day (ridge)": w.get("ridge_net_bps_all")})
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format({"Ridge: FULL IC": "{:+.4f}", "Ridge: COMBO IC": "{:+.4f}", "Ridge: FULL-COMBO (pit)": "{:+.4f}", "t": "{:+.2f}",
                                                      "LightGBM: FULL-COMBO (pit)": "{:+.4f}", "t ": "{:+.2f}", "Top-3 net bps/day (ridge)": "{:+.1f}"}, na_rep="n/a"),
                     use_container_width=True, hide_index=True)


def main():
    if "cache_bust" not in st.session_state:
        st.session_state.cache_bust = 0
    head, refresh = st.columns([5, 1])
    with head:
        st.markdown('<div class="app-title">🧬 P2 Constituent-Entropy-Transition</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Is an ETF\'s move carried by many constituents or a few — and does the transition between the two say '
                    'something about tomorrow? Judged only on point-in-time holdings.</div>', unsafe_allow_html=True)
    with refresh:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.cache_bust += 1
            st.rerun()
    data, notes = load_data(_cache_key=str(st.session_state.cache_bust))
    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            for n in notes or ["No diagnostic information was returned."]:
                st.write("• " + n)
            st.write("Run `python trainer.py` to create a local `entropy_results_*.json`, or check the HF results dataset and HF_TOKEN.")
        return
    st.caption("🕒 Results generated: **" + str(data.get("run_date", "Unknown")) + "**")
    if notes:
        st.caption("📡 " + notes[0])
    if data.get("status") != "ok":
        st.warning(str(data.get("message", "The run did not produce a ranking.")))
        render_data(data)
        return
    t1, t2, t3, t4, t5 = st.tabs(["🎯 Today", "📊 Evidence", "🔀 Transitions", "🗄️ Data", "🧪 Validation"])
    with t1:
        render_today(data)
    with t2:
        render_evidence(data)
    with t3:
        render_transitions(data)
    with t4:
        render_data(data)
    with t5:
        render_validation(data)


if __name__ == "__main__":
    main()
