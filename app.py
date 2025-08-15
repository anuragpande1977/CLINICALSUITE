"""
Streamlit app: Upload an IPSS Excel file (4 time points) and compute
**between-group** statistics ONLY (Placebo vs USPlus).
Includes:
  • End-of-trial (final timepoint) tests on final values
  • End-of-trial tests on Δ from Baseline
  • Δ-by-visit tests for each post-baseline visit
  • **Van Elteren** pooled test on Δ across visits (stratified Wilcoxon)
  • **ANCOVA** at final timepoint: Final ~ Group + Baseline (HC3 SEs) with adjusted
    group difference and 95% CI

Within-group analyses are removed by request.

Deploy notes (Streamlit Cloud):
- Python >=3.9
- requirements.txt:
  streamlit
  pandas
  scipy>=1.9
  numpy
  openpyxl
  xlsxwriter
  matplotlib
  statsmodels

Expected columns (case-sensitive):
- 'SUBJECT ID'
- 'GROUP'  (must contain exactly two arms: 'Placebo' and 'USPlus')
- Four timepoint columns for the same IPSS scale, e.g.:
  'Baseline IPSS  Total Score', 'Day 28 IPSS  Total Score',
  'Day 56 IPSS  Total Score', 'Day 84 IPSS  Total Score'

The app auto-detects the four timepoints via regex and sorts them chronologically.
"""

import io
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="IPSS — Placebo vs USPlus", layout="wide")
st.title("IPSS (4 Time Points) — Placebo vs USPlus")
st.caption("Between-group only: final values, Δ from baseline, Δ-by-visit, Van Elteren (pooled Δ), and ANCOVA (Final ~ Group + Baseline).")

# -------------------------------
# Utilities
# -------------------------------
TIME_ORDER_MAP = {"baseline": 0, "day 28": 28, "day 56": 56, "day 84": 84}
IPSS_COL_REGEX = re.compile(r"^(baseline|day\s*28|day\s*56|day\s*84).*ipss.*total\s*score", re.IGNORECASE)
REQUIRED_ID_COLS = ["SUBJECT ID", "GROUP"]


def find_ipss_columns(columns: List[str]) -> List[str]:
    cand = [c for c in columns if IPSS_COL_REGEX.search(str(c))]
    def time_key(col: str) -> int:
        s = col.lower()
        for k, v in TIME_ORDER_MAP.items():
            if k in s:
                return v
        return 10_000
    return sorted(cand, key=time_key)


def validate_template(df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
    missing_ids = [c for c in REQUIRED_ID_COLS if c not in df.columns]
    if missing_ids:
        return False, f"Missing required columns: {missing_ids}", []
    ipss_cols = find_ipss_columns(df.columns.tolist())
    if len(ipss_cols) != 4:
        return False, "Need exactly 4 timepoint columns (Baseline/Day28/Day56/Day84 IPSS  Total Score).", ipss_cols
    return True, "", ipss_cols


def normalize_groups(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = np.where(s == "placebo", "Placebo", np.where(s == "usplus", "USPlus", np.nan))
    return pd.Series(out, index=series.index)


# -------------------------------
# Van Elteren (stratified Wilcoxon) on Δ across visits
# -------------------------------

def _wilcoxon_stratum_stats(active: np.ndarray, placebo: np.ndarray) -> Tuple[float, float, float]:
    """Return (W, E, Var) for Wilcoxon rank-sum within one stratum with tie correction."""
    x = np.asarray(active, dtype=float)
    y = np.asarray(placebo, dtype=float)
    n1, n2 = x.size, y.size
    N = n1 + n2
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled, method="average")
    W = ranks[:n1].sum()  # rank sum for Active (USPlus)
    E = n1 * (N + 1) / 2.0
    # tie correction
    vals, counts = np.unique(pooled, return_counts=True)
    tie_term = ((counts ** 3 - counts).sum()) / (N * (N - 1)) if N > 1 else 0.0
    Var = (n1 * n2 / 12.0) * ((N + 1) - tie_term)
    return float(W), float(E), float(Var)


def van_elteren_delta(usplus_by_visit: Dict[str, np.ndarray], placebo_by_visit: Dict[str, np.ndarray]) -> Tuple[Optional[float], Optional[float], List[Tuple[str, int, int]]]:
    W_sum = E_sum = Var_sum = 0.0
    counts = []
    for visit in sorted(usplus_by_visit.keys(), key=lambda s: TIME_ORDER_MAP.get(s.lower(), 9999)):
        ax = np.asarray(usplus_by_visit[visit])
        px = np.asarray(placebo_by_visit.get(visit, np.array([])))
        n1, n2 = ax.size, px.size
        if n1 < 1 or n2 < 1:
            continue
        W, E, Var = _wilcoxon_stratum_stats(ax, px)
        W_sum += W; E_sum += E; Var_sum += Var
        counts.append((visit, int(n2), int(n1)))  # (visit, N Placebo, N USPlus)
    if Var_sum <= 0 or len(counts) == 0:
        return None, None, counts
    Z = (W_sum - E_sum) / np.sqrt(Var_sum)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(Z)))
    return float(Z), float(p), counts


@st.cache_data(show_spinner=False)
def analyze_between_groups(df: pd.DataFrame, ipss_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Between-group analyses only (Placebo vs USPlus)."""
    work = df.copy()
    work["GROUP_NORM"] = normalize_groups(work["GROUP"])  # Placebo / USPlus / NaN
    # keep only the two arms of interest
    filt = work["GROUP_NORM"].isin(["Placebo", "USPlus"]) 
    work = work.loc[filt].copy()

    # counts sanity check
    group_counts = work["GROUP_NORM"].value_counts().to_frame(name="N").rename_axis("Group").reset_index()

    # ---------- END-OF-TRIAL: final values & Δ from Baseline ----------
    base, final = ipss_cols[0], ipss_cols[-1]

    # Final values at final timepoint
    end_rows = []
    for col in [final]:
        plc = work.loc[work["GROUP_NORM"] == "Placebo", col].dropna()
        usp = work.loc[work["GROUP_NORM"] == "USPlus",  col].dropna()
        n_p, n_u = len(plc), len(usp)
        note = None
        if n_p >= 2 and n_u >= 2:
            # Welch t
            try:
                _, p_t = stats.ttest_ind(plc, usp, equal_var=False, nan_policy='omit')
            except Exception:
                p_t = np.nan
            # Mann–Whitney U
            try:
                _, p_u = stats.mannwhitneyu(plc, usp, alternative='two-sided')
            except Exception:
                p_u = np.nan
            # Brunner–Munzel
            try:
                _, p_bm = stats.brunnermunzel(plc, usp, alternative='two-sided')
            except Exception:
                p_bm = np.nan
        else:
            p_t = p_u = p_bm = np.nan
            note = "Insufficient N"
        end_rows.append([col, n_p, n_u, "Welch t-test (indep)", p_t, note])
        end_rows.append([col, n_p, n_u, "Mann–Whitney U", p_u, note])
        end_rows.append([col, n_p, n_u, "Brunner–Munzel", p_bm, note])
    end_final_df = pd.DataFrame(end_rows, columns=["Time", "N Placebo", "N USPlus", "Test", "p-value", "Note"])

    # Δ from Baseline to final
    end_delta_rows = []
    d = work[["GROUP_NORM", base, final]].dropna()
    d["DELTA"] = d[final] - d[base]
    plc = d.loc[d["GROUP_NORM"] == "Placebo", "DELTA"].to_numpy()
    usp = d.loc[d["GROUP_NORM"] == "USPlus",  "DELTA"].to_numpy()
    n_p, n_u = plc.size, usp.size
    note = None
    if n_p >= 2 and n_u >= 2:
        try:
            _, p_t = stats.ttest_ind(plc, usp, equal_var=False)
        except Exception:
            p_t = np.nan
        try:
            _, p_u = stats.mannwhitneyu(plc, usp, alternative='two-sided')
        except Exception:
            p_u = np.nan
        try:
            _, p_bm = stats.brunnermunzel(plc, usp, alternative='two-sided')
        except Exception:
            p_bm = np.nan
    else:
        p_t = p_u = p_bm = np.nan
        note = "Insufficient N"
    end_delta_df = pd.DataFrame([[f"{base} → {final}", n_p, n_u, "Welch t-test (indep)", p_t, note],
                                 [f"{base} → {final}", n_p, n_u, "Mann–Whitney U", p_u, note],
                                 [f"{base} → {final}", n_p, n_u, "Brunner–Munzel", p_bm, note]],
                                columns=["Change (Δ)", "N Placebo", "N USPlus", "Test", "p-value", "Note"])    

    # ---------- Δ-by-visit (each post-baseline) ----------
    delta_rows = []
    usplus_by_visit: Dict[str, np.ndarray] = {}
    placebo_by_visit: Dict[str, np.ndarray] = {}
    for col in ipss_cols[1:]:
        d = work[["GROUP_NORM", base, col]].dropna()
        d["DELTA"] = d[col] - d[base]
        plc = d.loc[d["GROUP_NORM"] == "Placebo", "DELTA"].to_numpy()
        usp = d.loc[d["GROUP_NORM"] == "USPlus",  "DELTA"].to_numpy()
        usplus_by_visit[col] = usp
        placebo_by_visit[col] = plc
        if plc.size >= 2 and usp.size >= 2:
            try:
                _, p_t = stats.ttest_ind(plc, usp, equal_var=False)
            except Exception:
                p_t = np.nan
            try:
                _, p_u = stats.mannwhitneyu(plc, usp, alternative='two-sided')
            except Exception:
                p_u = np.nan
            try:
                _, p_bm = stats.brunnermunzel(plc, usp, alternative='two-sided')
                
            except Exception:
                p_bm = np.nan
            delta_rows.extend([[f"{base} → {col}", "Welch t-test (indep)", p_t],
                               [f"{base} → {col}", "Mann–Whitney U", p_u],
                               [f"{base} → {col}", "Brunner–Munzel", p_bm]])
    between_delta_df = pd.DataFrame(delta_rows, columns=["Change (Δ)", "Test", "p-value"]) if delta_rows else pd.DataFrame(columns=["Change (Δ)", "Test", "p-value"])           

    # ---------- Van Elteren on Δ across visits ----------
    Z, p_ve, counts = van_elteren_delta(usplus_by_visit, placebo_by_visit)
    ve_df = pd.DataFrame({"Analysis": ["Van Elteren (Δ across visits; strata = visits)"], "Z": [Z], "p-value": [p_ve]})
    counts_df = pd.DataFrame(counts, columns=["Visit (stratum)", "N Placebo", "N USPlus"]) if counts else pd.DataFrame(columns=["Visit (stratum)", "N Placebo", "N USPlus"]) 

    # ---------- ANCOVA at final timepoint ----------
    anc_rows = []
    dd = work[["GROUP_NORM", base, final]].dropna().rename(columns={base: "Baseline", final: "Final", "GROUP_NORM": "Group"})
    # ensure both groups present
    if set(dd["Group"].unique()) == {"Placebo", "USPlus"} and len(dd) >= 10:
        model = smf.ols("Final ~ Baseline + C(Group)", data=dd).fit(cov_type="HC3")
        term = "C(Group)[T.USPlus]"
        est = float(model.params.get(term, np.nan))
        se = float(model.bse.get(term, np.nan))
        pval = float(model.pvalues.get(term, np.nan))
        df_resid = float(model.df_resid)
        t_crit = stats.t.ppf(0.975, df_resid) if df_resid > 0 else np.nan
        ci_lo = est - t_crit * se if np.isfinite(t_crit) else np.nan
        ci_hi = est + t_crit * se if np.isfinite(t_crit) else np.nan
        anc_rows.append([final, dd["Group"].value_counts().get("Placebo", 0), dd["Group"].value_counts().get("USPlus", 0), est, se, ci_lo, ci_hi, pval])
    ancova_df = pd.DataFrame(anc_rows, columns=["Final timepoint", "N Placebo", "N USPlus", "Adj diff (USPlus−Placebo)", "SE (HC3)", "95% CI lo", "95% CI hi", "p-value"]) if anc_rows else pd.DataFrame(columns=["Final timepoint", "N Placebo", "N USPlus", "Adj diff (USPlus−Placebo)", "SE (HC3)", "95% CI lo", "95% CI hi", "p-value"]) 

    return {
        "GROUP_COUNTS": group_counts,
        "END_FINAL": end_final_df,
        "END_DELTA": end_delta_df,
        "DELTA_BY_VISIT": between_delta_df,
        "VAN_ELTEREN_DELTA": ve_df,
        "VAN_ELTEREN_STRATA_COUNTS": counts_df,
        "ANCOVA_FINAL": ancova_df,
        "FINAL_LABEL": pd.DataFrame({"final_timepoint": [final]}),
    }


def to_excel_with_results(original_df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        original_df.to_excel(writer, index=False, sheet_name="Input")
        start = 0
        sheet = "RESULTS"
        sections = [
            ("GROUP COUNTS (kept)", tables.get("GROUP_COUNTS")),
            ("END-OF-TRIAL — Final values (Placebo vs USPlus)", tables.get("END_FINAL")),
            ("END-OF-TRIAL — Δ from Baseline (Placebo vs USPlus)", tables.get("END_DELTA")),
            ("ANCOVA — Final ~ Group + Baseline (HC3)", tables.get("ANCOVA_FINAL")),
            ("Δ-by-visit — between-group tests", tables.get("DELTA_BY_VISIT")),
            ("VAN ELTEREN — pooled Δ across visits (stratified Wilcoxon)", tables.get("VAN_ELTEREN_DELTA")),
            ("VAN ELTEREN — strata counts", tables.get("VAN_ELTEREN_STRATA_COUNTS")),
        ]
        for title, table in sections:
            pd.DataFrame({title: []}).to_excel(writer, index=False, sheet_name=sheet, startrow=start)
            start += 1
            if table is not None and not table.empty:
                table.to_excel(writer, index=False, sheet_name=sheet, startrow=start)
                start += len(table) + 3
            else:
                pd.DataFrame({"info": ["No results"]}).to_excel(writer, index=False, sheet_name=sheet, startrow=start)
                start += 3
    return output.getvalue()


# -------------------------------
# UI
# -------------------------------
with st.sidebar:
    st.header("Upload Excel")
    uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])    
    st.markdown("This app filters to **Placebo** and **USPlus** arms only.")

if uploaded is None:
    st.info("Upload an Excel to begin.")
    st.stop()

try:
    df = pd.read_excel(uploaded, sheet_name=0)
except Exception as e:
    st.error(f"Failed to read Excel: {e}")
    st.stop()

ok, msg, ipss_cols = validate_template(df)
if not ok:
    st.error(msg)
    if ipss_cols:
        st.write("Detected IPSS-like columns:", ipss_cols)
    st.stop()

st.subheader("Detected Time Points")
st.write(ipss_cols)

# Compute
with st.spinner("Running between-group analyses (Placebo vs USPlus)..."):
    tables = analyze_between_groups(df, ipss_cols)

# End-of-trial focus first
final_tp = tables["FINAL_LABEL"].iloc[0, 0]
st.subheader(f"End-of-Trial ▸ Final timepoint: {final_tp}")
st.dataframe(tables["END_FINAL"], use_container_width=True)
st.dataframe(tables["END_DELTA"], use_container_width=True)

st.subheader("ANCOVA (Final ~ Group + Baseline)")
st.caption("Adjusted group difference (USPlus − Placebo) with HC3 SE and 95% CI.")
st.dataframe(tables["ANCOVA_FINAL"], use_container_width=True)

st.subheader("Δ-by-visit — between-group tests")
st.dataframe(tables["DELTA_BY_VISIT"], use_container_width=True)

st.subheader("Van Elteren (pooled Δ across visits; strata = visits)")
st.dataframe(tables["VAN_ELTEREN_DELTA"], use_container_width=True)
with st.expander("Stratum sizes used in Van Elteren"):
    st.dataframe(tables["VAN_ELTEREN_STRATA_COUNTS"], use_container_width=True)

# Download
excel_bytes = to_excel_with_results(df, tables)
st.download_button(
    label="Download RESULTS workbook",
    data=excel_bytes,
    file_name="ipss_between_only_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("Done. Only between-group (Placebo vs USPlus) results are shown and exported.")
