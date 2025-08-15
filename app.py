"""
Streamlit app: Upload an IPSS Excel file (4 time points) and compute
parametric & non-parametric p-values within & between groups.
NOW ADDS: Δ-by-visit tests AND a Van Elteren test (stratified Wilcoxon)
that pools **change-from-baseline** across post-baseline visits
(typically Day 28, 56, 84) into a single between-group p-value.

Requirements (requirements.txt)
  streamlit
  pandas
  scipy
  numpy
  openpyxl
  xlsxwriter
  matplotlib

Template columns (case-sensitive):
- SUBJECT ID
- GROUP  (exact 'Placebo' treated as Placebo; everything else → Active)
- Four timepoint columns of the same IPSS scale, e.g.
  Baseline IPSS  Total Score, Day 28 IPSS  Total Score,
  Day 56 IPSS  Total Score, Day 84 IPSS  Total Score
"""

import io
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="IPSS P-Values (4 Time Points)", layout="wide")
st.title("IPSS (4 Time Points): P-Values Explorer — Van Elteren Enabled")
st.caption("Upload your Excel. The app computes within-/between-group tests, Δ-by-visit non-parametric p-values, and a pooled Van Elteren p-value.")

# -------------------------------
# Utilities
# -------------------------------
TIME_ORDER_MAP = {"baseline": 0, "day 28": 28, "day 56": 56, "day 84": 84}
IPSS_COL_REGEX = re.compile(r"^(baseline|day\s*28|day\s*56|day\s*84).*ipss.*total\s*score", re.IGNORECASE)
REQUIRED_ID_COLS = ["SUBJECT ID", "GROUP"]


def find_ipss_columns(columns: List[str]) -> List[str]:
    cand = []
    for c in columns:
        if IPSS_COL_REGEX.search(str(c)):
            cand.append(c)
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
        return False, "Could not detect exactly 4 IPSS timepoint columns (Baseline/Day 28/Day 56/Day 84 IPSS  Total Score).", ipss_cols
    return True, "", ipss_cols


def label_groups(series: pd.Series) -> pd.Series:
    return np.where(series.astype(str).str.strip().str.lower() == "placebo", "Placebo", "Active")


# -------------------------------
# Van Elteren (stratified Wilcoxon) on Δ across visits
# -------------------------------

def _wilcoxon_stratum_stats(active: np.ndarray, placebo: np.ndarray) -> Tuple[float, float, float]:
    """Return (W, E, Var) for Wilcoxon rank-sum within one stratum with tie correction.
    W is sum of ranks for Active using average ranks.
    """
    x = np.asarray(active, dtype=float)
    y = np.asarray(placebo, dtype=float)
    n1, n2 = x.size, y.size
    N = n1 + n2
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled, method="average")
    W = ranks[:n1].sum()
    # Expected value under H0
    E = n1 * (N + 1) / 2.0
    # Tie correction term
    # counts of equal values in pooled
    vals, counts = np.unique(pooled, return_counts=True)
    tie_term = ((counts ** 3 - counts).sum()) / (N * (N - 1)) if N > 1 else 0.0
    Var = (n1 * n2 / 12.0) * ((N + 1) - tie_term)
    return float(W), float(E), float(Var)


def van_elteren_delta(active_by_visit: Dict[str, np.ndarray], placebo_by_visit: Dict[str, np.ndarray]) -> Tuple[Optional[float], Optional[float], List[Tuple[str, int, int]]]:
    """Compute Van Elteren Z and two-sided p-value across visits using Δs.
    Returns (Z, p, strata_counts_list[ (visit, n_placebo, n_active), ... ]).
    Skips strata with <1 obs in either arm. If all skipped or Var_sum=0 -> (None, None, counts).
    """
    W_sum = 0.0
    E_sum = 0.0
    Var_sum = 0.0
    counts = []
    for visit in sorted(active_by_visit.keys(), key=lambda s: TIME_ORDER_MAP.get(s.lower(), 9999)):
        ax = np.asarray(active_by_visit[visit])
        px = np.asarray(placebo_by_visit.get(visit, np.array([])))
        n1, n2 = ax.size, px.size
        if n1 < 1 or n2 < 1:
            continue
        W, E, Var = _wilcoxon_stratum_stats(ax, px)
        W_sum += W
        E_sum += E
        Var_sum += Var
        counts.append((visit, int(n2), int(n1)))  # (visit, N Placebo, N Active)
    if Var_sum <= 0 or len(counts) == 0:
        return None, None, counts
    Z = (W_sum - E_sum) / np.sqrt(Var_sum)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(Z)))
    return float(Z), float(p), counts


@st.cache_data(show_spinner=False)
def compute_p_values(df: pd.DataFrame, ipss_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Compute:
    - WITHIN-GROUP (paired): Paired t-test, Wilcoxon (All, Placebo, Active)
    - BETWEEN-GROUP at each timepoint: Welch t, Mann–Whitney U
    - BETWEEN-GROUP on Δ by visit: Welch t, Mann–Whitney U
    - Van Elteren on Δ across visits: one pooled p-value accounting for visit strata
    """
    work = df.copy()
    work["GROUP_NORM"] = label_groups(work["GROUP"])  # Placebo vs Active

    # ---------- WITHIN-GROUP (paired) ----------
    within_ready = work.dropna(subset=ipss_cols)
    within_records = []
    for g in ("All", "Placebo", "Active"):
        dsub = within_ready if g == "All" else within_ready[within_ready["GROUP_NORM"] == g]
        if len(dsub) < 3:
            continue
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = ipss_cols[i], ipss_cols[j]
                try:
                    _, p_t = stats.ttest_rel(dsub[a], dsub[b], nan_policy='omit')
                except Exception:
                    p_t = np.nan
                try:
                    _, p_w = stats.wilcoxon(dsub[a], dsub[b], zero_method='wilcox', alternative='two-sided', mode='auto')
                except Exception:
                    p_w = np.nan
                within_records.append([g, a, b, "Paired t-test", p_t])
                within_records.append([g, a, b, "Wilcoxon signed-rank", p_w])
    within_df = pd.DataFrame(within_records, columns=["Group", "Time A", "Time B", "Test", "p-value"]) if within_records else pd.DataFrame(columns=["Group", "Time A", "Time B", "Test", "p-value"])            

    # ---------- BETWEEN-GROUP at each timepoint ----------
    between_rows = []
    for col in ipss_cols:
        placebo = work.loc[work["GROUP_NORM"] == "Placebo", col].dropna()
        active  = work.loc[work["GROUP_NORM"] == "Active",  col].dropna()
        if len(placebo) >= 2 and len(active) >= 2:
            try:
                _, p_t = stats.ttest_ind(placebo, active, equal_var=False, nan_policy='omit')
            except Exception:
                p_t = np.nan
            try:
                _, p_u = stats.mannwhitneyu(placebo, active, alternative='two-sided')
            except Exception:
                p_u = np.nan
            between_rows.append([col, "Welch t-test (indep)", p_t])
            between_rows.append([col, "Mann–Whitney U", p_u])
    between_df = pd.DataFrame(between_rows, columns=["Time", "Test", "p-value"]) if between_rows else pd.DataFrame(columns=["Time", "Test", "p-value"])           

    # ---------- Δ-by-visit BETWEEN-GROUP ----------
    base = ipss_cols[0]
    delta_rows = []
    active_by_visit: Dict[str, np.ndarray] = {}
    placebo_by_visit: Dict[str, np.ndarray] = {}
    for col in ipss_cols[1:]:  # post-baseline visits
        d = work[["GROUP_NORM", base, col]].dropna()
        d["DELTA"] = d[col] - d[base]
        pl = d.loc[d["GROUP_NORM"] == "Placebo", "DELTA"].to_numpy()
        ac = d.loc[d["GROUP_NORM"] == "Active",  "DELTA"].to_numpy()
        visit_label = col
        active_by_visit[visit_label] = ac
        placebo_by_visit[visit_label] = pl
        if pl.size >= 2 and ac.size >= 2:
            try:
                _, p_t = stats.ttest_ind(pl, ac, equal_var=False)
            except Exception:
                p_t = np.nan
            try:
                _, p_u = stats.mannwhitneyu(pl, ac, alternative='two-sided')
            except Exception:
                p_u = np.nan
            delta_rows.append([f"{base} → {col}", "Welch t-test (indep)", p_t])
            delta_rows.append([f"{base} → {col}", "Mann–Whitney U", p_u])
    between_delta_df = pd.DataFrame(delta_rows, columns=["Change (Δ)", "Test", "p-value"]) if delta_rows else pd.DataFrame(columns=["Change (Δ)", "Test", "p-value"])           

    # ---------- Van Elteren on Δ across visits ----------
    Z, p_ve, counts = van_elteren_delta(active_by_visit, placebo_by_visit)
    ve_df = pd.DataFrame({
        "Analysis": ["Van Elteren (Δ across visits; strata = visits)"],
        "Z": [Z],
        "p-value": [p_ve]
    })
    if counts:
        counts_df = pd.DataFrame(counts, columns=["Visit (stratum)", "N Placebo", "N Active"])
    else:
        counts_df = pd.DataFrame(columns=["Visit (stratum)", "N Placebo", "N Active"])

    return {
        "WITHIN": within_df,
        "BETWEEN": between_df,
        "BETWEEN_DELTA": between_delta_df,
        "VAN_ELTEREN_DELTA": ve_df,
        "VAN_ELTEREN_STRATA_COUNTS": counts_df,
    }


def to_excel_with_pvalues(original_df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        original_df.to_excel(writer, index=False, sheet_name="Input")
        row0 = 0
        pv_sheet = "P_VALUES"
        sections = [
            ("BETWEEN-GROUP (Placebo vs Active) — Final values", tables.get("BETWEEN")),
            ("BETWEEN-GROUP (Placebo vs Active) — Δ by visit", tables.get("BETWEEN_DELTA")),
            ("VAN ELTEREN — pooled Δ across visits (stratified Wilcoxon)", tables.get("VAN_ELTEREN_DELTA")),
            ("VAN ELTEREN — strata counts", tables.get("VAN_ELTEREN_STRATA_COUNTS")),
            ("WITHIN-GROUP (paired across time)", tables.get("WITHIN")),
        ]
        for title, table in sections:
            pd.DataFrame({title: []}).to_excel(writer, index=False, sheet_name=pv_sheet, startrow=row0)
            row0 += 1
            if table is not None and not table.empty:
                table.to_excel(writer, index=False, sheet_name=pv_sheet, startrow=row0)
                row0 += len(table) + 3
            else:
                pd.DataFrame({"info": ["No results"]}).to_excel(writer, index=False, sheet_name=pv_sheet, startrow=row0)
                row0 += 3
    return output.getvalue()


def draw_boxplots(df: pd.DataFrame, ipss_cols: List[str]):
    df = df.copy(); df["GROUP_NORM"] = label_groups(df["GROUP"])  
    for col in ipss_cols:
        groups = [
            df.loc[df["GROUP_NORM"] == "Placebo", col].dropna(),
            df.loc[df["GROUP_NORM"] == "Active",  col].dropna(),
        ]
        labels = ["Placebo", "Active"]
        fig, ax = plt.subplots()
        ax.boxplot(groups, labels=labels)
        ax.set_title(f"{col} — distribution by group")
        ax.set_ylabel("IPSS Score")
        st.pyplot(fig, use_container_width=True)


# -------------------------------
# UI
# -------------------------------
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])    
    st.markdown("**Tip:** The app auto-detects timepoints; ensure the four expected columns exist.")

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

with st.expander("Preview first 10 rows"):
    st.dataframe(df.head(10), use_container_width=True)

with st.spinner("Computing p-values (includes Van Elteren on Δ)..."):
    tables = compute_p_values(df, ipss_cols)

# Display
c1, c2 = st.columns(2)
with c1:
    st.subheader("Between-Group p-values — Final values (by timepoint)")
    st.dataframe(tables["BETWEEN"], use_container_width=True)
with c2:
    st.subheader("Between-Group p-values — Δ by visit")
    st.dataframe(tables["BETWEEN_DELTA"], use_container_width=True)

st.subheader("Van Elteren (Δ across visits; strata = visits)")
st.dataframe(tables["VAN_ELTEREN_DELTA"], use_container_width=True)
with st.expander("Stratum sizes used in Van Elteren"):
    st.dataframe(tables["VAN_ELTEREN_STRATA_COUNTS"], use_container_width=True)

st.subheader("Within-Group p-values (paired across time)")
st.dataframe(tables["WITHIN"], use_container_width=True)

# Plots
st.subheader("Distribution plots by time point")
draw_boxplots(df, ipss_cols)

# Download
excel_bytes = to_excel_with_pvalues(df, tables)
st.download_button(
    label="Download Excel with P_VALUES (incl. Van Elteren)",
    data=excel_bytes,
    file_name="ipss_pvalues_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("Done. Van Elteren added. You can now download results.")

