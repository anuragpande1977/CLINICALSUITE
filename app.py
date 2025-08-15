"""
Streamlit app: Upload an IPSS Excel file (4 time points) and compute all
parametric & non-parametric p-values for within-group and between-group analyses.
Also renders box plots and lets you download an Excel with a P_VALUES sheet.

Deploy notes (works on Streamlit Cloud):
- Python >=3.9
- requirements.txt should include (pin versions as you like):
  streamlit
  pandas
  scipy>=1.9
  openpyxl
  xlsxwriter
  matplotlib

Expected columns in your Excel (case-sensitive by default):
- 'SUBJECT ID'
- 'GROUP' (values like 'Placebo' vs anything else considered Active)
- Four IPSS columns for the same scale across time, e.g.:
  'Baseline IPSS  Total Score', 'Day 28 IPSS  Total Score',
  'Day 56 IPSS  Total Score', 'Day 84 IPSS  Total Score'

The app tries to auto-detect these by regex and sort them in time order.
"""

import io
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="IPSS P-Values (4 Time Points)", layout="wide")
st.title("IPSS (4 Time Points): P-Values Explorer")
st.caption(
    "Upload your Excel. The app computes within- and between-group p-values using parametric (t-tests) and non-parametric (Wilcoxon, Mann–Whitney, Brunner–Munzel) tests, and provides a downloadable Excel with results."
)

# -------------------------------
# Utilities
# -------------------------------
TIME_ORDER_MAP = {
    "baseline": 0,
    "day 28": 28,
    "day 56": 56,
    "day 84": 84,
}

IPSS_COL_REGEX = re.compile(
    r"^(baseline|day\s*28|day\s*56|day\s*84).*ipss.*total\s*score",
    re.IGNORECASE,
)

REQUIRED_ID_COLS = ["SUBJECT ID", "GROUP"]


def find_ipss_columns(columns: List[str]) -> List[str]:
    cand = []
    for c in columns:
        if IPSS_COL_REGEX.search(str(c)):
            cand.append(c)
    # sort by time order using TIME_ORDER_MAP
    def time_key(col: str) -> int:
        s = col.lower()
        for k, v in TIME_ORDER_MAP.items():
            if k in s:
                return v
        return 10_000

    cand_sorted = sorted(cand, key=time_key)
    return cand_sorted


def validate_template(df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
    missing_ids = [c for c in REQUIRED_ID_COLS if c not in df.columns]
    if missing_ids:
        return False, f"Missing required columns: {missing_ids}", []
    ipss_cols = find_ipss_columns(df.columns.tolist())
    if len(ipss_cols) != 4:
        return (
            False,
            "Could not detect exactly 4 IPSS timepoint columns. Ensure columns contain 'Baseline', 'Day 28', 'Day 56', 'Day 84' and 'IPSS  Total Score'.",
            ipss_cols,
        )
    return True, "", ipss_cols


def label_groups(series: pd.Series) -> pd.Series:
    # Treat literal 'Placebo' as Placebo, everything else as Active
    return np.where(series.astype(str).str.strip().str.lower() == "placebo", "Placebo", "Active")


def group_counts(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        for g in ("Placebo", "Active"):
            n = df.loc[df["GROUP_NORM"] == g, col].dropna().shape[0]
            rows.append([col, g, n])
    return pd.DataFrame(rows, columns=["Time", "Group", "N"])


@st.cache_data(show_spinner=False)
def compute_p_values(df: pd.DataFrame, ipss_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Compute:
    - WITHIN-GROUP (paired): Paired t-test, Wilcoxon for All, Placebo, Active
    - BETWEEN-GROUP (timepoint): Welch t-test, Mann–Whitney U, Brunner–Munzel
    - BETWEEN-GROUP (delta from baseline): Welch t-test, Mann–Whitney U, Brunner–Munzel
    Returns dict of tables.
    """
    work = df.copy()
    work["GROUP_NORM"] = label_groups(work["GROUP"])  # Placebo vs Active

    # ---------- WITHIN-GROUP (paired) ----------
    within_ready = work.dropna(subset=ipss_cols)
    within_records = []
    groups_for_within = ["All", "Placebo", "Active"]
    for g in groups_for_within:
        dsub = within_ready if g == "All" else within_ready[within_ready["GROUP_NORM"] == g]
        if len(dsub) < 3:
            continue
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = ipss_cols[i], ipss_cols[j]
                # Paired t-test
                try:
                    _, p_t = stats.ttest_rel(dsub[a], dsub[b], nan_policy="omit")
                except Exception:
                    p_t = np.nan
                # Wilcoxon signed-rank
                try:
                    _, p_w = stats.wilcoxon(dsub[a], dsub[b], zero_method="wilcox", alternative="two-sided", mode="auto")
                except Exception:
                    p_w = np.nan
                within_records.append([g, a, b, "Paired t-test", p_t])
                within_records.append([g, a, b, "Wilcoxon signed-rank", p_w])
    within_df = (
        pd.DataFrame(within_records, columns=["Group", "Time A", "Time B", "Test", "p-value"])
        if within_records
        else pd.DataFrame(columns=["Group", "Time A", "Time B", "Test", "p-value"])
    )

    # ---------- BETWEEN-GROUP (timepoint) ----------
    bt_time_records = []
    for col in ipss_cols:
        placebo = work.loc[work["GROUP_NORM"] == "Placebo", col].dropna()
        active = work.loc[work["GROUP_NORM"] == "Active", col].dropna()
        n_p, n_a = len(placebo), len(active)
        note = None
        if n_p >= 2 and n_a >= 2:
            # Welch t-test
            try:
                _, p_t = stats.ttest_ind(placebo, active, equal_var=False, nan_policy="omit")
            except Exception:
                p_t = np.nan
            # Mann–Whitney U
            try:
                _, p_u = stats.mannwhitneyu(placebo, active, alternative="two-sided")
            except Exception:
                p_u = np.nan
            # Brunner–Munzel (more robust to unequal variances/ties)
            try:
                _, p_bm = stats.brunnermunzel(placebo, active, alternative="two-sided")
            except Exception:
                p_bm = np.nan
        else:
            p_t = p_u = p_bm = np.nan
            note = "Insufficient N in one or both groups"
        bt_time_records.append([col, n_p, n_a, "Welch t-test (indep)", p_t, note])
        bt_time_records.append([col, n_p, n_a, "Mann–Whitney U", p_u, note])
        bt_time_records.append([col, n_p, n_a, "Brunner–Munzel", p_bm, note])
    between_time_df = pd.DataFrame(
        bt_time_records, columns=["Time", "N Placebo", "N Active", "Test", "p-value", "Note"]
    )

    # ---------- BETWEEN-GROUP (delta from baseline) ----------
    bt_delta_records = []
    base = ipss_cols[0]
    for col in ipss_cols[1:]:
        d = work[["GROUP_NORM", base, col]].dropna()
        d["DELTA"] = d[col] - d[base]
        placebo = d.loc[d["GROUP_NORM"] == "Placebo", "DELTA"].dropna()
        active = d.loc[d["GROUP_NORM"] == "Active", "DELTA"].dropna()
        n_p, n_a = len(placebo), len(active)
        note = None
        if n_p >= 2 and n_a >= 2:
            try:
                _, p_t = stats.ttest_ind(placebo, active, equal_var=False)
            except Exception:
                p_t = np.nan
            try:
                _, p_u = stats.mannwhitneyu(placebo, active, alternative="two-sided")
            except Exception:
                p_u = np.nan
            try:
                _, p_bm = stats.brunnermunzel(placebo, active, alternative="two-sided")
            except Exception:
                p_bm = np.nan
        else:
            p_t = p_u = p_bm = np.nan
            note = "Insufficient N in one or both groups"
        bt_delta_records.append([f"{base} → {col}", n_p, n_a, "Welch t-test (indep)", p_t, note])
        bt_delta_records.append([f"{base} → {col}", n_p, n_a, "Mann–Whitney U", p_u, note])
        bt_delta_records.append([f"{base} → {col}", n_p, n_a, "Brunner–Munzel", p_bm, note])
    between_delta_df = pd.DataFrame(
        bt_delta_records, columns=["Change (Δ)", "N Placebo", "N Active", "Test", "p-value", "Note"]
    )

    return {
        "WITHIN": within_df,
        "BETWEEN_TIME": between_time_df,
        "BETWEEN_DELTA": between_delta_df,
        "COUNTS": group_counts(work.assign(GROUP_NORM=work["GROUP_NORM"]), ipss_cols),
    }


def to_excel_with_pvalues(original_df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> bytes:
    """Create an Excel file in memory with original data plus a P_VALUES sheet."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        original_df.to_excel(writer, index=False, sheet_name="Input")
        # One sheet summarizing all p-values, with section headers
        start = 0
        sheet = "P_VALUES"
        for name in ("WITHIN", "BETWEEN_TIME", "BETWEEN_DELTA"):
            pd.DataFrame({name: []}).to_excel(writer, index=False, sheet_name=sheet, startrow=start)
            start += 1
            table = tables.get(name)
            if table is not None and not table.empty:
                table.to_excel(writer, index=False, sheet_name=sheet, startrow=start)
                start += len(table) + 3
            else:
                pd.DataFrame({"info": ["No results"]}).to_excel(writer, index=False, sheet_name=sheet, startrow=start)
                start += 3
        # counts sheet for transparency
        if "COUNTS" in tables and not tables["COUNTS"].empty:
            tables["COUNTS"].to_excel(writer, index=False, sheet_name="GROUP_COUNTS")
    return output.getvalue()


def draw_boxplots(df: pd.DataFrame, ipss_cols: List[str]):
    """Draw separate box plots by group for each time point."""
    df = df.copy()
    df["GROUP_NORM"] = label_groups(df["GROUP"])  # Placebo vs Active

    for col in ipss_cols:
        # Prepare data
        groups = [
            df.loc[df["GROUP_NORM"] == "Placebo", col].dropna(),
            df.loc[df["GROUP_NORM"] == "Active", col].dropna(),
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
    st.markdown(
        "**Tip:** Keep consistent column names; the app auto-detects timepoint columns. Non-parametric between-group tests include Mann–Whitney and Brunner–Munzel."
    )

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

# Preview
with st.expander("Preview first 10 rows"):
    st.dataframe(df.head(10), use_container_width=True)

# Compute
with st.spinner("Computing p-values..."):
    tables = compute_p_values(df, ipss_cols)

left, right = st.columns([1, 1])
with left:
    st.subheader("Between-Group p-values (Timepoints)")
    st.caption("Welch t-test, Mann–Whitney U, Brunner–Munzel at each timepoint.")
    st.dataframe(tables["BETWEEN_TIME"], use_container_width=True)
with right:
    st.subheader("Between-Group p-values (Δ from Baseline)")
    st.caption("Welch t-test, Mann–Whitney U, Brunner–Munzel on change scores vs Baseline.")
    st.dataframe(tables["BETWEEN_DELTA"], use_container_width=True)

st.subheader("Within-Group p-values (paired across time)")
st.caption("Shown for All, Placebo, and Active groups (paired t-test and Wilcoxon).")
st.dataframe(tables["WITHIN"], use_container_width=True)

# Diagnostics
with st.expander("Group counts by timepoint"):
    st.dataframe(tables["COUNTS"], use_container_width=True)

# Plots
st.subheader("Distribution plots by time point")
draw_boxplots(df, ipss_cols)

# Download
excel_bytes = to_excel_with_pvalues(df, tables)
st.download_button(
    label="Download Excel with P_VALUES sheet",
    data=excel_bytes,
    file_name="ipss_pvalues_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("Done. You can now download your results or try another file.")
