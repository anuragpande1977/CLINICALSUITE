"""
Streamlit app: Upload an IPSS Excel file (4 time points) and compute
parametric & non-parametric stats for within-/between-group analyses.
Adds *non‑p-value* metrics for skewed data (effect sizes + HL shift),
renders charts, and exports a clear P_VALUES sheet with headings.

Deploy notes (Streamlit Cloud)
- Python >=3.9
- requirements.txt
  streamlit
  pandas
  scipy>=1.9
  numpy
  openpyxl
  xlsxwriter
  matplotlib

Expected columns in Excel (case-sensitive):
- SUBJECT ID
- GROUP  (exact 'Placebo' treated as Placebo; everything else is Active)
- 4 timepoint columns for the *same* IPSS scale, e.g.
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

st.set_page_config(page_title="IPSS P-Values (4 TP) — v3", layout="wide")
st.title("IPSS (4 Time Points) — Stats & Charts (v3)")
st.caption("Upload your Excel, see within/ between-group parametric + non-parametric tests, robust effect sizes, and download a results workbook.")

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


def label_groups(series: pd.Series) -> pd.Series:
    return np.where(series.astype(str).str.strip().str.lower() == "placebo", "Placebo", "Active")


# ---------- Non-parametrics beyond p-values ----------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x); y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return None
    diffs = x[:, None] - y[None, :]
    num = (diffs > 0).sum() - (diffs < 0).sum()
    return float(num) / float(x.size * y.size)


def rank_biserial_from_u(u: float, n1: int, n2: int, sign_hint: float) -> Optional[float]:
    if n1 * n2 == 0:
        return None
    r = 1.0 - 2.0 * (u / (n1 * n2))
    # apply sign from median difference (so direction is interpretable)
    return float(np.sign(sign_hint)) * abs(float(r)) if sign_hint != 0 else float(r)


def common_language_effect(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    # CLES = P(X > Y); intuitive probability effect size
    x = np.asarray(x); y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return None
    diffs = x[:, None] - y[None, :]
    return float((diffs > 0).sum() / (x.size * y.size))


def hl_shift_and_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 7) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Hodges–Lehmann shift estimate (median of pairwise diffs) + bootstrap CI
    x = np.asarray(x); y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return None, None, None
    try:
        hl = float(stats.hodgeslehmann(x, y))
    except Exception:
        # fallback: median difference
        hl = float(np.median(x) - np.median(y))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        try:
            boots.append(float(stats.hodgeslehmann(xb, yb)))
        except Exception:
            boots.append(float(np.median(xb) - np.median(yb)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return hl, float(lo), float(hi)


@st.cache_data(show_spinner=False)
def compute_p_values(df: pd.DataFrame, ipss_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Compute within (paired) + between (timepoint & Δ) parametric and non-parametric tests,
    plus non-p-value effect sizes suitable for skewed data.
    """
    work = df.copy()
    work["GROUP_NORM"] = label_groups(work["GROUP"])  # Placebo vs Active

    # WITHIN (paired)
    within_ready = work.dropna(subset=ipss_cols)
    within_rows = []
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
                within_rows.append([g, a, b, "Paired t-test", p_t])
                within_rows.append([g, a, b, "Wilcoxon signed-rank", p_w])
    within_df = pd.DataFrame(within_rows, columns=["Group", "Time A", "Time B", "Test", "p-value"]) if within_rows else pd.DataFrame(columns=["Group", "Time A", "Time B", "Test", "p-value"])            

    # BETWEEN at each timepoint + effect sizes
    bt_rows = []
    eff_rows = []
    for col in ipss_cols:
        pl = work.loc[work["GROUP_NORM"] == "Placebo", col].dropna().to_numpy()
        ac = work.loc[work["GROUP_NORM"] == "Active",  col].dropna().to_numpy()
        n_p, n_a = pl.size, ac.size
        note = None
        if n_p >= 2 and n_a >= 2:
            try:
                _, p_t = stats.ttest_ind(pl, ac, equal_var=False, nan_policy='omit')
            except Exception:
                p_t = np.nan
            try:
                u_stat, p_u = stats.mannwhitneyu(pl, ac, alternative='two-sided')
            except Exception:
                u_stat, p_u = np.nan, np.nan
            try:
                _, p_bm = stats.brunnermunzel(pl, ac, alternative='two-sided')
            except Exception:
                p_bm = np.nan
            # effect sizes
            cd = cliffs_delta(ac, pl)  # positive if Active > Placebo
            sign_hint = float(np.median(ac) - np.median(pl))
            r_rb = rank_biserial_from_u(u_stat if not np.isnan(u_stat) else 0.5*n_p*n_a, n_a, n_p, sign_hint)
            cles = common_language_effect(ac, pl)  # P(Active > Placebo)
            hl, lo, hi = hl_shift_and_ci(ac, pl)
        else:
            p_t = p_u = p_bm = np.nan
            cd = r_rb = cles = hl = lo = hi = None
            note = "Insufficient N in one or both groups"
        bt_rows.append([col, n_p, n_a, "Welch t-test (indep)", p_t, note])
        bt_rows.append([col, n_p, n_a, "Mann–Whitney U", p_u, note])
        bt_rows.append([col, n_p, n_a, "Brunner–Munzel", p_bm, note])
        eff_rows.append([col, n_p, n_a, cd, r_rb, cles, hl, lo, hi])
    between_time_df = pd.DataFrame(bt_rows, columns=["Time", "N Placebo", "N Active", "Test", "p-value", "Note"])
    effect_df = pd.DataFrame(eff_rows, columns=["Time", "N Placebo", "N Active", "Cliff's δ (Active−Placebo)", "Rank-biserial r", "CLES P(Active>Placebo)", "HL shift (A−P)", "HL 95% CI lo", "HL 95% CI hi"])    

    # BETWEEN on Δ from Baseline
    base = ipss_cols[0]
    bt_delta_rows = []
    eff_delta_rows = []
    for col in ipss_cols[1:]:
        d = work[["GROUP_NORM", base, col]].dropna()
        d["DELTA"] = d[col] - d[base]
        pl = d.loc[d["GROUP_NORM"] == "Placebo", "DELTA"].to_numpy()
        ac = d.loc[d["GROUP_NORM"] == "Active", "DELTA"].to_numpy()
        n_p, n_a = pl.size, ac.size
        note = None
        if n_p >= 2 and n_a >= 2:
            try:
                _, p_t = stats.ttest_ind(pl, ac, equal_var=False)
            except Exception:
                p_t = np.nan
            try:
                u_stat, p_u = stats.mannwhitneyu(pl, ac, alternative='two-sided')
            except Exception:
                u_stat, p_u = np.nan, np.nan
            try:
                _, p_bm = stats.brunnermunzel(pl, ac, alternative='two-sided')
            except Exception:
                p_bm = np.nan
            cd = cliffs_delta(ac, pl)
            sign_hint = float(np.median(ac) - np.median(pl))
            r_rb = rank_biserial_from_u(u_stat if not np.isnan(u_stat) else 0.5*n_p*n_a, n_a, n_p, sign_hint)
            cles = common_language_effect(ac, pl)
            hl, lo, hi = hl_shift_and_ci(ac, pl)
        else:
            p_t = p_u = p_bm = np.nan
            cd = r_rb = cles = hl = lo = hi = None
            note = "Insufficient N in one or both groups"
        bt_delta_rows.append([f"{base} → {col}", n_p, n_a, "Welch t-test (indep)", p_t, note])
        bt_delta_rows.append([f"{base} → {col}", n_p, n_a, "Mann–Whitney U", p_u, note])
        bt_delta_rows.append([f"{base} → {col}", n_p, n_a, "Brunner–Munzel", p_bm, note])
        eff_delta_rows.append([f"{base} → {col}", n_p, n_a, cd, r_rb, cles, hl, lo, hi])
    between_delta_df = pd.DataFrame(bt_delta_rows, columns=["Change (Δ)", "N Placebo", "N Active", "Test", "p-value", "Note"])
    effect_delta_df = pd.DataFrame(eff_delta_rows, columns=["Change (Δ)", "N Placebo", "N Active", "Cliff's δ (Active−Placebo)", "Rank-biserial r", "CLES P(Active>Placebo)", "HL shift (A−P)", "HL 95% CI lo", "HL 95% CI hi"])    

    return {
        "WITHIN": within_df,
        "BETWEEN_TIME": between_time_df,
        "BETWEEN_DELTA": between_delta_df,
        "EFFECTS_TIME": effect_df,
        "EFFECTS_DELTA": effect_delta_df,
    }


def to_excel_with_pvalues(original_df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        original_df.to_excel(writer, index=False, sheet_name="Input")
        start = 0
        sheet = "P_VALUES"
        # Explicit section headers for clarity
        sections = [
            ("BETWEEN-GROUP (Placebo vs Active) — Timepoints", tables.get("BETWEEN_TIME")),
            ("BETWEEN-GROUP (Placebo vs Active) — Δ from Baseline", tables.get("BETWEEN_DELTA")),
            ("WITHIN-GROUP (paired across time) — All / Placebo / Active", tables.get("WITHIN")),
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
        # Effect sizes to separate sheets for readability
        eff_time = tables.get("EFFECTS_TIME")
        if eff_time is not None and not eff_time.empty:
            eff_time.to_excel(writer, index=False, sheet_name="EFFECT_SIZES_TIME")
        eff_delta = tables.get("EFFECTS_DELTA")
        if eff_delta is not None and not eff_delta.empty:
            eff_delta.to_excel(writer, index=False, sheet_name="EFFECT_SIZES_DELTA")
    return output.getvalue()


# -------------------------------
# Charts
# -------------------------------

def draw_boxplots(df: pd.DataFrame, ipss_cols: List[str]):
    df = df.copy(); df["GROUP_NORM"] = label_groups(df["GROUP"])  
    for col in ipss_cols:
        pl = df.loc[df["GROUP_NORM"] == "Placebo", col].dropna()
        ac = df.loc[df["GROUP_NORM"] == "Active",  col].dropna()
        fig, ax = plt.subplots()
        ax.boxplot([pl.values, ac.values], labels=["Placebo", "Active"])
        ax.set_title(f"{col} — Box plot by group")
        ax.set_ylabel("IPSS Score")
        st.pyplot(fig, use_container_width=True)


def draw_violins(df: pd.DataFrame, ipss_cols: List[str]):
    df = df.copy(); df["GROUP_NORM"] = label_groups(df["GROUP"])  
    for col in ipss_cols:
        pl = df.loc[df["GROUP_NORM"] == "Placebo", col].dropna()
        ac = df.loc[df["GROUP_NORM"] == "Active",  col].dropna()
        fig, ax = plt.subplots()
        parts = ax.violinplot([pl.values, ac.values], showmeans=True, showmedians=True)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Placebo", "Active"])
        ax.set_title(f"{col} — Violin plot (distribution)")
        ax.set_ylabel("IPSS Score")
        st.pyplot(fig, use_container_width=True)


def draw_trend_summary(df: pd.DataFrame, ipss_cols: List[str]):
    df = df.copy(); df["GROUP_NORM"] = label_groups(df["GROUP"])  
    # compute medians and IQR per group per time
    times = ipss_cols
    for g in ("Placebo", "Active"):
        med = []; q1 = []; q3 = []
        for col in times:
            vals = df.loc[df["GROUP_NORM"] == g, col].dropna().values
            if vals.size:
                med.append(np.median(vals)); q1.append(np.percentile(vals, 25)); q3.append(np.percentile(vals, 75))
            else:
                med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
        # line with IQR band
        fig, ax = plt.subplots()
        x = np.arange(len(times))
        ax.plot(x, med, marker='o')
        ax.fill_between(x, q1, q3, alpha=0.2)
        ax.set_xticks(x); ax.set_xticklabels(times, rotation=20)
        ax.set_title(f"{g}: Median trend with IQR")
        ax.set_ylabel("IPSS Score")
        st.pyplot(fig, use_container_width=True)


# -------------------------------
# UI
# -------------------------------
with st.sidebar:
    st.header("1) Upload Excel")
    uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])    
    st.markdown("**Tip:** Stick to the template column names; the app auto-detects the four timepoints.")

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
with st.expander("Preview (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# Compute
with st.spinner("Computing tests & effect sizes..."):
    tables = compute_p_values(df, ipss_cols)

# Tables
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Between-Group (Placebo vs Active) — Timepoints")
    st.caption("Welch t, Mann–Whitney U, Brunner–Munzel + sample sizes & notes.")
    st.dataframe(tables["BETWEEN_TIME"], use_container_width=True)
with col2:
    st.subheader("Between-Group (Placebo vs Active) — Δ from Baseline")
    st.caption("Tests on change scores vs Baseline.")
    st.dataframe(tables["BETWEEN_DELTA"], use_container_width=True)

st.subheader("Effect Sizes & Robust Shifts (non‑parametric friendly)")
st.caption("Cliff's δ (directional), rank-biserial r, common language effect size, and Hodges–Lehmann shift with 95% CI.")
st.dataframe(tables["EFFECTS_TIME"], use_container_width=True)
st.dataframe(tables["EFFECTS_DELTA"], use_container_width=True)

st.subheader("Within-Group (paired across time)")
st.caption("All / Placebo / Active — Paired t-test & Wilcoxon across timepoint pairs.")
st.dataframe(tables["WITHIN"], use_container_width=True)

# Charts
st.subheader("Charts ▸ Distribution (Box & Violin)")
draw_boxplots(df, ipss_cols)
draw_violins(df, ipss_cols)

st.subheader("Charts ▸ Trends (Median + IQR by group)")
draw_trend_summary(df, ipss_cols)

# Download
excel_bytes = to_excel_with_pvalues(df, tables)
st.download_button(
    label="Download results workbook (with P_VALUES & EFFECT_SIZES)",
    data=excel_bytes,
    file_name="ipss_stats_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("Done. Results and charts are ready. Try another file if needed.")
