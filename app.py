"""
IPSS (4 Time Points) — Between-Group Longitudinal Suite

Upload an Excel with:
  - 'SUBJECT ID'
  - 'GROUP' (maps robustly: Placebo vs USPlus; aliases like "US Plus"/"Active" -> USPlus)
  - Four columns of the SAME IPSS scale across time (Baseline, Day 28, Day 56, Day 84),
    e.g. "Baseline IPSS  Total Score", "Day 28 IPSS  Total Score", ...

This app computes BETWEEN-GROUP ONLY:
  • ANCOVA at final visit (Final ~ Group + Baseline, HC3 SEs)
  • MMRM (Mixed Linear Model; Group×Time + Baseline; random intercept + slope)
  • GEE (Gaussian, AR(1) working corr; Group×Time + Baseline)
  • Mixed / RM ANOVA (Pingouin; with AnovaRM fallback for within-only)
  • MANOVA across Day 28/56/84
  • Δ-per-visit tests (Welch t, Mann–Whitney, Brunner–Munzel)
  • Van Elteren pooled Δ across visits (stratified Wilcoxon)
  • Optional rank-based ANOVA-type (ATS) via R nparLD (if rpy2 + R + nparLD available)

Exports an Excel workbook with a RESULTS sheet (all tables) and a CHARTS sheet
(box plots + median±IQR trends embedded as PNGs).
"""

import io
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

# Optional: mixed/RM ANOVA convenience
try:
    import pingouin as pg
except Exception:
    pg = None

st.set_page_config(page_title="IPSS — Between-Group Longitudinal Suite", layout="wide")
st.title("IPSS (4 Time Points) — Placebo vs USPlus (Between-Group Only)")
st.caption(
    "ANCOVA (final), MMRM, GEE, Mixed/RM ANOVA, MANOVA, Δ tests, Van Elteren, and optional rank-based ATS (nparLD). "
    "Charts shown and embedded in Excel."
)

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
        return False, "Need exactly 4 IPSS timepoint columns (Baseline/Day28/Day56/Day84).", ipss_cols
    return True, "", ipss_cols

def normalize_groups(series: pd.Series) -> pd.Series:
    """
    Map GROUP labels to 'Placebo' or 'USPlus'.
    Tolerant to spaces/case/punctuation; legacy 'Active' -> 'USPlus'.
    """
    s = series.astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    mapping = {
        "placebo": "Placebo",
        "control": "Placebo",
        "usplus": "USPlus",
        "us": "USPlus",
        "active": "USPlus",
    }
    out = s.map(mapping)
    out.name = "GROUP_NORM"
    return out

@st.cache_data(show_spinner=False)
def make_long(df: pd.DataFrame, ipss_cols: List[str]) -> pd.DataFrame:
    long = df.copy()
    long["GROUP_NORM"] = normalize_groups(long["GROUP"])
    long = long.loc[long["GROUP_NORM"].isin(["Placebo", "USPlus"])].copy()
    long = long.dropna(subset=[ipss_cols[0]])  # need Baseline for covariate
    long = long[["SUBJECT ID", "GROUP_NORM"] + ipss_cols].copy()
    long = long.rename(columns={ipss_cols[0]: "Baseline"})
    # melt to long
    idcol = "SUBJECT ID"
    long_m = long.melt(id_vars=[idcol, "GROUP_NORM", "Baseline"], value_vars=ipss_cols,
                       var_name="Time", value_name="Score")
    long_m["Time"] = pd.Categorical(long_m["Time"], categories=ipss_cols, ordered=True)
    # numeric time for models
    def _tn(s: str) -> int:
        s = s.lower()
        if "28" in s: return 28
        if "56" in s: return 56
        if "84" in s: return 84
        return 0
    long_m["TimeNum"] = long_m["Time"].astype(str).map(_tn)
    return long_m

def _table_or_empty(cols: List[str], rows: List[List[object]]):
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# -------------------------------
# Analyses
# -------------------------------
def ancova_final(df_long: pd.DataFrame, final_col: str) -> pd.DataFrame:
    wide = (df_long.drop_duplicates(["SUBJECT ID"])  # Baseline per subject
            .merge(
                df_long[df_long["Time"] == final_col][["SUBJECT ID", "Score"]]
                .rename(columns={"Score": "Final"}),
                on="SUBJECT ID", how="left"
            ))
    wide = wide.dropna(subset=["Final"])
    if wide.empty:
        return _table_or_empty(
            ["Final timepoint", "N Placebo", "N USPlus", "Adj diff (USPlus−Placebo)", "SE (HC3)", "95% CI lo", "95% CI hi", "p-value"], []
        )
    model = smf.ols("Final ~ Baseline + C(GROUP_NORM)", data=wide).fit(cov_type="HC3")
    term = "C(GROUP_NORM)[T.USPlus]"
    est = float(model.params.get(term, np.nan))
    se = float(model.bse.get(term, np.nan))
    pval = float(model.pvalues.get(term, np.nan))
    df_resid = float(model.df_resid)
    t_crit = stats.t.ppf(0.975, df_resid) if df_resid > 0 else np.nan
    ci_lo = est - t_crit * se if np.isfinite(t_crit) else np.nan
    ci_hi = est + t_crit * se if np.isfinite(t_crit) else np.nan
    n_p = int((wide["GROUP_NORM"] == "Placebo").sum())
    n_u = int((wide["GROUP_NORM"] == "USPlus").sum())
    return _table_or_empty(
        ["Final timepoint", "N Placebo", "N USPlus", "Adj diff (USPlus−Placebo)", "SE (HC3)", "95% CI lo", "95% CI hi", "p-value"],
        [[final_col, n_p, n_u, est, se, ci_lo, ci_hi, pval]]
    )

def mmrm(df_long: pd.DataFrame) -> pd.DataFrame:
    """Mixed Linear Model: Score ~ C(Time)*C(Group) + Baseline, random intercept + slope(TimeNum)."""
    use = df_long[df_long["Time"] != df_long["Time"].cat.categories[0]].copy()  # exclude Baseline from response
    if use.empty:
        return _table_or_empty(["Effect", "Param", "Estimate", "SE", "z", "p-value"], [])
    use["TimeC"] = use["Time"].astype(str)
    try:
        md = smf.mixedlm("Score ~ C(TimeC) * C(GROUP_NORM) + Baseline",
                         data=use, groups=use["SUBJECT ID"], re_formula="~TimeNum")
        mdf = md.fit(method="lbfgs", reml=True)
        rows = []
        for p, est in mdf.params.items():
            if p == "Intercept":
                effect = "Intercept"
            elif "C(GROUP_NORM)" in p:
                effect = "Group"
            elif "C(TimeC)" in p and ":" not in p:
                effect = "Time"
            elif ":" in p:
                effect = "Group×Time"
            else:
                effect = "Covariate"
            se = mdf.bse.get(p, np.nan)
            z = est / se if (se not in (0, np.nan)) else np.nan
            pval = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            rows.append([effect, p, est, se, z, pval])
        return _table_or_empty(["Effect", "Param", "Estimate", "SE", "z", "p-value"], rows)
    except Exception as e:
        return _table_or_empty(["Effect", "Param", "Estimate", "SE", "z", "p-value"],
                               [["ERROR", str(e), np.nan, np.nan, np.nan, np.nan]])

def gee(df_long: pd.DataFrame) -> pd.DataFrame:
    """Gaussian GEE with AR(1) working correlation; Score ~ C(Time)*C(Group) + Baseline."""
    use = df_long[df_long["Time"] != df_long["Time"].cat.categories[0]].copy()
    if use.empty:
        return _table_or_empty(["Param", "Estimate", "SE", "z", "p-value"], [])
    use["TimeC"] = use["Time"].astype(str)
    try:
        ind = sm.cov_struct.Autoregressive()
        fam = sm.families.Gaussian()
        model = smf.gee("Score ~ C(TimeC) * C(GROUP_NORM) + Baseline",
                        groups="SUBJECT ID", cov_struct=ind, family=fam, data=use)
        res = model.fit()
        rows = []
        for p, est in res.params.items():
            se = res.bse[p]
            z = est / se if (se not in (0, np.nan)) else np.nan
            pval = res.pvalues[p]
            rows.append([p, est, se, z, pval])
        return _table_or_empty(["Param", "Estimate", "SE", "z", "p-value"], rows)
    except Exception as e:
        return _table_or_empty(["Param", "Estimate", "SE", "z", "p-value"], [[str(e), np.nan, np.nan, np.nan, np.nan]])

def mixed_or_rm_anova(df_long: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pingouin mixed ANOVA (between=Group, within=Time) + RM ANOVA; fallback to AnovaRM within-only."""
    out = {}
    use = df_long[df_long["Time"] != df_long["Time"].cat.categories[0]].copy()
    if pg is not None:
        try:
            mx = pg.mixed_anova(data=use, dv="Score", within="Time", between="GROUP_NORM", subject="SUBJECT ID")
            out["MIXED_ANOVA_PINGOUIN"] = mx
        except Exception as e:
            out["MIXED_ANOVA_PINGOUIN"] = _table_or_empty(["Source", "SS", "p-unc"], [[str(e), np.nan, np.nan]])
        try:
            rm = pg.rm_anova(data=use, dv="Score", within="Time", subject="SUBJECT ID", detailed=True)
            out["RM_ANOVA_PINGOUIN"] = rm
        except Exception:
            pass
    # statsmodels fallback (within factor only)
    try:
        from statsmodels.stats.anova import AnovaRM
        aov = AnovaRM(use, depvar="Score", subject="SUBJECT ID", within=["Time"]).fit()
        out["RM_ANOVA_SM_WITHIN_ONLY"] = aov.anova_table  # pandas DataFrame
    except Exception:
        pass
    return out

def manova(df_long: pd.DataFrame, ipss_cols: List[str]) -> pd.DataFrame:
    """MANOVA across Day 28/56/84 by Group."""
    wide = df_long.pivot_table(index=["SUBJECT ID", "GROUP_NORM", "Baseline"],
                               columns="Time", values="Score", aggfunc="first").reset_index()
    visits = [c for c in ipss_cols[1:]]
    have = [c for c in visits if c in wide.columns]
    sub = wide.dropna(subset=have)
    if sub.empty or len(have) < 2:
        return _table_or_empty(["Test", "Statistic", "df1", "df2", "p-value"], [])
    try:
        formula = " + ".join([f\"Q('{c}')\" for c in have]) + " ~ C(GROUP_NORM)"
        mv = sm.multivariate.MANOVA.from_formula(formula, data=sub)
        w = mv.mv_test()
        test_tbl = w.results['C(GROUP_NORM)']['stat']
        row = test_tbl.loc['Wilks' if 'Wilks' in test_tbl.index else test_tbl.index[0]]
        stat = float(row['Value']); df1 = float(row['Num DF']); df2 = float(row['Den DF']); pval = float(row['Pr > F'])
        return _table_or_empty(["Test", "Statistic", "df1", "df2", "p-value"],
                               [["Wilks' Lambda (Group)", stat, df1, df2, pval]])
    except Exception as e:
        return _table_or_empty(["Test", "Statistic", "df1", "df2", "p-value"], [[f"ERROR: {e}", np.nan, np.nan, np.nan, np.nan]])

def delta_tests(df_long: pd.DataFrame, ipss_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-visit Δ(visit−Baseline) between groups + Van Elteren pooled Δ."""
    base = ipss_cols[0]
    rows_per_visit = []
    usplus_by_visit = {}
    placebo_by_visit = {}

    for col in ipss_cols[1:]:
        d = (df_long[["SUBJECT ID", "GROUP_NORM", "Baseline"]]
             .drop_duplicates("SUBJECT ID")
             .merge(df_long[df_long["Time"] == col][["SUBJECT ID", "Score"]]
                    .rename(columns={"Score": col}), on="SUBJECT ID", how="inner"))
        d["DELTA"] = d[col] - d["Baseline"]
        pl = d.loc[d["GROUP_NORM"] == "Placebo", "DELTA"].dropna().to_numpy()
        us = d.loc[d["GROUP_NORM"] == "USPlus",  "DELTA"].dropna().to_numpy()
        usplus_by_visit[col] = us
        placebo_by_visit[col] = pl

        n_p, n_u = pl.size, us.size
        note = None
        if n_p >= 2 and n_u >= 2:
            try: _, p_t = stats.ttest_ind(pl, us, equal_var=False)
            except Exception: p_t = np.nan
            try: _, p_u = stats.mannwhitneyu(pl, us, alternative='two-sided')
            except Exception: p_u = np.nan
            try: _, p_bm = stats.brunnermunzel(pl, us, alternative='two-sided')
            except Exception: p_bm = np.nan
        else:
            p_t = p_u = p_bm = np.nan; note = "Insufficient N"

        rows_per_visit.extend([
            [f"{base} → {col}", n_p, n_u, "Welch t-test (indep)", p_t, note],
            [f"{base} → {col}", n_p, n_u, "Mann–Whitney U", p_u, note],
            [f"{base} → {col}", n_p, n_u, "Brunner–Munzel", p_bm, note]
        ])

    per_visit = _table_or_empty(["Change (Δ)", "N Placebo", "N USPlus", "Test", "p-value", "Note"], rows_per_visit)

    # Van Elteren (stratified Wilcoxon) on Δ across visits
    Z, p_ve, counts = van_elteren(usplus_by_visit, placebo_by_visit, ipss_cols)
    ve = _table_or_empty(["Analysis", "Z", "p-value"], [["Van Elteren (Δ across visits; strata = visits)", Z, p_ve]])
    counts_df = _table_or_empty(["Visit (stratum)", "N Placebo", "N USPlus"], counts)
    return per_visit, ve, counts_df

def van_elteren(usplus_by_visit: Dict[str, np.ndarray], placebo_by_visit: Dict[str, np.ndarray], ipss_cols: List[str]):
    """Compute Z and p for stratified Wilcoxon (Van Elteren) over timepoint strata."""
    def _wilcox_stats(ax, px):
        n1, n2 = len(ax), len(px); N = n1 + n2
        pooled = np.concatenate([ax, px])
        ranks = stats.rankdata(pooled, method="average")
        W = ranks[:n1].sum(); E = n1 * (N + 1)/2.0
        vals, counts = np.unique(pooled, return_counts=True)
        tie_term = ((counts ** 3 - counts).sum()) / (N * (N - 1)) if N > 1 else 0.0
        Var = (n1 * n2 / 12.0) * ((N + 1) - tie_term)
        return W, E, Var

    W_sum = E_sum = Var_sum = 0.0
    counts = []
    for v in ipss_cols[1:]:
        ax = np.asarray(usplus_by_visit.get(v, np.array([])))
        px = np.asarray(placebo_by_visit.get(v, np.array([])))
        if len(ax) < 1 or len(px) < 1:
            continue
        W,E,Var = _wilcox_stats(ax, px)
        W_sum += W; E_sum += E; Var_sum += Var
        counts.append((v, int(len(px)), int(len(ax))))
    if Var_sum <= 0 or not counts:
        return None, None, counts
    Z = (W_sum - E_sum) / np.sqrt(Var_sum)
    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    return float(Z), float(p), counts

# Optional: rank-based ANOVA-type (ATS) via R nparLD
def _has_r_nparld() -> bool:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        ro.r("suppressMessages(require(nparLD))")
        ok = bool(ro.r("'nparLD' %in% rownames(installed.packages())")[0])
        return ok
    except Exception:
        return False

def rank_ats_nparld(df_long: pd.DataFrame) -> pd.DataFrame:
    if not _has_r_nparld():
        return _table_or_empty(["Effect", "ATS", "df", "p-value"], [["nparLD not available", np.nan, np.nan, np.nan]])
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        use = df_long[df_long["Time"] != df_long["Time"].cat.categories[0]].copy()
        use = use.rename(columns={"SUBJECT ID": "ID", "GROUP_NORM": "Group"})
        r_df = pandas2ri.py2rpy(use[["ID", "Group", "Time", "Score"]])
        ro.globalenv["d"] = r_df
        ro.r("d$ID <- as.factor(d$ID); d$Group <- as.factor(d$Group); d$Time <- as.factor(d$Time)")
        ro.r("library(nparLD)")
        ro.r("res <- f1.ld.f1(y=Score, time1=Time, group=Group, subject=ID, data=d, description=FALSE)")
        tab = ro.r("as.data.frame(res$ANOVA.test)")
        pdf = pandas2ri.rpy2py(tab).reset_index().rename(columns={"index":"Effect"})
        keep = pdf[pdf['Effect'].str.contains('group', case=False) | pdf['Effect'].str.contains('time:group', case=False)]
        # normalize col names
        cols = {c: c for c in keep.columns}
        for c in list(cols):
            cl = c.lower()
            if cl.startswith('ats'): cols[c] = 'ATS'
            if cl.startswith('df'): cols[c] = 'df'
            if 'p-value' in cl or 'pvalue' in cl or 'pr > f' in cl or 'pr(>f' in cl: cols[c] = 'p-value'
        keep = keep.rename(columns=cols)
        for c in ["Effect", "ATS", "df", "p-value"]:
            if c not in keep.columns: keep[c] = np.nan
        return keep[["Effect", "ATS", "df", "p-value"]]
    except Exception as e:
        return _table_or_empty(["Effect", "ATS", "df", "p-value"], [[f"ERROR: {e}", np.nan, np.nan, np.nan]])

# -------------------------------
# Charts
# -------------------------------
def _fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def draw_boxplots(df_long: pd.DataFrame, ipss_cols: List[str]) -> List[Tuple[str, io.BytesIO]]:
    figs = []
    for col in ipss_cols:
        pl = (df_long[df_long["Time"]==col].loc[df_long["GROUP_NORM"]=="Placebo","Score"]).dropna()
        us = (df_long[df_long["Time"]==col].loc[df_long["GROUP_NORM"]=="USPlus","Score"]).dropna()
        if len(pl)==0 and len(us)==0: continue
        fig, ax = plt.subplots()
        ax.boxplot([pl.values, us.values], labels=["Placebo","USPlus"])
        ax.set_title(f"{col} — Box plot by group")
        ax.set_ylabel("IPSS Score")
        st.pyplot(fig, use_container_width=True)
        figs.append((f"Box: {col}", _fig_to_buf(fig)))
    return figs

def draw_trend(df_long: pd.DataFrame, ipss_cols: List[str]) -> List[Tuple[str, io.BytesIO]]:
    figs = []
    for g in ("Placebo","USPlus"):
        med=[]; q1=[]; q3=[]
        for col in ipss_cols:
            vals=df_long[(df_long["GROUP_NORM"]==g)&(df_long["Time"]==col)]["Score"].dropna().values
            med.append(np.nanmedian(vals) if vals.size else np.nan)
            q1.append(np.nanpercentile(vals,25) if vals.size else np.nan)
            q3.append(np.nanpercentile(vals,75) if vals.size else np.nan)
        fig, ax = plt.subplots()
        x=np.arange(len(ipss_cols))
        ax.plot(x, med, marker='o')
        ax.fill_between(x,q1,q3,alpha=0.2)
        ax.set_xticks(x); ax.set_xticklabels(ipss_cols, rotation=20)
        ax.set_title(f"{g}: Median trend with IQR")
        ax.set_ylabel("IPSS Score")
        st.pyplot(fig, use_container_width=True)
        figs.append((f"Trend Median+IQR: {g}", _fig_to_buf(fig)))
    return figs

# -------------------------------
# Export
# -------------------------------
def to_excel_with_results(original_df: pd.DataFrame, tables: Dict[str, pd.DataFrame], charts: List[Tuple[str, io.BytesIO]]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        original_df.to_excel(writer, index=False, sheet_name="Input")
        start = 0; sheet = "RESULTS"
        for title in [
            "ANCOVA_FINAL", "MMRM", "GEE",
            "MIXED_ANOVA_PINGOUIN", "RM_ANOVA_PINGOUIN", "RM_ANOVA_SM_WITHIN_ONLY",
            "MANOVA", "DELTA_BY_VISIT", "VAN_ELTEREN", "VAN_ELTEREN_STRATA_COUNTS",
            "RANK_ATS_NPARLD"
        ]:
            tbl = tables.get(title)
            pd.DataFrame({title: []}).to_excel(writer, index=False, sheet_name=sheet, startrow=start); start += 1
            if tbl is not None and not tbl.empty:
                tbl.to_excel(writer, index=False, sheet_name=sheet, startrow=start)
                start += len(tbl) + 3
            else:
                pd.DataFrame({"info":["No results"]}).to_excel(writer, index=False, sheet_name=sheet, startrow=start); start += 3
        # CHARTS sheet
        wb = writer.book
        ws = wb.add_worksheet("CHARTS")
        r = 0
        for title, buf in charts:
            ws.write(r, 0, title)
            ws.insert_image(r+1, 0, "chart.png", {"image_data": buf, "x_scale": 0.9, "y_scale": 0.9})
            r += 30
    return output.getvalue()

# -------------------------------
# UI
# -------------------------------
with st.sidebar:
    st.header("Upload Excel")
    uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])
    st.markdown("Labels like 'US Plus' or legacy 'Active' are auto-mapped to **USPlus**.")

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

# Build long format
long = make_long(df, ipss_cols)
final_tp = ipss_cols[-1]

# Run analyses
with st.spinner("Running between-group analyses..."):
    tables: Dict[str, pd.DataFrame] = {}
    tables["ANCOVA_FINAL"] = ancova_final(long, final_tp)
    tables["MMRM"] = mmrm(long)
    tables["GEE"] = gee(long)
    tables.update(mixed_or_rm_anova(long))
    tables["MANOVA"] = manova(long, ipss_cols)
    per_visit, ve, counts = delta_tests(long, ipss_cols)
    tables["DELTA_BY_VISIT"] = per_visit
    tables["VAN_ELTEREN"] = ve
    tables["VAN_ELTEREN_STRATA_COUNTS"] = counts
    tables["RANK_ATS_NPARLD"] = rank_ats_nparld(long)

# Display (end-of-trial first)
st.markdown(f"### End-of-trial (Final = {final_tp})")
st.dataframe(tables["ANCOVA_FINAL"], use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("MMRM — Mixed Linear Model")
    st.dataframe(tables["MMRM"], use_container_width=True)
with c2:
    st.subheader("GEE — Gaussian with AR(1)")
    st.dataframe(tables["GEE"], use_container_width=True)

st.subheader("Mixed / RM ANOVA")
if "MIXED_ANOVA_PINGOUIN" in tables:
    st.caption("Pingouin mixed ANOVA (between = Group, within = Time).")
    st.dataframe(tables["MIXED_ANOVA_PINGOUIN"], use_container_width=True)
if "RM_ANOVA_PINGOUIN" in tables:
    st.caption("Pingouin RM ANOVA (within = Time only).")
    st.dataframe(tables["RM_ANOVA_PINGOUIN"], use_container_width=True)
if "RM_ANOVA_SM_WITHIN_ONLY" in tables:
    st.caption("statsmodels AnovaRM (within = Time, fallback).")
    st.dataframe(tables["RM_ANOVA_SM_WITHIN_ONLY"], use_container_width=True)

st.subheader("MANOVA across Day 28/56/84")
st.dataframe(tables["MANOVA"], use_container_width=True)

st.subheader("Δ-per-visit (between groups) + Van Elteren")
st.dataframe(tables["DELTA_BY_VISIT"], use_container_width=True)
st.dataframe(tables["VAN_ELTEREN"], use_container_width=True)
with st.expander("Stratum sizes used in Van Elteren"):
    st.dataframe(tables["VAN_ELTEREN_STRATA_COUNTS"], use_container_width=True)

st.subheader("Rank-based ANOVA-type (nparLD)")
st.caption("Runs only if rpy2 + R + nparLD are available; otherwise a note is shown.")
st.dataframe(tables["RANK_ATS_NPARLD"], use_container_width=True)

# Charts on screen + collect for Excel
st.subheader("Charts ▸ Box plots & Median±IQR trends")
chart_bufs: List[Tuple[str, io.BytesIO]] = []
chart_bufs += draw_boxplots(long, ipss_cols)
chart_bufs += draw_trend(long, ipss_cols)

# Download Excel
excel_bytes = to_excel_with_results(df, tables, chart_bufs)
st.download_button(
    label="Download RESULTS workbook (with CHARTS)",
    data=excel_bytes,
    file_name="ipss_between_longitudinal_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("Done. Between-group results computed and exported.")
