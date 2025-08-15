"""
IPSS (4 Time Points) — Between-Group Longitudinal Suite (SAFE START)

- Boots even if some optional libs are missing (shows helpful notes instead of crashing).
- Between-group only (Placebo vs USPlus/Active aliases).
- Tests: ANCOVA(final), MMRM, GEE, Mixed/RM ANOVA, MANOVA, Δ-per-visit (Welch/MWU/BM),
  pooled Δ (Van Elteren), optional rank-based ATS (nparLD via R).
- Exports RESULTS + CHARTS sheets.

Name this file: app.py
"""

import io, re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import streamlit as st

# Optional
try:
    import pingouin as pg
except Exception:
    pg = None

st.set_page_config(page_title="IPSS — Between-Group Suite (Safe)", layout="wide")
st.title("IPSS (4 Time Points) — Placebo vs USPlus (Safe Start)")
st.caption("Graceful fallbacks: app runs even if some dependencies are missing. Advanced tests show guidance instead of crashing.")

# ---------- Debug panel ----------
with st.sidebar:
    st.header("Upload Excel")
    uploaded = st.file_uploader("Excel (.xlsx)", type=["xlsx"])
    st.divider()
    st.header("Debug")
    if st.checkbox("Show versions / env"):
        rows = []
        def ver(name, mod=None):
            try:
                if mod is None:
                    mod = __import__(name)
                rows.append([name, getattr(mod, "__version__", "unknown")])
            except Exception as e:
                rows.append([name, f"NOT INSTALLED ({e})"])
        ver("streamlit")
        ver("pandas")
        ver("numpy")
        ver("scipy")
        ver("matplotlib")
        # optional:
        try:
            import statsmodels as sm
            ver("statsmodels", sm)
        except Exception as e:
            rows.append(["statsmodels", f"NOT INSTALLED ({e})"])
        try:
            import pingouin as _pg
            ver("pingouin", _pg)
        except Exception as e:
            rows.append(["pingouin", f"NOT INSTALLED ({e})"])
        try:
            import rpy2 as _rpy2
            ver("rpy2", _rpy2)
        except Exception as e:
            rows.append(["rpy2", f"NOT INSTALLED ({e})"])
        st.dataframe(pd.DataFrame(rows, columns=["Package", "Version"]), use_container_width=True)

# ---------- Utils ----------
TIME_ORDER_MAP = {"baseline": 0, "day 28": 28, "day 56": 56, "day 84": 84}
IPSS_COL_REGEX = re.compile(r"^(baseline|day\s*28|day\s*56|day\s*84).*ipss.*total\s*score", re.IGNORECASE)
REQUIRED_ID_COLS = ["SUBJECT ID", "GROUP"]

def find_ipss_columns(columns: List[str]) -> List[str]:
    cand = [c for c in columns if IPSS_COL_REGEX.search(str(c))]
    def time_key(col: str) -> int:
        s = col.lower()
        for k, v in TIME_ORDER_MAP.items():
            if k in s: return v
        return 10_000
    return sorted(cand, key=time_key)

def validate_template(df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
    missing_ids = [c for c in REQUIRED_ID_COLS if c not in df.columns]
    if missing_ids:
        return False, f"Missing required columns: {missing_ids}", []
    ipss_cols = find_ipss_columns(df.columns.tolist())
    if len(ipss_cols) != 4:
        return False, (
            "Could not detect exactly 4 IPSS timepoint columns. "
            "Ensure columns contain Baseline/Day 28/Day 56/Day 84 and 'IPSS  Total Score'."
        ), ipss_cols
    return True, "", ipss_cols

def normalize_groups(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    mapping = {"placebo":"Placebo","control":"Placebo","usplus":"USPlus","us":"USPlus","active":"USPlus"}
    out = s.map(mapping); out.name = "GROUP_NORM"; return out

@st.cache_data(show_spinner=False)
def make_long(df: pd.DataFrame, ipss_cols: List[str]) -> pd.DataFrame:
    long = df.copy()
    long["GROUP_NORM"] = normalize_groups(long["GROUP"])
    bad = long.loc[long["GROUP_NORM"].isna(),"GROUP"].dropna().astype(str).unique()
    if len(bad):
        st.warning(f"Dropped rows with unrecognized GROUP labels: {', '.join(sorted(bad))[:300]}")
    long = long.loc[long["GROUP_NORM"].isin(["Placebo","USPlus"])].copy()
    long = long.dropna(subset=[ipss_cols[0]])
    long = long[["SUBJECT ID","GROUP_NORM"] + ipss_cols].copy()
    long = long.rename(columns={ipss_cols[0]:"Baseline"})
    long = long.melt(id_vars=["SUBJECT ID","GROUP_NORM","Baseline"], value_vars=ipss_cols,
                     var_name="Time", value_name="Score")
    long["Time"] = pd.Categorical(long["Time"], categories=ipss_cols, ordered=True)
    def _tn(s:str)->int:
        s=s.lower()
        if "28" in s: return 28
        if "56" in s: return 56
        if "84" in s: return 84
        return 0
    long["TimeNum"] = long["Time"].astype(str).map(_tn)
    return long

def df_note(msg: str, cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame([[msg] + [np.nan]*(len(cols)-1)], columns=cols)

# ---------- Analyses (advanced imports inside functions) ----------
def ancova_final(df_long: pd.DataFrame, final_col: str) -> pd.DataFrame:
    cols = ["Final timepoint","N Placebo","N USPlus","Adj diff (USPlus−Placebo)","SE (HC3)","95% CI lo","95% CI hi","p-value"]
    try:
        import statsmodels.formula.api as smf
    except Exception as e:
        return df_note(f"Install statsmodels to run ANCOVA ({e})", cols)
    wide = (df_long.drop_duplicates(["SUBJECT ID"])
            .merge(df_long[df_long["Time"]==final_col][["SUBJECT ID","Score"]].rename(columns={"Score":"Final"}),
                   on="SUBJECT ID", how="left")).dropna(subset=["Final"])
    if wide.empty: return df_note("No final-visit data", cols)
    m = smf.ols("Final ~ Baseline + C(GROUP_NORM)", data=wide).fit(cov_type="HC3")
    term = "C(GROUP_NORM)[T.USPlus]"
    import scipy.stats as stx
    est = float(m.params.get(term, np.nan)); se = float(m.bse.get(term, np.nan)); p = float(m.pvalues.get(term, np.nan))
    df_res = float(m.df_resid); tcrit = stx.t.ppf(0.975, df_res) if df_res>0 else np.nan
    lo = est - tcrit*se if np.isfinite(tcrit) else np.nan
    hi = est + tcrit*se if np.isfinite(tcrit) else np.nan
    n_p = int((wide["GROUP_NORM"]=="Placebo").sum()); n_u = int((wide["GROUP_NORM"]=="USPlus").sum())
    return pd.DataFrame([[final_col,n_p,n_u,est,se,lo,hi,p]], columns=cols)

def mmrm(df_long: pd.DataFrame) -> pd.DataFrame:
    cols = ["Effect","Param","Estimate","SE","z","p-value"]
    try:
        import statsmodels.formula.api as smf
    except Exception as e:
        return df_note(f"Install statsmodels for MMRM ({e})", cols)
    use = df_long[df_long["Time"] != df_long["Time"].cat.categories[0]].copy()
    if use.empty: return df_note("No post-baseline data", cols)
    use["TimeC"] = use["Time"].astype(str)
    try:
        md = smf.mixedlm("Score ~ C(TimeC) * C(GROUP_NORM) + Baseline",
                         data=use, groups=use["SUBJECT ID"], re_formula="~TimeNum")
        fit = md.fit(method="lbfgs", reml=True)
        rows=[]
        for p, est in fit.params.items():
            if p=="Intercept": eff="Intercept"
            elif "C(GROUP_NORM)" in p: eff="Group"
            elif "C(TimeC)" in p and ":" not in p: eff="Time"
            elif ":" in p: eff="Group×Time"
            else: eff="Covariate"
            se = fit.bse.get(p, np.nan)
            z  = est/se if (se not in (0, np.nan)) else np.nan
            pv = 2*(1-stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            rows.append([eff,p,est,se,z,pv])
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        return df_note(f"MMRM error: {e}", cols)

def gee(df_long: pd.DataFrame) -> pd.DataFrame:
    cols = ["Param","Estimate","SE","z","p-value"]
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        return df_note(f"Install statsmodels for GEE ({e})", cols)
    use = df_long[df_long["Time"] != df_long["Time"].cat.categories[0]].copy()
    if use.empty: return df_note("No post-baseline data", cols)
    use["TimeC"] = use["Time"].astype(str)
    try:
        model = smf.gee("Score ~ C(TimeC) * C(GROUP_NORM) + Baseline",
                        groups="SUBJECT ID", cov_struct=sm.cov_struct.Autoregressive(),
                        family=sm.families.Gaussian(), data=use)
        res = model.fit()
        rows=[]
        for p, est in res.params.items():
            se = res.bse[p]; z = est/se if (se not in (0,np.nan)) else np.nan
            rows.append([p, est, se, z, res.pvalues[p]])
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        return df_note(f"GEE error: {e}", cols)

def mixed_or_rm_anova(df_long: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    use = df_long[df_long["Time"] != df_long["Time"].cat.categories[0]].copy()
    if pg is not None:
        try:
            mx = pg.mixed_anova(data=use, dv="Score", within="Time", between="GROUP_NORM", subject="SUBJECT ID")
            out["MIXED_ANOVA_PINGOUIN"] = mx
        except Exception as e:
            out["MIXED_ANOVA_PINGOUIN"] = pd.DataFrame({"note":[f"Pingouin mixed_anova error: {e}"]})
        try:
            rm = pg.rm_anova(data=use, dv="Score", within="Time", subject="SUBJECT ID", detailed=True)
            out["RM_ANOVA_PINGOUIN"] = rm
        except Exception:
            pass
    # statsmodels fallback (within-only)
    try:
        from statsmodels.stats.anova import AnovaRM
        aov = AnovaRM(use, depvar="Score", subject="SUBJECT ID", within=["Time"]).fit()
        try:
            out["RM_ANOVA_SM_WITHIN_ONLY"] = aov.anova_table
        except Exception:
            import pandas as _pd
            out["RM_ANOVA_SM_WITHIN_ONLY"] = _pd.read_html(aov.summary().tables[0].as_html(), header=0, index_col=0)[0]
    except Exception as e:
        out["RM_ANOVA_SM_WITHIN_ONLY"] = pd.DataFrame({"note":[f"AnovaRM unavailable: {e}"]})
    return out

def manova(df_long: pd.DataFrame, ipss_cols: List[str]) -> pd.DataFrame:
    cols = ["Test","Statistic","df1","df2","p-value"]
    try:
        import statsmodels.api as sm
    except Exception as e:
        return df_note(f"Install statsmodels for MANOVA ({e})", cols)
    wide = df_long.pivot_table(index=["SUBJECT ID","GROUP_NORM","Baseline"], columns="Time",
                               values="Score", aggfunc="first").reset_index()
    visits = [c for c in ipss_cols[1:]]
    have = [c for c in visits if c in wide.columns]
    sub = wide.dropna(subset=have)
    if sub.empty or len(have)<2: return df_note("Not enough complete cases for MANOVA", cols)
    try:
        formula = " + ".join([f"Q('{c}')" for c in have]) + " ~ C(GROUP_NORM)"
        mv = sm.multivariate.MANOVA.from_formula(formula, data=sub)
        w = mv.mv_test()
        test_tbl = w.results['C(GROUP_NORM)']['stat']
        row = test_tbl.loc['Wilks' if 'Wilks' in test_tbl.index else test_tbl.index[0]]
        stat = float(row['Value']); df1=float(row['Num DF']); df2=float(row['Den DF']); p=float(row['Pr > F'])
        return pd.DataFrame([["Wilks' Lambda (Group)", stat, df1, df2, p]], columns=cols)
    except Exception as e:
        return df_note(f"MANOVA error: {e}", cols)

def delta_tests(df_long: pd.DataFrame, ipss_cols: List[str]):
    base = ipss_cols[0]
    rows=[]; us_by={}; pl_by={}
    for col in ipss_cols[1:]:
        d = (df_long[["SUBJECT ID","GROUP_NORM","Baseline"]].drop_duplicates("SUBJECT ID")
             .merge(df_long[df_long["Time"]==col][["SUBJECT ID","Score"]].rename(columns={"Score":col}),
                    on="SUBJECT ID", how="inner"))
        d["DELTA"] = d[col]-d["Baseline"]
        pl = d.loc[d["GROUP_NORM"]=="Placebo","DELTA"].dropna().to_numpy()
        us = d.loc[d["GROUP_NORM"]=="USPlus","DELTA"].dropna().to_numpy()
        us_by[col]=us; pl_by[col]=pl
        n_p, n_u = pl.size, us.size
        note=None
        if n_p>=2 and n_u>=2:
            try: _, p_t = stats.ttest_ind(pl, us, equal_var=False)
            except Exception: p_t=np.nan
            try: _, p_u = stats.mannwhitneyu(pl, us, alternative="two-sided")
            except Exception: p_u=np.nan
            try: _, p_bm = stats.brunnermunzel(pl, us, alternative="two-sided")
            except Exception: p_bm=np.nan
        else:
            p_t=p_u=p_bm=np.nan; note="Insufficient N"
        rows += [[f"{base} → {col}", n_p, n_u, "Welch t-test (indep)", p_t, note],
                 [f"{base} → {col}", n_p, n_u, "Mann–Whitney U", p_u, note],
                 [f"{base} → {col}", n_p, n_u, "Brunner–Munzel", p_bm, note]]
    per_visit = pd.DataFrame(rows, columns=["Change (Δ)","N Placebo","N USPlus","Test","p-value","Note"])
    # Van Elteren pooled Δ
    Z, p_ve, counts = van_elteren(us_by, pl_by, ipss_cols)
    ve = pd.DataFrame([["Van Elteren (Δ across visits; strata = visits)", Z, p_ve]], columns=["Analysis","Z","p-value"])
    ct = pd.DataFrame(counts, columns=["Visit (stratum)","N Placebo","N USPlus"])
    return per_visit, ve, ct

def van_elteren(us_by, pl_by, ipss_cols):
    def _stats(ax, px):
        n1, n2 = len(ax), len(px); N=n1+n2
        pooled = np.concatenate([ax,px]); ranks = stats.rankdata(pooled, method="average")
        W = ranks[:n1].sum(); E = n1*(N+1)/2.0
        vals, counts = np.unique(pooled, return_counts=True)
        tie = ((counts**3-counts).sum())/(N*(N-1)) if N>1 else 0.0
        Var = (n1*n2/12.0)*((N+1)-tie)
        return W,E,Var
    W=E=V=0.0; counts=[]
    for v in ipss_cols[1:]:
        ax=np.asarray(us_by.get(v,[])); px=np.asarray(pl_by.get(v,[]))
        if len(ax)<1 or len(px)<1: continue
        w,e,var=_stats(ax,px); W+=w; E+=e; V+=var; counts.append((v, int(len(px)), int(len(ax))))
    if V<=0 or not counts: return None,None,counts
    Z=(W-E)/np.sqrt(V); p=2*(1-stats.norm.cdf(abs(Z))); return float(Z), float(p), counts

# ---------- Charts ----------
def _fig_to_buf(fig)->io.BytesIO:
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0); return buf

def draw_boxplots(df_long: pd.DataFrame, ipss_cols: List[str]):
    figs=[]
    for col in ipss_cols:
        sub=df_long[df_long["Time"]==col]
        pl=sub.loc[sub["GROUP_NORM"]=="Placebo","Score"].dropna()
        us=sub.loc[sub["GROUP_NORM"]=="USPlus","Score"].dropna()
        if len(pl)==0 and len(us)==0: continue
        fig, ax = plt.subplots(); ax.boxplot([pl.values, us.values], labels=["Placebo","USPlus"])
        ax.set_title(f"{col} — Box plot by group"); ax.set_ylabel("IPSS Score")
        st.pyplot(fig, use_container_width=True); figs.append((f"Box: {col}", _fig_to_buf(fig)))
    return figs

def draw_trend(df_long: pd.DataFrame, ipss_cols: List[str]):
    figs=[]
    for g in ("Placebo","USPlus"):
        med=[]; q1=[]; q3=[]
        for col in ipss_cols:
            vals=df_long[(df_long["GROUP_NORM"]==g)&(df_long["Time"]==col)]["Score"].dropna().values
            med.append(np.nanmedian(vals) if vals.size else np.nan)
            q1.append(np.nanpercentile(vals,25) if vals.size else np.nan)
            q3.append(np.nanpercentile(vals,75) if vals.size else np.nan)
        fig, ax = plt.subplots(); x=np.arange(len(ipss_cols))
        ax.plot(x, med, marker='o'); ax.fill_between(x, q1, q3, alpha=0.2)
        ax.set_xticks(x); ax.set_xticklabels(ipss_cols, rotation=20); ax.set_title(f"{g}: Median trend with IQR"); ax.set_ylabel("IPSS Score")
        st.pyplot(fig, use_container_width=True); figs.append((f"Trend Median+IQR: {g}", _fig_to_buf(fig)))
    return figs

# ---------- Export ----------
def to_excel_with_results(original_df: pd.DataFrame, tables: Dict[str, pd.DataFrame], charts):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        original_df.to_excel(writer, index=False, sheet_name="Input")
        start=0; sheet="RESULTS"
        order=[
            "ANCOVA_FINAL","MMRM","GEE","MIXED_ANOVA_PINGOUIN","RM_ANOVA_PINGOUIN","RM_ANOVA_SM_WITHIN_ONLY",
            "MANOVA","DELTA_BY_VISIT","VAN_ELTEREN","VAN_ELTEREN_STRATA_COUNTS","RANK_ATS_NPARLD"
        ]
        for name in order:
            tbl = tables.get(name, pd.DataFrame())
            pd.DataFrame({name:[]}).to_excel(writer, index=False, sheet_name=sheet, startrow=start); start+=1
            if not tbl.empty:
                tbl.to_excel(writer, index=False, sheet_name=sheet, startrow=start); start+=len(tbl)+3
            else:
                pd.DataFrame({"info":["No results / dependency missing"]}).to_excel(writer, index=False, sheet_name=sheet, startrow=start); start+=3
        # CHARTS
        wb=writer.book; ws=wb.add_worksheet("CHARTS"); r=0
        for title, buf in charts:
            ws.write(r,0,title); ws.insert_image(r+1,0,"chart.png", {"image_data":buf,"x_scale":0.9,"y_scale":0.9}); r+=30
    return output.getvalue()

# ---------- Early exit if no file ----------
if uploaded is None:
    st.info("Upload an Excel to begin.")
    st.stop()

# ---------- Read Excel ----------
try:
    df = pd.read_excel(uploaded, sheet_name=0)
except Exception as e:
    st.error(f"Failed to read Excel: {e}")
    st.stop()

ok, msg, ipss_cols = validate_template(df)
if not ok:
    st.error(msg)
    if ipss_cols: st.write("Detected IPSS-like columns:", ipss_cols)
    st.stop()

st.subheader("Detected Time Points"); st.write(ipss_cols)
with st.expander("Preview first 10 rows"):
    st.dataframe(df.head(10), use_container_width=True)

# ---------- Build long & run ----------
long = make_long(df, ipss_cols)
final_tp = ipss_cols[-1]

with st.spinner("Running between-group analyses..."):
    tables: Dict[str, pd.DataFrame] = {}
    # ANCOVA / MMRM / GEE / ANOVA / MANOVA
    tables["ANCOVA_FINAL"] = ancova_final(long, final_tp)
    tables["MMRM"] = mmrm(long)
    tables["GEE"] = gee(long)
    tables.update(mixed_or_rm_anova(long))
    tables["MANOVA"] = manova(long, ipss_cols)
    # Δ + Van Elteren
    per, ve, ct = delta_tests(long, ipss_cols)
    tables["DELTA_BY_VISIT"] = per
    tables["VAN_ELTEREN"] = ve
    tables["VAN_ELTEREN_STRATA_COUNTS"] = ct
    # Rank-based ATS via R (optional)
    def rank_ats_nparld(df_long: pd.DataFrame):
        cols=["Effect","ATS","df","p-value"]
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            ok = bool(ro.r("'nparLD' %in% rownames(installed.packages())")[0])
            if not ok: return df_note("R package 'nparLD' not installed", cols)
            use = df_long[df_long["Time"] != df_long["Time"].cat.categories[0]].copy()
            use = use.rename(columns={"SUBJECT ID":"ID","GROUP_NORM":"Group"})
            r_df = pandas2ri.py2rpy(use[["ID","Group","Time","Score"]]); ro.globalenv["d"]=r_df
            ro.r("d$ID <- as.factor(d$ID); d$Group <- as.factor(d$Group); d$Time <- as.factor(d$Time)")
            ro.r("library(nparLD)"); ro.r("res <- f1.ld.f1(y=Score, time1=Time, group=Group, subject=ID, data=d, description=FALSE)")
            tab = ro.r("as.data.frame(res$ANOVA.test)")
            pdf = pandas2ri.rpy2py(tab).reset_index().rename(columns={"index":"Effect"})
            keep = pdf[pdf["Effect"].str.contains("group", case=False) | pdf["Effect"].str.contains("time:group", case=False)]
            # standardize column names
            for c in list(keep.columns):
                cl=c.lower()
                if cl.startswith("ats"): keep = keep.rename(columns={c:"ATS"})
                if cl.startswith("df"):  keep = keep.rename(columns={c:"df"})
                if "p-value" in cl or "pvalue" in cl or "pr(>f" in cl or "pr > f" in cl:
                    keep = keep.rename(columns={c:"p-value"})
            for c in ["Effect","ATS","df","p-value"]:
                if c not in keep.columns: keep[c]=np.nan
            return keep[["Effect","ATS","df","p-value"]]
        except Exception as e:
            return df_note(f"Rank ATS via R not available: {e}", cols)
    tables["RANK_ATS_NPARLD"] = rank_ats_nparld(long)

# ---------- Display (End-of-trial first) ----------
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
if "MIXED_ANOVA_PINGOUIN" in tables: st.dataframe(tables["MIXED_ANOVA_PINGOUIN"], use_container_width=True)
if "RM_ANOVA_PINGOUIN" in tables:   st.dataframe(tables["RM_ANOVA_PINGOUIN"], use_container_width=True)
if "RM_ANOVA_SM_WITHIN_ONLY" in tables: st.dataframe(tables["RM_ANOVA_SM_WITHIN_ONLY"], use_container_width=True)

st.subheader("MANOVA across Day 28/56/84")
st.dataframe(tables["MANOVA"], use_container_width=True)

st.subheader("Δ-per-visit (between groups) + Van Elteren")
st.dataframe(tables["DELTA_BY_VISIT"], use_container_width=True)
st.dataframe(tables["VAN_ELTEREN"], use_container_width=True)
with st.expander("Stratum sizes used in Van Elteren"):
    st.dataframe(tables["VAN_ELTEREN_STRATA_COUNTS"], use_container_width=True)

st.subheader("Rank-based ANOVA-type (nparLD)")
st.dataframe(tables["RANK_ATS_NPARLD"], use_container_width=True)

# ---------- Charts ----------
st.subheader("Charts ▸ Box plots & Median±IQR trends")
chart_bufs=[]; chart_bufs += draw_boxplots(long, ipss_cols); chart_bufs += draw_trend(long, ipss_cols)

# ---------- Download ----------
excel_bytes = to_excel_with_results(df, tables, chart_bufs)
st.download_button("Download RESULTS workbook (with CHARTS)", excel_bytes,
                   file_name="ipss_between_longitudinal_results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.success("Ready. If some tables show 'Install ...', just add that package to requirements.txt and redeploy.")
