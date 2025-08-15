import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---- Hard dependency guard for Excel (.xlsx) ----
try:
    import openpyxl  # noqa: F401
except Exception as e:
    st.error("This app requires the 'openpyxl' package to read .xlsx files.\n"
             "Add 'openpyxl' to requirements.txt at the repo root and redeploy.\n\n"
             f"Import error: {e}")
    st.stop()

from scipy.stats import mannwhitneyu, brunnermunzel, ttest_rel, ttest_ind, wilcoxon, fisher_exact, norm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import mixedlm

st.set_page_config(page_title="Clinical Analysis Suite v2", layout="wide")
st.title("Clinical Analysis Suite v2 — Multi-Endpoint • Analysis-Sets • Custom Responders • Weight Modeling")

st.caption(
    "Map your columns (Subject, Group, optional Status/Age/Weight/BMI, Baseline, Day28/56/84). "
    "Choose analysis sets (ITT/PP/Completers/filters), responder rules, and run model-based and nonparam tests. "
    "Includes: **MMRM Group×Weight** (final-visit effect vs weight) and **maximally-selected weight cut** (permutation-adjusted)."
)

DEFAULT_VISITS = ["Day 28", "Day 56", "Day 84"]
VISIT_KEYS     = ["Day28", "Day56", "Day84"]  # internal keys

def norm(s): return str(s).strip().lower()
def try_numeric(series): return pd.to_numeric(series, errors="coerce")
def ensure_cat_time(df, order): df["Time"] = pd.Categorical(df["Time"], categories=order, ordered=True); return df

def safe_pct_improve(baseline, final):
    baseline = np.asarray(baseline, dtype=float)
    final = np.asarray(final, dtype=float)
    out = np.full_like(baseline, np.nan, dtype=float)
    mask = np.isfinite(baseline) & np.isfinite(final) & (baseline != 0)
    out[mask] = (baseline[mask] - final[mask]) / baseline[mask]
    return out

# ---------- Upload ----------
st.subheader("1) Upload Excel")
uploaded = st.file_uploader("Upload a workbook with a single endpoint table (Baseline, Day28/56/84).", type=["xlsx"])
if not uploaded: st.stop()

# Use BytesIO for safety on Streamlit Cloud
file_bytes = uploaded.read()
bio = io.BytesIO(file_bytes)

# Sheet and preview
try:
    xl = pd.ExcelFile(bio, engine="openpyxl")
    sheet_name = st.selectbox("Select sheet", xl.sheet_names, index=0)
    raw = pd.read_excel(bio, sheet_name=sheet_name, engine="openpyxl")
except Exception as e:
    st.error("Failed to read the uploaded .xlsx file with openpyxl. "
             "Please confirm the file is a valid Excel workbook. "
             f"Details: {e}")
    st.stop()

st.write("Preview:", raw.head())

# ---------- Column mapping ----------
st.subheader("2) Map columns")
cols = list(raw.columns)
col_subject = st.selectbox("Subject ID column", cols, index=0)
col_group   = st.selectbox("Group column", cols, index=1 if len(cols)>1 else 0)
opt_status  = st.selectbox("Status column (optional)", ["<none>"] + cols, index=0)
opt_age     = st.selectbox("Age column (optional)", ["<none>"] + cols, index=0)
opt_weight  = st.selectbox("Weight column (optional)", ["<none>"] + cols, index=0)
opt_bmi     = st.selectbox("BMI column (optional)", ["<none>"] + cols, index=0)

col_baseline = st.selectbox("Baseline column", cols, index=min(2, len(cols)-1))
col_d28      = st.selectbox("Day 28 column", cols, index=min(3, len(cols)-1))
col_d56      = st.selectbox("Day 56 column", cols, index=min(4, len(cols)-1))
col_d84      = st.selectbox("Day 84 column", cols, index=min(5, len(cols)-1))
endpoint_name = st.text_input("Endpoint name", value="IPSS")

# Standardize
inp = pd.DataFrame({
    "Subject": raw[col_subject],
    "Group": raw[col_group].astype(str).str.strip(),
    "Baseline": try_numeric(raw[col_baseline]),
    "Day28": try_numeric(raw[col_d28]),
    "Day56": try_numeric(raw[col_d56]),
    "Day84": try_numeric(raw[col_d84]),
})
if opt_status != "<none>": inp["Status"] = raw[opt_status].astype(str)
if opt_age    != "<none>": inp["Age"]    = try_numeric(raw[opt_age])
if opt_weight != "<none>": inp["Weight"] = try_numeric(raw[opt_weight])
if opt_bmi    != "<none>": inp["BMI"]    = try_numeric(raw[opt_bmi])

# Long change + raw
rows = []
for _, r in inp.iterrows():
    for v in ["Day28","Day56","Day84"]:
        if pd.notna(r[v]):
            rows.append({"Subject": r["Subject"], "Group": r["Group"], "Time": v, "Change": float(r[v]-r["Baseline"])})
long_change = ensure_cat_time(pd.DataFrame(rows), ["Day28","Day56","Day84"]) if len(rows)>0 else pd.DataFrame()

rows_raw = []
for _, r in inp.iterrows():
    for v in ["Day28","Day56","Day84"]:
        if pd.notna(r[v]):
            rows_raw.append({"Subject": r["Subject"], "Group": r["Group"], "Time": v,
                             endpoint_name: float(r[v]), "Baseline": float(r["Baseline"]),
                             "Weight": r.get("Weight", np.nan)})
long_raw = ensure_cat_time(pd.DataFrame(rows_raw), ["Day28","Day56","Day84"]) if len(rows_raw)>0 else pd.DataFrame()

# ---------- Analysis sets & filters ----------
st.subheader("3) Analysis sets & filters")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: use_itt = st.checkbox("ITT/mITT", value=True)
with c2: use_pp = st.checkbox("PP (Status==ACTIVE)", value=False)
with c3: use_completers = st.checkbox("Completers (has Day84)", value=False)
with c4: excl_dae = st.checkbox("Exclude 'Dropped AE'", value=False)
with c5: custom_pair = st.checkbox("Custom pair", value=True)

c6, c7 = st.columns(2)
with c6:
    w_thr_on = st.checkbox("Filter: Weight < threshold", value=False)
    w_thr = st.number_input("Weight threshold (kg)", value=120.0, step=1.0)
with c7:
    b_thr_on = st.checkbox("Filter: BMI < threshold", value=False)
    b_thr = st.number_input("BMI threshold", value=35.0, step=0.5)

# Pair groups
groups_all = sorted(inp["Group"].dropna().unique().tolist())
if custom_pair and len(groups_all) >= 2:
    colA, colB = st.columns(2)
    with colA: gA = st.selectbox("Group A", groups_all, index=0)
    with colB: gB = st.selectbox("Group B", groups_all, index=1)
else:
    gA, gB = (groups_all[0], groups_all[1]) if len(groups_all)>=2 else (None, None)

# Subject mask
keep = pd.Series(True, index=inp["Subject"])
if excl_dae and ("Status" in inp.columns):
    bad = inp["Subject"][inp["Status"].astype(str).str.contains("Dropped AE", case=False, na=False)]
    keep.loc[bad] = False
if use_pp and ("Status" in inp.columns):
    keep &= (inp["Status"].astype(str).str.upper() == "ACTIVE")
if use_completers: keep &= inp["Day84"].notna()
if w_thr_on and ("Weight" in inp.columns): keep &= (inp["Weight"] < w_thr)
if b_thr_on and ("BMI" in inp.columns): keep &= (inp["BMI"] < b_thr)

keep_ids = set(inp["Subject"][keep])
inp_f = inp[inp["Subject"].isin(keep_ids)].copy()
lc_f  = long_change[long_change["Subject"].isin(keep_ids)].copy()
lr_f  = long_raw[long_raw["Subject"].isin(keep_ids)].copy()

st.write(f"Kept N subjects: {len(keep_ids)}")

# ---------- RM (rank ANOVA) ----------
st.subheader("4) Repeated measures (rank ANOVA on change)")
if not lc_f.empty:
    lc = lc_f.copy(); lc["rank_change"] = lc["Change"].rank(method="average")
    model = ols("rank_change ~ C(Subject) + C(Group) * C(Time)", data=lc).fit()
    an = anova_lm(model, typ=3)
    rm_df = pd.DataFrame({
        "Effect": ["Group (all groups)", "Time (Day28, Day56, Day84)", "Group × Time"],
        "p-value": [float(an.loc["C(Group)","PR(>F)"]),
                    float(an.loc["C(Time)","PR(>F)"]),
                    float(an.loc["C(Group):C(Time)","PR(>F)"])]
    })
else:
    rm_df = pd.DataFrame({"Effect":["Group (all groups)", "Time", "Group × Time"], "p-value":[np.nan,np.nan,np.nan]})
st.dataframe(rm_df)

# ---------- Pairwise per visit ----------
st.subheader("5) Pairwise tests per visit")
pair_df = pd.DataFrame()
if gA and gB and not lc_f.empty:
    rows = []
    for v in ["Day28","Day56","Day84"]:
        a = lc_f[(lc_f["Group"]==gA) & (lc_f["Time"]==v)]["Change"].dropna()
        b = lc_f[(lc_f["Group"]==gB) & (lc_f["Time"]==v)]["Change"].dropna()
        if len(a)>0 and len(b)>0:
            U, mw_p = mannwhitneyu(a, b, alternative="two-sided")
            bm_stat = bm_p = np.nan
            if len(a)>3 and len(b)>3: bm_stat, bm_p = brunnermunzel(a, b, alternative="two-sided")
            t_stat = t_p = (np.nan, np.nan)
            if len(a)>1 and len(b)>1: t_stat, t_p = ttest_ind(a, b, equal_var=False)
            rows.append({"Visit": v, f"N {gA}":len(a), f"N {gB}":len(b),
                         "Mann–Whitney p": float(mw_p),
                         "Brunner–Munzel p": float(bm_p) if bm_p==bm_p else np.nan,
                         "Welch t p": float(t_p) if t_p==t_p else np.nan})
        else:
            rows.append({"Visit": v, f"N {gA}":len(a), f"N {gB}":len(b),
                         "Mann–Whitney p": np.nan, "Brunner–Munzel p": np.nan, "Welch t p": np.nan})
    pair_df = pd.DataFrame(rows)
st.dataframe(pair_df)

# ---------- Within-group ----------
st.subheader("6) Within-group change tests")
within_rows = []
if not inp_f.empty:
    for grp in sorted(inp_f["Group"].unique().tolist()):
        g = inp_f[inp_f["Group"]==grp]
        base = g["Baseline"]
        for v in ["Day28","Day56","Day84"]:
            fol = g[v]
            paired = pd.concat([base, fol], axis=1).dropna()
            if paired.shape[0]>0:
                try: w_p = wilcoxon(paired.iloc[:,0], paired.iloc[:,1], alternative="two-sided", zero_method="wilcox")[1]
                except Exception: w_p = np.nan
                try: t_p = ttest_rel(paired.iloc[:,0], paired.iloc[:,1])[1]
                except Exception: t_p = np.nan
            else:
                w_p = t_p = np.nan
            within_rows.append({"Group":grp, "Visit":v, "N (paired)":int(paired.shape[0]),
                                "Wilcoxon p": float(w_p) if w_p==w_p else np.nan,
                                "Paired t p": float(t_p) if t_p==t_p else np.nan})
within_df = pd.DataFrame(within_rows)
st.dataframe(within_df)

# ---------- End-of-trial (ANCOVA & MMRM) ----------
st.subheader("7) End-of-trial (final visit) — model-based p-values")
final_key = "Day84"
anc_row = {"Method":"ANCOVA Δ(final)","p-value":np.nan}
if not inp_f.empty and final_key in inp_f.columns:
    use = inp_f[["Subject","Group","Baseline",final_key]].dropna().copy()
    if use["Group"].nunique() >= 2:
        use["chg"] = use[final_key] - use["Baseline"]
        use["Group_bin"] = (use["Group"].astype(str).str.strip() != "Placebo").astype(int)
        m = smf.ols("chg ~ Group_bin + Baseline", data=use).fit(cov_type="HC3")
        anc_row["p-value"] = float(m.pvalues.get("Group_bin", np.nan))
mmrm_row = {"Method":"MMRM Day-84 contrast","p-value":np.nan}
if not lr_f.empty:
    lr = lr_f.dropna(subset=[endpoint_name,"Group","Time","Baseline"]).copy()
    if not lr.empty:
        lr["BL_centered"] = lr["Baseline"] - lr["Baseline"].mean()
        lr["Group_bin"]   = (lr["Group"].astype(str).str.strip() != "Placebo").astype(int)
        try:
            model = mixedlm(f"{endpoint_name} ~ BL_centered + C(Time) + Group_bin + Group_bin:C(Time)",
                            lr, groups=lr["Subject"], re_formula="1").fit(method="lbfgs")
            names = model.model.exog_names
            coef = model.params.get("Group_bin", 0.0) + model.params.get("Group_bin:C(Time)[T.Day84]", 0.0)
            cvec = np.zeros(len(names))
            if "Group_bin" in names: cvec[names.index("Group_bin")] = 1.0
            if "Group_bin:C(Time)[T.Day84]" in names: cvec[names.index("Group_bin:C(Time)[T.Day84]")] = 1.0
            cov = model.cov_params().loc[names, names].values
            se = float(np.sqrt(cvec @ cov @ cvec.T))
            z = coef / se if se != 0 else np.nan
            mmrm_row["p-value"] = float(2*(1-norm.cdf(abs(z)))) if z==z else np.nan
        except Exception as e:
            st.warning(f"MMRM fit failed: {e}")
final_model_df = pd.DataFrame([anc_row, mmrm_row])
st.dataframe(final_model_df)

# ---------- MMRM Group×Weight ----------
st.subheader("8) MMRM with Group×Weight — final visit effect vs weight")
run_gxw = st.checkbox("Run Group×Weight model and plot Day-84 effect vs weight", value=False)
gxw_df = pd.DataFrame()
if run_gxw and ("Weight" in lr_f.columns) and not lr_f["Weight"].isna().all():
    lr = lr_f.dropna(subset=[endpoint_name, "Group", "Time", "Baseline", "Weight"]).copy()
    lr["BL_centered"] = lr["Baseline"] - lr["Baseline"].mean()
    lr["Group_bin"]   = (lr["Group"].astype(str).str.strip() != "Placebo").astype(int)
    lr["Weight_c"]    = lr["Weight"] - lr["Weight"].mean()
    try:
        model = mixedlm(f"{endpoint_name} ~ BL_centered + C(Time) + Group_bin + Weight_c + Group_bin:Weight_c + Group_bin:C(Time)",
                        lr, groups=lr["Subject"], re_formula="1").fit(method="lbfgs")
        names = model.model.exog_names; params = model.params[names]; cov = model.cov_params().loc[names, names].values
        weights = np.linspace(lr["Weight"].min(), lr["Weight"].max(), 100)
        rows = []
        for w in weights:
            wc = w - lr["Weight"].mean()
            cvec = np.zeros(len(names))
            if "Group_bin" in names: cvec[names.index("Group_bin")] = 1.0
            key84 = "Group_bin:C(Time)[T.Day84]"
            if key84 in names: cvec[names.index(key84)] = 1.0
            if "Group_bin:Weight_c" in names: cvec[names.index("Group_bin:Weight_c")] = wc
            diff = float(cvec @ params.values); se = float(np.sqrt(cvec @ cov @ cvec.T))
            rows.append({"Weight": w, "Effect (USPlus−Placebo)": diff, "LCL": diff-1.96*se, "UCL": diff+1.96*se})
        gxw_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(gxw_df["Weight"], gxw_df["Effect (USPlus−Placebo)"])
        ax.fill_between(gxw_df["Weight"], gxw_df["LCL"], gxw_df["UCL"], alpha=0.3)
        ax.axhline(0, linestyle="--")
        ax.set_title(f"{endpoint_name}: Day-84 effect vs Weight (negative favors USPlus)")
        ax.set_xlabel("Weight (kg)"); ax.set_ylabel("Effect (USPlus − Placebo)")
        st.pyplot(fig); st.dataframe(gxw_df.round(3))
    except Exception as e:
        st.warning(f"Group×Weight model could not be fit: {e}")
else:
    st.caption("Toggle the checkbox to run; requires a Weight column.")

# ---------- Maximally selected cut ----------
st.subheader("9) Maximally-selected weight threshold (ΔDay84 ANCOVA)")
run_cut = st.checkbox("Run threshold finder with permutation-adjusted p-value", value=False)
perm_N = st.number_input("Permutations", value=200, min_value=50, max_value=2000, step=50)
cut_low_pct  = st.slider("Search lower percentile of weight", 0, 40, 10)
cut_high_pct = st.slider("Search upper percentile of weight", 60, 100, 90)
cut_results = {}; cut_table = pd.DataFrame()

if run_cut and not inp_f.empty and "Weight" in inp_f.columns:
    w = inp_f["Weight"].dropna().to_numpy()
    if w.size >= 10:
        lo = np.percentile(w, cut_low_pct); hi = np.percentile(w, cut_high_pct)
        candidates = np.unique(np.round(np.linspace(lo, hi, 25), 2))
        rows = []
        for t in candidates:
            sub = inp_f[(inp_f["Weight"] < t) & inp_f["Day84"].notna()].copy()
            if sub["Group"].nunique() < 2 or sub.shape[0] < 10:
                rows.append({"Cut_kg": t, "N": sub.shape[0], "p_value": np.nan}); continue
            sub["chg"] = sub["Day84"] - sub["Baseline"]
            sub["Group_bin"] = (sub["Group"].astype(str).str.strip() != "Placebo").astype(int)
            m = smf.ols("chg ~ Group_bin + Baseline", data=sub).fit(cov_type="HC3")
            p = float(m.pvalues.get("Group_bin", np.nan))
            rows.append({"Cut_kg": t, "N": sub.shape[0], "p_value": p})
        cut_table = pd.DataFrame(rows).dropna(subset=["p_value"])
        if cut_table.empty:
            st.warning("No valid candidate cuts (insufficient data in subsets).")
        else:
            obs_min_p = cut_table["p_value"].min()
            best_cut = float(cut_table.loc[cut_table["p_value"].idxmin(), "Cut_kg"])
            st.write("Observed best cut:", best_cut, "kg; min p:", obs_min_p)
            use = inp_f[["Subject","Group","Baseline","Day84","Weight"]].dropna().copy()
            use["Group_bin"] = (use["Group"].astype(str).str.strip() != "Placebo").astype(int)
            rng = np.random.default_rng(42)
            perm_min_ps = []
            for i in range(int(perm_N)):
                shuffled = use.copy()
                shuffled["Group_bin"] = rng.permutation(shuffled["Group_bin"].values)
                min_p_i = 1.0
                for t in candidates:
                    sub = shuffled[(shuffled["Weight"] < t)].copy()
                    if sub.shape[0] < 10 or sub["Group_bin"].nunique()<2: continue
                    sub["chg"] = sub["Day84"] - sub["Baseline"]
                    m = smf.ols("chg ~ Group_bin + Baseline", data=sub).fit()
                    p = m.pvalues.get("Group_bin", np.nan)
                    if pd.notna(p) and p < min_p_i: min_p_i = p
                perm_min_ps.append(min_p_i)
            perm_min_ps = np.array(perm_min_ps)
            adj_p = float((perm_min_ps <= obs_min_p).mean())
            cut_results = {"Best_cut_kg": best_cut, "Observed_min_p": obs_min_p, "Adjusted_p": adj_p}
            st.write(f"Permutation-adjusted p-value: **{adj_p:.4f}**")
            st.dataframe(cut_table.sort_values("p_value"))
    else:
        st.info("Not enough weight data (need ≥10 subjects with weight).")
else:
    st.caption("Toggle the checkbox to run; requires Weight and Day84.")

# ---------- Responders ----------
st.subheader("10) Responders at final visit (custom rules)")
colR1, colR2, colR3 = st.columns(3)
with colR1: thr_points = st.number_input("Absolute improvement ≥ (points)", value=5, step=1)
with colR2: thr_pct = st.number_input("Percent improvement ≥ (%)", value=25, step=5) / 100.0
with colR3: dir_lower_better = st.selectbox("Direction (lower score = better?)", ["Yes","No"], index=0) == "Yes"

resp_rows = []
if not inp_f.empty and "Day84" in inp_f.columns:
    use = inp_f[["Subject","Group","Baseline","Day84"]].dropna().copy()
    if use["Group"].nunique() >= 2:
        if dir_lower_better:
            use["chg"] = use["Day84"] - use["Baseline"]
            abs_resp = (use["chg"] <= -thr_points).astype(int)
            pct_resp = (safe_pct_improve(use["Baseline"], use["Day84"]) >= thr_pct).astype(int)
        else:
            use["chg"] = use["Day84"] - use["Baseline"]
            abs_resp = (use["chg"] >= thr_points).astype(int)
            pct_resp = (((use["Day84"] - use["Baseline"]) / use["Baseline"]) >= thr_pct).astype(int)
        for label, resp in [(f"≥{thr_points} points", abs_resp),
                            (f"≥{int(thr_pct*100)}%", pct_resp)]:
            tab = pd.crosstab((use["Group"].astype(str).str.strip()!="Placebo"), resp)
            if tab.shape == (2,2):
                OR, p = fisher_exact(tab.values, alternative="greater")
                resp_rows.append({"Endpoint":label, "Odds Ratio":float(OR), "p-value (1-sided)":float(p)})
            else:
                resp_rows.append({"Endpoint":label, "Odds Ratio":np.nan, "p-value (1-sided)":np.nan})
resp_df = pd.DataFrame(resp_rows)
st.dataframe(resp_df)

# ---------- Export ----------
st.subheader("11) Download Excel with all results")
out = io.BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    inp_f.to_excel(writer, sheet_name="INPUT_FILTERED", index=False)
    lc_f.to_excel(writer, sheet_name="CHANGE_LONG", index=False)
    if not lr_f.empty: lr_f.to_excel(writer, sheet_name="RAW_LONG", index=False)
    rm_df.to_excel(writer, sheet_name="RM_NONPARAM", index=False)
    pair_df.to_excel(writer, sheet_name="PAIRWISE", index=False)
    within_df.to_excel(writer, sheet_name="WITHIN_GROUP", index=False)
    final_model_df.to_excel(writer, sheet_name="FINAL_MODELS", index=False)
    if not gxw_df.empty: gxw_df.to_excel(writer, sheet_name="MMRM_GxW_EffectVsWeight", index=False)
    if isinstance(cut_results, dict) and cut_results:
        pd.DataFrame([cut_results]).to_excel(writer, sheet_name="CUT_RESULT", index=False)
    if not resp_df.empty: resp_df.to_excel(writer, sheet_name="RESPONDERS", index=False)
out.seek(0)
base = uploaded.name.rsplit(".",1)[0]
st.download_button(
    "⬇️ Download results workbook",
    data=out,
    file_name=f"{base}_{endpoint_name}_RESULTS_v2.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
