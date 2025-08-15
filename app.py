
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, brunnermunzel, ttest_rel, ttest_ind, wilcoxon
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="Nonparam P-Values — Pairwise + Repeated Measures", layout="wide")
st.title("Nonparametric Analysis — Pairwise (A vs B) + Repeated Measures (All Groups)")
st.caption(
    "Upload an Excel with either an **Input** sheet (Baseline & Day 28/56/84 columns) "
    "or a **Change Wide** sheet (Day28_change/Day56_change/Day84_change). "
    "Select any two groups for pairwise testing (MWU/BM/Welch). "
    "Repeated-measures (Group, Time, Group×Time) is run across **all groups**."
)

# ---------- Settings ----------
DEFAULT_VISITS = ["Day28", "Day56", "Day84"]
with st.sidebar:
    st.header("Settings")
    visits_text = st.text_input("Visits (comma-separated)", ",".join(DEFAULT_VISITS))
    VISITS = [v.strip() for v in visits_text.split(",") if v.strip()]
    st.caption("E.g., Day28, Day56, Day84")

# ---------- Helpers ----------
def norm(s): return str(s).strip().lower()

def infer_col(df, must_have=(), any_of=()):
    for c in df.columns:
        lc = norm(c)
        if all(m in lc for m in must_have) and (not any_of or any(a in lc for a in any_of)):
            return c
    return None

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m, n = len(a), len(b)
    if m == 0 or n == 0: return np.nan
    wins = 0
    for x in a:
        wins += np.sum(x > b) - np.sum(x < b)
    return wins / (m * n)

def hodges_lehmann(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    diffs = a.reshape(-1,1) - b.reshape(1,-1)
    return float(np.median(diffs))

# ---------- Builders ----------
def build_long_from_input(df, visits):
    df = df.rename(columns=lambda x: str(x).strip())
    subj = (infer_col(df, ("subject",)) or infer_col(df, ("participant","id")) or
            infer_col(df, ("subject","id")) or infer_col(df, ("id",)) or "Subject")
    grp  = infer_col(df, ("group",)) or "Group"
    base = (infer_col(df, ("baseline",)) or infer_col(df, ("base",)) or
            infer_col(df, ("bl_",)) or infer_col(df, ("bl ",)) or
            infer_col(df, (" bl",)) or infer_col(df, ("bl",)))
    if not base:
        return None, None, None, "Baseline column not found (baseline/base/BL_)."

    visit_map = {}
    for v in visits:
        aliases = [v.lower(), v.lower().replace("day","day ")]
        col = None
        for c in df.columns:
            lc = norm(c)
            if any(a in lc for a in aliases):
                col = c; break
        if not col:
            return None, None, None, f"Raw column for {v} not found."
        visit_map[v] = col

    inp = df[[subj, grp, base] + list(visit_map.values())].copy()
    inp.columns = ["Subject","Group","Baseline"] + visits

    # normalize groups (trim/case only; keep original names like Placebo/USPlus/Permixon)
    inp["Group"] = inp["Group"].astype(str).str.strip()

    # force numeric
    for c in ["Baseline"] + visits:
        inp[c] = pd.to_numeric(inp[c], errors="coerce")

    # CHANGE_WIDE
    chg_wide = inp[["Subject","Group"]].copy()
    for v in visits:
        chg_wide[f"{v}_change"] = inp[v] - inp["Baseline"]

    # long
    rows = []
    for _, r in chg_wide.iterrows():
        for v in visits:
            val = r.get(f"{v}_change")
            if pd.notna(val):
                rows.append({"Subject": r["Subject"], "Group": r["Group"], "Time": v, "Change": float(val)})
    long_change = pd.DataFrame(rows)
    if long_change.empty:
        return None, None, None, "No change values computed (non-numeric inputs?)."
    long_change["Time"] = pd.Categorical(long_change["Time"], categories=visits, ordered=True)
    return long_change, chg_wide, inp, None

def build_long_from_changewide(df, visits):
    df = df.rename(columns=lambda x: str(x).strip())
    subj = (infer_col(df, ("subject",)) or infer_col(df, ("participant","id")) or
            infer_col(df, ("subject","id")) or infer_col(df, ("id",)) or "Subject")
    grp  = infer_col(df, ("group",)) or "Group"
    change_words = ("change","delta","diff")

    vmap = {}
    for v in visits:
        aliases = [v.lower(), v.lower().replace("day","day ")]
        col = None
        for c in df.columns:
            lc = norm(c)
            if any(a in lc for a in aliases) and any(w in lc for w in change_words):
                col = c; break
        if not col:
            exact = f"{v.lower()}_change"
            for c in df.columns:
                if norm(c) == exact:
                    col = c; break
        if not col: return None, None, f"Change column for {v} not found."
        vmap[v] = col

    use = df[[subj, grp] + list(vmap.values())].copy()
    use.columns = ["Subject","Group"] + visits
    use["Group"] = use["Group"].astype(str).str.strip()
    for v in visits:
        use[v] = pd.to_numeric(use[v], errors="coerce")

    rows = []
    for _, r in use.iterrows():
        for v in visits:
            val = r[v]
            if pd.notna(val):
                rows.append({"Subject": r["Subject"], "Group": r["Group"], "Time": v, "Change": float(val)})
    long_change = pd.DataFrame(rows)
    if long_change.empty:
        return None, None, "No change values found."
    long_change["Time"] = pd.Categorical(long_change["Time"], categories=visits, ordered=True)
    return long_change, use, None

# ---------- Core stats ----------
def compute_rm_all_groups(long_change, visits):
    """RM across ALL groups (Group, Time, Interaction). Drops visits lacking any group data."""
    lc = long_change.copy()
    lc["Change"] = pd.to_numeric(lc["Change"], errors="coerce")
    lc = lc.dropna(subset=["Group","Time","Change"]).copy()

    # Keep visits that actually appear
    observed = [v for v in visits if v in lc["Time"].astype(str).unique().tolist()]
    # Need at least 2 time levels overall
    if len(observed) < 2:
        return pd.DataFrame({
            "Effect": ["Group (all groups)", f"Time ({', '.join(observed) if observed else '—'})", "Group × Time"],
            "p-value": [np.nan, np.nan, np.nan]
        }), observed

    lc["rank_change"] = lc["Change"].rank(method="average")
    model = ols("rank_change ~ C(Subject) + C(Group) * C(Time)", data=lc).fit()
    an = anova_lm(model, typ=3)

    rm_df = pd.DataFrame({
        "Effect": ["Group (all groups)", f"Time ({', '.join(observed)})", "Group × Time"],
        "p-value": [float(an.loc["C(Group)","PR(>F)"]),
                    float(an.loc["C(Time)","PR(>F)"]),
                    float(an.loc["C(Group):C(Time)","PR(>F)"])]
    })
    return rm_df, observed

def compute_pairwise_between(long_change, visits, group_a, group_b):
    """Between-group tests (MWU, BM, Welch) for a selected pair, per visit, and Summary for chart."""
    lc = long_change[long_change["Group"].isin([group_a, group_b])].copy()
    lc["Change"] = pd.to_numeric(lc["Change"], errors="coerce")
    lc = lc.dropna(subset=["Change","Time","Group"]).copy()

    # valid visits require data in both groups
    valid = []
    diag = []
    for v in [vv for vv in visits if vv in lc["Time"].astype(str).unique().tolist()]:
        nA = lc[(lc["Time"]==v) & (lc["Group"]==group_a)]["Change"].notna().sum()
        nB = lc[(lc["Time"]==v) & (lc["Group"]==group_b)]["Change"].notna().sum()
        if nA>0 and nB>0: valid.append(v)
        diag.append({"Visit": v, f"N {group_a}": int(nA), f"N {group_b}": int(nB),
                     "Status": "kept" if (nA>0 and nB>0) else "dropped: need both groups"})

    rows = []
    sumrows = []
    for v in valid:
        a = lc[(lc["Time"]==v) & (lc["Group"]==group_a)]["Change"]
        b = lc[(lc["Time"]==v) & (lc["Group"]==group_b)]["Change"]

        U = mw_p = bm_stat = bm_p = np.nan
        if len(a)>0 and len(b)>0:
            U, mw_p = mannwhitneyu(a, b, alternative="two-sided")
            if len(a)>3 and len(b)>3:
                bm_stat, bm_p = brunnermunzel(a, b, alternative="two-sided")

        d = cliffs_delta(a, b) if len(a)>0 and len(b)>0 else np.nan
        hl = hodges_lehmann(a, b) if len(a)>0 and len(b)>0 else np.nan

        r_rb = np.nan
        if len(a)>0 and len(b)>0 and not np.isnan(U):
            r_rb = 1 - 2 * U / (len(a)*len(b))

        rows.append({
            "Visit": v,
            f"N {group_a}": int(len(a)),
            f"N {group_b}": int(len(b)),
            f"{group_a} median Δ": float(np.median(a)) if len(a)>0 else np.nan,
            f"{group_b} median Δ": float(np.median(b)) if len(b)>0 else np.nan,
            "Mann–Whitney U": float(U) if U==U else np.nan,
            "Mann–Whitney p": float(mw_p) if mw_p==mw_p else np.nan,
            "Brunner–Munzel stat": float(bm_stat) if bm_stat==bm_stat else np.nan,
            "Brunner–Munzel p": float(bm_p) if bm_p==bm_p else np.nan,
            "Cliff's delta": float(d) if d==d else np.nan,
            "Hodges–Lehmann (Δ A−B)": float(hl) if hl==hl else np.nan,
            "Rank-biserial r": float(r_rb) if r_rb==r_rb else np.nan,
        })

        def qstats(x):
            if len(x)==0: return (np.nan, np.nan, np.nan, np.nan, np.nan)
            med = np.median(x); q1 = np.percentile(x,25); q3 = np.percentile(x,75)
            return med, q1, q3, (q3-med), (med-q1)

        amed, aq1, aq3, aplus, aminus = qstats(a)
        bmed, bq1, bq3, bplus, bminus = qstats(b)
        sumrows.append({
            "Time": v,
            f"{group_a}_median": amed, f"{group_a}_Q1": aq1, f"{group_a}_Q3": aq3,
            f"{group_b}_median": bmed, f"{group_b}_Q1": bq1, f"{group_b}_Q3": bq3,
            f"{group_a}_plus": aplus, f"{group_a}_minus": aminus,
            f"{group_b}_plus": bplus, f"{group_b}_minus": bminus
        })

    pv_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(sumrows)
    diag_df = pd.DataFrame(diag)
    return pv_df, summary_df, valid, diag_df

def compute_within_group_tests(input_echo, visits):
    """Within-group (Baseline vs each visit) for ALL groups present."""
    if input_echo is None or input_echo.empty:
        return pd.DataFrame()
    groups = sorted(input_echo["Group"].dropna().astype(str).str.strip().unique().tolist())
    rows = []
    for grp in groups:
        g = input_echo[input_echo["Group"] == grp]
        base = pd.to_numeric(g["Baseline"], errors="coerce")
        for v in visits:
            if v not in g.columns: 
                rows.append({"Group": grp, "Visit": v, "N (paired)": 0,
                             "Wilcoxon stat": np.nan, "Wilcoxon p": np.nan,
                             "Paired t stat": np.nan, "Paired t p": np.nan})
                continue
            follow = pd.to_numeric(g[v], errors="coerce")
            paired = pd.concat([base, follow], axis=1).dropna()
            if paired.shape[0] == 0:
                w_stat = w_p = t_stat = t_p = np.nan
            else:
                try:
                    w_stat, w_p = wilcoxon(paired.iloc[:,0], paired.iloc[:,1],
                                           alternative="two-sided", zero_method="wilcox")
                except Exception:
                    w_stat, w_p = (np.nan, np.nan)
                try:
                    t_stat, t_p = ttest_rel(paired.iloc[:,0], paired.iloc[:,1])
                except Exception:
                    t_stat, t_p = (np.nan, np.nan)
            rows.append({
                "Group": grp, "Visit": v, "N (paired)": int(paired.shape[0]),
                "Wilcoxon stat": float(w_stat) if w_stat==w_stat else np.nan,
                "Wilcoxon p": float(w_p) if w_p==w_p else np.nan,
                "Paired t stat": float(t_stat) if t_stat==t_stat else np.nan,
                "Paired t p": float(t_p) if t_p==t_p else np.nan,
            })
    return pd.DataFrame(rows)

def compute_welch_on_changes(long_change, visits, group_a, group_b):
    rows = []
    lc = long_change[long_change["Group"].isin([group_a, group_b])].copy()
    for v in visits:
        a = lc[(lc["Time"]==v) & (lc["Group"]==group_a)]["Change"].dropna()
        b = lc[(lc["Time"]==v) & (lc["Group"]==group_b)]["Change"].dropna()
        if len(a)>1 and len(b)>1:
            t_stat, t_p = ttest_ind(a, b, equal_var=False)
        else:
            t_stat, t_p = (np.nan, np.nan)
        rows.append({"Visit": v, f"N {group_a}": int(len(a)), f"N {group_b}": int(len(b)),
                     "Welch t stat": float(t_stat) if t_stat==t_stat else np.nan,
                     "Welch t p": float(t_p) if t_p==t_p else np.nan})
    return pd.DataFrame(rows)

def make_chart(summary_df, visits, group_a, group_b):
    # guard
    needed = {f"{group_a}_median", f"{group_a}_minus", f"{group_a}_plus",
              f"{group_b}_median", f"{group_b}_minus", f"{group_b}_plus", "Time"}
    if summary_df is None or summary_df.empty or not needed.issubset(set(summary_df.columns)):
        return None
    df = summary_df.copy().set_index("Time").reindex(visits).reset_index()

    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(df))

    a_med = df[f"{group_a}_median"].to_numpy()
    a_lo  = a_med - df[f"{group_a}_minus"].to_numpy()
    a_hi  = a_med + df[f"{group_a}_plus"].to_numpy()
    b_med = df[f"{group_b}_median"].to_numpy()
    b_lo  = b_med - df[f"{group_b}_minus"].to_numpy()
    b_hi  = b_med + df[f"{group_b}_plus"].to_numpy()

    ax.plot(x, a_med, marker="o", label=f"{group_a} (median Δ)")
    ax.fill_between(x, a_lo, a_hi, alpha=0.3, label=f"{group_a} IQR")
    ax.plot(x, b_med, marker="o", label=f"{group_b} (median Δ)")
    ax.fill_between(x, b_lo, b_hi, alpha=0.3, label=f"{group_b} IQR")

    ax.set_xticks(x, visits); ax.set_xlabel("Visit"); ax.set_ylabel("Change from Baseline")
    ax.set_title(f"Median Change with IQR — {group_a} vs {group_b}")
    ax.legend(); fig.tight_layout()

    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150); buf.seek(0)
    return buf

# ---------- UI ----------
st.subheader("1) Upload your Excel (.xlsx)")
uploaded = st.file_uploader(
    "Upload a workbook that contains either an 'Input' sheet (raw) or a 'Change Wide' sheet (changes).",
    type=["xlsx"]
)

if uploaded:
    try:
        xl = pd.ExcelFile(uploaded)
    except Exception as e:
        st.error(f"Could not read Excel: {e}"); st.stop()

    long_change = None; change_wide = None; input_echo = None
    used_mode = None; used_sheet = None
    RAW_DF = None  # <---- NEW holder for the detected raw sheet

    # Try Change Wide
    for s in xl.sheet_names:
        try:
            df = pd.read_excel(uploaded, sheet_name=s)
            lc, chg_wide, err = build_long_from_changewide(df, VISITS)
            if lc is not None and err is None and not lc.empty:
                long_change = lc; change_wide = chg_wide; used_mode, used_sheet = "CHANGE", s
                RAW_DF = df.copy()
                break
        except Exception: pass

    # Try Input
    if long_change is None:
        for s in xl.sheet_names:
            try:
                df = pd.read_excel(uploaded, sheet_name=s)
                lc, chg_wide, inp, err = build_long_from_input(df, VISITS)
                if lc is not None and err is None and not lc.empty:
                    long_change = lc; change_wide = chg_wide; input_echo = inp; used_mode, used_sheet = "RAW", s
                    RAW_DF = df.copy()
                    break
            except Exception: pass

    if long_change is None or long_change.empty:
        st.error("Could not auto-detect a usable sheet."); st.stop()

    st.success(f"Detected: **{used_mode}** on sheet **{used_sheet}**")

    # ======================================================================
    # NEW: Analysis Set & Filters (BMI, Age, Weight, ITT, PP, Completers, Status/Subject exclusions)
    # ======================================================================
    with st.sidebar:
        st.markdown("### Analysis Set & Filters")
        # Analysis-set toggles
        use_itt = st.checkbox("ITT / mITT (default include all)", value=True)
        use_pp = st.checkbox("PP (Status == ACTIVE)", value=False)
        use_completers = st.checkbox("Completers (has Day84)", value=False)
        excl_dae = st.checkbox("Exclude 'Dropped AE'", value=False)

        st.markdown("---")
        st.caption("Numeric filters (apply only if boxes are ticked)")

        # Try to detect meta columns on RAW_DF
        age_col    = infer_col(RAW_DF, ("age",))
        weight_col = infer_col(RAW_DF, ("weight",)) or infer_col(RAW_DF, ("wt",)) or infer_col(RAW_DF, ("kg",))
        bmi_col    = infer_col(RAW_DF, ("bmi",))
        status_col = infer_col(RAW_DF, ("status",))

        # Age filter
        use_age = False
        if age_col:
            a_vals = pd.to_numeric(RAW_DF[age_col], errors="coerce")
            if a_vals.notna().any():
                a_min, a_max = float(np.nanmin(a_vals)), float(np.nanmax(a_vals))
                use_age = st.checkbox(f"Filter by Age ({age_col})", value=False)
                if use_age:
                    age_range = st.slider("Age range", min_value=int(np.floor(a_min)),
                                          max_value=int(np.ceil(a_max)),
                                          value=(int(np.floor(a_min)), int(np.ceil(a_max))))

        # Weight filter
        use_wt = False
        if weight_col:
            w_vals = pd.to_numeric(RAW_DF[weight_col], errors="coerce")
            if w_vals.notna().any():
                w_min, w_max = float(np.nanmin(w_vals)), float(np.nanmax(w_vals))
                use_wt = st.checkbox(f"Filter by Weight ({weight_col})", value=False)
                if use_wt:
                    weight_range = st.slider("Weight (kg)", min_value=float(np.floor(w_min)),
                                             max_value=float(np.ceil(w_max)),
                                             value=(float(np.floor(w_min)), float(np.ceil(w_max))), step=0.5)

        # BMI filter
        use_bmi = False
        if bmi_col:
            b_vals = pd.to_numeric(RAW_DF[bmi_col], errors="coerce")
            if b_vals.notna().any():
                b_min, b_max = float(np.nanmin(b_vals)), float(np.nanmax(b_vals))
                use_bmi = st.checkbox(f"Filter by BMI ({bmi_col})", value=False)
                if use_bmi:
                    bmi_range = st.slider("BMI", min_value=float(np.floor(b_min)),
                                          max_value=float(np.ceil(b_max)),
                                          value=(float(np.floor(b_min)), float(np.ceil(b_max))), step=0.1)

        st.markdown("---")
        # Optional status-based exclusions
        custom_excl_status = []
        if status_col:
            status_vals = (RAW_DF[status_col].astype(str).str.strip()
                           .replace({"": np.nan}).dropna().unique().tolist())
            status_vals = sorted(status_vals)
            if status_vals:
                custom_excl_status = st.multiselect("Exclude Status values", status_vals, default=[])

        # Optional subject ID exclusions (comma-separated)
        manual_excl_subjects = st.text_input("Exclude Subject IDs (comma-separated)", value="").strip()

    # Build subject-level meta (Subject, Status, Age, Weight, BMI)
    subj_col = infer_col(RAW_DF, ("subject",)) or infer_col(RAW_DF, ("participant","id")) or infer_col(RAW_DF, ("id",))
    if not subj_col:
        subj_col = "Subject"

    subject_meta = pd.DataFrame({"Subject": RAW_DF[subj_col].astype(str)})
    if status_col: subject_meta["Status"] = RAW_DF[status_col].astype(str)
    if age_col:    subject_meta["Age"]    = pd.to_numeric(RAW_DF[age_col], errors="coerce")
    if weight_col: subject_meta["Weight"] = pd.to_numeric(RAW_DF[weight_col], errors="coerce")
    if bmi_col:    subject_meta["BMI"]    = pd.to_numeric(RAW_DF[bmi_col], errors="coerce")

    # Completers logic (has Day84 in input_echo or Day84_change in change_wide)
    complete_subjects = set()
    if input_echo is not None and "Day84" in input_echo.columns:
        complete_subjects = set(input_echo[input_echo["Day84"].notna()]["Subject"].astype(str))
    elif change_wide is not None:
        day84_change_col = None
        for c in change_wide.columns:
            if "day84" in str(c).lower() and "change" in str(c).lower():
                day84_change_col = c; break
        if day84_change_col:
            complete_subjects = set(change_wide[change_wide[day84_change_col].notna()]["Subject"].astype(str))

    # Build keep mask
    keep_mask = pd.Series(True, index=subject_meta["Subject"])

    if use_pp and "Status" in subject_meta.columns:
        keep_mask &= (subject_meta["Status"].str.upper().str.strip() == "ACTIVE")

    if use_completers and len(complete_subjects) > 0:
        keep_mask &= subject_meta["Subject"].isin(complete_subjects)

    if excl_dae and "Status" in subject_meta.columns:
        keep_mask &= ~subject_meta["Status"].str.contains("Dropped AE", case=False, na=False)

    # Custom status exclusions
    if custom_excl_status and "Status" in subject_meta.columns:
        keep_mask &= ~subject_meta["Status"].isin(set(custom_excl_status))

    # Numeric filters
    if 'use_age' in locals() and use_age and "Age" in subject_meta.columns:
        keep_mask &= subject_meta["Age"].between(age_range[0], age_range[1], inclusive="both")
    if 'use_wt' in locals() and use_wt and "Weight" in subject_meta.columns:
        keep_mask &= subject_meta["Weight"].between(weight_range[0], weight_range[1], inclusive="both")
    if 'use_bmi' in locals() and use_bmi and "BMI" in subject_meta.columns:
        keep_mask &= subject_meta["BMI"].between(bmi_range[0], bmi_range[1], inclusive="both")

    # Manual subject exclusions
    if manual_excl_subjects:
        excl_list = [s.strip() for s in manual_excl_subjects.split(",") if s.strip()]
        if excl_list:
            keep_mask &= ~subject_meta["Subject"].isin(excl_list)

    keep_ids = set(subject_meta.loc[keep_mask, "Subject"].astype(str))

    # Apply subject filter to analysis tables
    def _apply_subject_filter(df, subject_col="Subject"):
        if df is None or df.empty or subject_col not in df.columns:
            return df
        return df[df[subject_col].astype(str).isin(keep_ids)].copy()

    long_change = _apply_subject_filter(long_change, "Subject")
    change_wide = _apply_subject_filter(change_wide, "Subject")
    if input_echo is not None:
        input_echo = _apply_subject_filter(input_echo, "Subject")

    st.info(f"Filters applied — kept **{len(keep_ids)}** subjects.")

    # Recompute available groups AFTER filtering
    all_groups = sorted(long_change["Group"].dropna().astype(str).str.strip().unique().tolist())

    # Pairwise selection UI (unchanged logic, but uses filtered groups)
    with st.sidebar:
        st.markdown("### Pairwise comparison")
        if len(all_groups) < 2:
            st.error("Need at least two groups in data after filtering."); st.stop()
        default_pair = (all_groups[0], all_groups[1]) if len(all_groups) >= 2 else (all_groups[0], all_groups[0])
        group_a = st.selectbox("Group A", all_groups, index=all_groups.index(default_pair[0]))
        group_b = st.selectbox("Group B", all_groups, index=all_groups.index(default_pair[1]))
        if group_a == group_b:
            st.warning("Pick two different groups for pairwise tests.")

    st.subheader("2) Preview (long-format change)")
    st.dataframe(long_change.head(10))

    # RM across ALL groups
    st.subheader("3) Repeated-measures across ALL groups")
    rm_df, observed = compute_rm_all_groups(long_change, VISITS)
    st.dataframe(rm_df)

    # Pairwise between-group tests
    st.subheader(f"4) Pairwise between-group tests — {group_a} vs {group_b}")
    pv_df, summary_df, valid_visits, diag_df = compute_pairwise_between(long_change, VISITS, group_a, group_b)
    st.markdown("**P_VALUES (per visit)**")
    st.dataframe(pv_df)

    with st.expander("Diagnostics (per-visit Ns for selected pair)"):
        st.dataframe(diag_df)

    # Final-visit MW added to RM table (primary endpoint for selected pair)
    if len(valid_visits) > 0:
        final_visit = valid_visits[-1]
        a_final = long_change[(long_change["Group"]==group_a) & (long_change["Time"]==final_visit)]["Change"].dropna()
        b_final = long_change[(long_change["Group"]==group_b) & (long_change["Time"]==final_visit)]["Change"].dropna()
        if len(a_final)>0 and len(b_final)>0:
            _, mw_p_final = mannwhitneyu(a_final, b_final, alternative="two-sided")
            rm_df = pd.concat([rm_df, pd.DataFrame({
                "Effect": [f"Final visit Mann–Whitney ({group_a} vs {group_b}) at {final_visit}"],
                "p-value": [float(mw_p_final)]
            })], ignore_index=True)
            st.markdown("**RM_NONPARAM_SUMMARY (with final-visit MW for selected pair)**")
            st.dataframe(rm_df)
        else:
            st.info("Final-visit Mann–Whitney not added: one of the groups has no data at the final visit.")
    else:
        st.info("No valid visits with both groups for pairwise tests; RM summary shown above for all groups.")

    # Within-group for ALL groups
    st.subheader("5) Within-group tests (Baseline → each visit, all groups)")
    within_df = compute_within_group_tests(input_echo if input_echo is not None else pd.DataFrame(), observed or VISITS)
    st.dataframe(within_df)

    # Welch t (pairwise on selected pair)
    st.subheader(f"6) Welch t-tests on change (pairwise: {group_a} vs {group_b})")
    ttests_df = compute_welch_on_changes(long_change, valid_visits or observed or VISITS, group_a, group_b)
    st.dataframe(ttests_df)

    # Chart for selected pair
    st.subheader("7) Chart (selected pair)")
    chart_buf = make_chart(summary_df, valid_visits or observed, group_a, group_b)
    if chart_buf is None:
        st.info("Chart skipped: not enough data for selected pair.")
    else:
        st.image(chart_buf, caption=f"Median Δ with IQR — {group_a} vs {group_b}", use_column_width=True)

    # Build Excel
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        if input_echo is not None: input_echo.to_excel(writer, sheet_name="INPUT_ECHO", index=False)
        change_wide.to_excel(writer, sheet_name="CHANGE_WIDE", index=False)
        if summary_df is not None and not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="SUMMARY_PAIR", index=False)
        pv_df.to_excel(writer, sheet_name="P_VALUES_PAIR", index=False)
        rm_df.to_excel(writer, sheet_name="RM_NONPARAM_SUMMARY", index=False)
        within_df.to_excel(writer, sheet_name="WITHIN_GROUP_TESTS", index=False)
        ttests_df.to_excel(writer, sheet_name="TTESTS_PAIR", index=False)
        # chart
        workbook = writer.book
        ws_chart = workbook.add_worksheet("CHARTS")
        writer.sheets["CHARTS"] = ws_chart
        if chart_buf is not None:
            ws_chart.insert_image("B2", "chart.png", {"image_data": chart_buf})
        else:
            ws_chart.write("B2", "Chart unavailable for selected pair.")

    out.seek(0)
    base = uploaded.name.rsplit(".",1)[0]
    st.download_button(
        "⬇️ Download Excel (INPUT/CHANGE/SUMMARY_PAIR/P_VALUES_PAIR/RM/WITHIN/TTESTS_PAIR/CHARTS)",
        data=out,
        file_name=f"{base}_RESULTS_PAIRWISE_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

