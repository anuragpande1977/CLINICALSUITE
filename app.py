st.success(f"Detected: **{used_mode}** on sheet **{used_sheet}**")
```)
2) And **before** you compute `all_groups` and the pairwise selections.

> If you can, also set `RAW_DF = df` at the point where you detect the usable sheet, so we can reference the original sheet for Age/Weight/BMI/Status discovery. (See the first 10 lines of the snippet for how to add that if you don’t already have it.)

---

### Paste this block

```python
# ===== NEW: Capture the raw sheet used (do this where you detect the usable sheet) =====
# If you already have RAW_DF bound to the detected sheet DataFrame, you can skip this.
try:
    RAW_DF  # noqa: F821
except NameError:
    RAW_DF = df  # the 'df' you used to build long_change/change_wide/input_echo

# ===== NEW: Analysis set & filter controls (Sidebar) =====
with st.sidebar:
    st.markdown("### Analysis Set & Filters")
    # Analysis-set toggles
    use_itt = st.checkbox("ITT / mITT (default include all)", value=True)
    use_pp = st.checkbox("PP (Status == ACTIVE)", value=False)
    use_completers = st.checkbox("Completers (has Day84)", value=False)
    excl_dae = st.checkbox("Exclude 'Dropped AE'", value=False)

    st.markdown("---")
    st.caption("Numeric filters (apply only if boxes are ticked)")

    # Try to detect useful columns on RAW_DF
    age_col    = infer_col(RAW_DF, ("age",))
    weight_col = infer_col(RAW_DF, ("weight",)) or infer_col(RAW_DF, ("wt",)) or infer_col(RAW_DF, ("kg",))
    bmi_col    = infer_col(RAW_DF, ("bmi",))
    status_col = infer_col(RAW_DF, ("status",))

    # Age filter
    use_age = False
    if age_col:
        a_vals = pd.to_numeric(RAW_DF[age_col], errors="coerce")
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
        status_vals = (RAW_DF[status_col].astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist())
        status_vals = sorted(status_vals)
        if status_vals:
            custom_excl_status = st.multiselect("Exclude Status values", status_vals, default=[])

    # Optional subject ID exclusions (comma-separated)
    manual_excl_subjects = st.text_input("Exclude Subject IDs (comma-separated)", value="").strip()

# ===== NEW: Build subject-level meta (Subject, Status, Age, Weight, BMI) =====
# Find a Subject column on RAW_DF; otherwise use what's already in long_change/input_echo
subj_col = infer_col(RAW_DF, ("subject",)) or infer_col(RAW_DF, ("id",))
if not subj_col:
    # Fallbacks
    if input_echo is not None and "Subject" in input_echo.columns:
        subj_col = "Subject"
        RAW_DF = RAW_DF.copy()
        RAW_DF[subj_col] = input_echo["Subject"]
    elif "Subject" in long_change.columns:
        subj_col = "Subject"
        RAW_DF = RAW_DF.copy()
        RAW_DF[subj_col] = long_change["Subject"].drop_duplicates().to_list() + [np.nan]  # best-effort

subject_meta = pd.DataFrame({"Subject": RAW_DF[subj_col].astype(str)})

if status_col:
    subject_meta["Status"] = RAW_DF[status_col].astype(str)
if age_col:
    subject_meta["Age"] = pd.to_numeric(RAW_DF[age_col], errors="coerce")
if weight_col:
    subject_meta["Weight"] = pd.to_numeric(RAW_DF[weight_col], errors="coerce")
if bmi_col:
    subject_meta["BMI"] = pd.to_numeric(RAW_DF[bmi_col], errors="coerce")

# ===== NEW: Completers logic (has Day84) =====
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

# ===== NEW: Build keep mask =====
keep_mask = pd.Series(True, index=subject_meta["Subject"])

# ITT/mITT is the default; other toggles refine that:
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
if use_age and "Age" in subject_meta.columns:
    keep_mask &= subject_meta["Age"].between(age_range[0], age_range[1], inclusive="both")
if use_wt and "Weight" in subject_meta.columns:
    keep_mask &= subject_meta["Weight"].between(weight_range[0], weight_range[1], inclusive="both")
if use_bmi and "BMI" in subject_meta.columns:
    keep_mask &= subject_meta["BMI"].between(bmi_range[0], bmi_range[1], inclusive="both")

# Manual subject exclusions
if manual_excl_subjects:
    excl_list = [s.strip() for s in manual_excl_subjects.split(",") if s.strip()]
    if excl_list:
        keep_mask &= ~subject_meta["Subject"].isin(excl_list)

keep_ids = set(subject_meta.loc[keep_mask, "Subject"].astype(str))

# ===== NEW: Apply to analysis tables (in place) =====
def _apply_subject_filter(df, subject_col="Subject"):
    if df is None or df.empty or subject_col not in df.columns:
        return df
    return df[df[subject_col].astype(str).isin(keep_ids)].copy()

# Filter the three working tables your program uses downstream
long_change = _apply_subject_filter(long_change, "Subject")
change_wide = _apply_subject_filter(change_wide, "Subject")
if input_echo is not None:
    input_echo = _apply_subject_filter(input_echo, "Subject")

st.info(f"Filters applied — kept **{len(keep_ids)}** subjects.")

# ===== OPTIONAL: Recompute available groups AFTER filtering =====
all_groups = sorted(long_change["Group"].dropna().astype(str).str.strip().unique().tolist())

