#!/usr/bin/env python3
"""
=============================================================================
Grad-CAM ROI Statistical Analysis Pipeline
=============================================================================
Performs comprehensive statistical analysis addressing reviewer comments:
  1. Descriptive statistics: per-model mean ± SEM for each ROI
  2. Model comparison: ANOVA + planned contrasts + post-hoc tests
  3. Effect sizes: Cohen's d, Hedges' g, partial η²
  4. Enhanced publication-ready visualizations and tables

Output:
  - Statistical_Analysis/
    ├── Descriptive_Stats.csv          # Per-model × condition mean ± SEM
    ├── ANOVA_Results.csv              # Full ANOVA table
    ├── Pairwise_Comparisons.csv       # Post-hoc with effect sizes
    ├── Effect_Sizes.csv               # Cohen's d / partial η² summary
    ├── Key_Comparisons_Table.csv      # Publication-ready key findings
    ├── Figures/
    │   ├── Fig1_DiagnosticRatio_Forest.png
    │   ├── Fig2_EffectSize_Heatmap.png
    │   ├── Fig3_ROI_Profile_ByGroup.png
    │   ├── Fig4_Condition_Interaction.png
    │   └── Fig5_Enhanced_Summary.png
    └── Stats_Report.txt               # Human-readable statistical report
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ============================================================================
# 0. Configuration
# ============================================================================
RESULTS_CSV = './Results/GradCAM_Results.csv'
OUTPUT_DIR = './Statistical_Analysis'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'Figures')

# ROI definitions (from README)
ROIS = {
    'Eyes':  {'x': (52, 171), 'y': (70, 99),   'color': '#00BCD4'},
    'Nose':  {'x': (52, 171), 'y': (120, 149), 'color': '#FFC107'},
    'Mouth': {'x': (52, 171), 'y': (169, 199), 'color': '#F44336'}
}
ROI_COLORS = ['#00BCD4', '#FFC107', '#F44336']  # cyan, gold, red
ROI_NAMES = ['Eyes_PoS', 'Nose_PoS', 'Mouth_PoS']

# Model grouping
INSERTION_ORDER = ['L1', 'L2', 'L3', 'L4', 'FC-L1', 'FC-L2', 'FC-L3']
SE_TYPE_ORDER = ['Baseline'] + INSERTION_ORDER
CONDITION_ORDER = ['Full', 'E', 'M', 'N']
CONDITION_LABELS = {'Full': 'Full Face', 'E': 'Eyes Masked',
                     'M': 'Mouth Masked', 'N': 'Nose Masked'}
REDUCTION_ORDER = [2, 4, 8, 16, 32]

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# 1. Data Loading & Preprocessing
# ============================================================================
print("=" * 70)
print("1. Loading and preprocessing data...")
print("=" * 70)

df = pd.read_csv(RESULTS_CSV)
print(f"   Loaded {len(df)} rows, {df['ModelFile'].nunique()} models")

# --- Parse model metadata ---
def parse_model(model_name):
    """Extract structured metadata from model filename."""
    info = {'SE_Type': 'Baseline', 'BaseType': 'Unknown', 'Reduction': None,
            'Arch': 'AlexNet', 'InsertionLayer': None}

    if 'VGG' in model_name:
        info['Arch'] = 'VGG16'

    if 'FaceBased' in model_name:
        info['BaseType'] = 'FaceBased'
    elif 'ObjectBased' in model_name:
        info['BaseType'] = 'ObjectBased'

    # SE type
    if 'SE-Conv-L' in model_name:
        info['SE_Type'] = 'SE-Conv'
    elif 'SE-FC-L' in model_name:
        info['SE_Type'] = 'SE-FC'

    # Insertion position
    import re
    pos_match = re.search(r'L(\d+)', model_name)
    if pos_match:
        pos_num = int(pos_match.group(1))
        info['InsertionLayer'] = pos_num
        if info['SE_Type'] == 'SE-FC':
            info['SE_Position_Label'] = f'FC-L{pos_num}'
        else:
            info['SE_Position_Label'] = f'L{pos_num}'
    else:
        info['SE_Position_Label'] = 'Baseline'

    # Reduction ratio
    r_match = re.search(r'_R(\d+)', model_name)
    if r_match:
        info['Reduction'] = int(r_match.group(1))

    # Full model label
    if info['SE_Type'] == 'Baseline':
        info['ModelGroup'] = f"Baseline_{info['Arch']}_{info['BaseType']}"
    else:
        info['ModelGroup'] = f"SE-{'Conv' if info['SE_Type']=='SE-Conv' else 'FC'}-L{info['InsertionLayer']}_{info['BaseType']}"

    return info

# Apply parsing (avoid column collision: drop columns already in df)
model_info = df['ModelFile'].apply(parse_model).apply(pd.Series)
dup_cols = [c for c in model_info.columns if c in df.columns]
model_info = model_info.drop(columns=dup_cols)
df = pd.concat([df, model_info], axis=1)

# --- Compute key metrics ---
# DiagnosticRatio: (Eyes + Mouth) / (Eyes + Nose + Mouth)
# This is the metric used in Figure 5 of the paper
total_roi = df['Eyes_PoS'] + df['Nose_PoS'] + df['Mouth_PoS']
df['DiagnosticRatio'] = np.where(total_roi > 0,
                                  (df['Eyes_PoS'] + df['Mouth_PoS']) / total_roi,
                                  np.nan)

# Also compute raw Eye+Mouth proportion
df['EyesMouth_PoS'] = df['Eyes_PoS'] + df['Mouth_PoS']

# Create a composite model identifier (use apply to avoid column-alignment issues)
def _make_model_id(row):
    r_str = f"_R{int(row['Reduction'])}" if pd.notna(row['Reduction']) else ""
    return f"{row['SE_Position_Label']}_{row['BaseType']}{r_str}"

df['ModelID'] = df.apply(_make_model_id, axis=1)

# Print data overview
print(f"\n   Models by SE_Type:")
print(f"   {df.groupby('SE_Type')['ModelFile'].nunique().to_string()}")
print(f"\n   Models by BaseType:")
print(f"   {df.groupby('BaseType')['ModelFile'].nunique().to_string()}")
print(f"\n   NaN DiagnosticRatio: {df['DiagnosticRatio'].isna().sum()} rows "
      f"(all-zeros ROI → excluded from ratio analyses)")

# ============================================================================
# 2. Descriptive Statistics
# ============================================================================
print("\n" + "=" * 70)
print("2. Computing descriptive statistics...")
print("=" * 70)

# --- Per ModelGroup × Condition descriptive stats ---
desc_cols = ['Eyes_PoS', 'Nose_PoS', 'Mouth_PoS', 'DiagnosticRatio']
group_cols = ['ModelGroup', 'SE_Type', 'BaseType', 'SE_Position_Label',
              'Reduction', 'Condition']

def sem(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

def ci95(x):
    """95% confidence interval half-width"""
    if len(x) <= 1:
        return np.nan
    return stats.t.ppf(0.975, len(x)-1) * np.std(x, ddof=1) / np.sqrt(len(x))

desc_stats = df.groupby(group_cols).agg(
    N=('DiagnosticRatio', 'count'),
    Eyes_Mean=('Eyes_PoS', 'mean'),
    Eyes_SD=('Eyes_PoS', 'std'),
    Eyes_SEM=('Eyes_PoS', sem),
    Eyes_CI95=('Eyes_PoS', ci95),
    Nose_Mean=('Nose_PoS', 'mean'),
    Nose_SD=('Nose_PoS', 'std'),
    Nose_SEM=('Nose_PoS', sem),
    Nose_CI95=('Nose_PoS', ci95),
    Mouth_Mean=('Mouth_PoS', 'mean'),
    Mouth_SD=('Mouth_PoS', 'std'),
    Mouth_SEM=('Mouth_PoS', sem),
    Mouth_CI95=('Mouth_PoS', ci95),
    DiagRatio_Mean=('DiagnosticRatio', 'mean'),
    DiagRatio_SD=('DiagnosticRatio', 'std'),
    DiagRatio_SEM=('DiagnosticRatio', sem),
    DiagRatio_CI95=('DiagnosticRatio', ci95),
).reset_index()

desc_path = os.path.join(OUTPUT_DIR, 'Descriptive_Stats.csv')
desc_stats.to_csv(desc_path, index=False, float_format='%.6f')
print(f"   Saved: {desc_path}")

# --- Per SE_Position_Label × BaseType (collapsed across Reductions & Conditions) ---
# This is the most relevant grouping for the paper's main claim
print("\n   === DiagnosticRatio by SE Position × BaseType (Full Face) ===")
summary_full = df[df['Condition'] == 'Full'].groupby(
    ['SE_Position_Label', 'BaseType']
).agg(
    N=('DiagnosticRatio', 'count'),
    DiagRatio_Mean=('DiagnosticRatio', 'mean'),
    DiagRatio_SD=('DiagnosticRatio', 'std'),
    DiagRatio_SEM=('DiagnosticRatio', sem),
    DiagRatio_CI95=('DiagnosticRatio', ci95),
).reset_index()

# Sort by SE position
summary_full['_sort'] = summary_full['SE_Position_Label'].apply(
    lambda x: SE_TYPE_ORDER.index(x) if x in SE_TYPE_ORDER else 99
)
summary_full = summary_full.sort_values(['_sort', 'BaseType']).drop(columns=['_sort'])
print(summary_full.round(4).to_string())

# ============================================================================
# 3. Statistical Tests
#    Design: 2 (BaseType: FaceBased/ObjectBased) ×
#            3 (SE_Pos: FC-L1/FC-L2/FC-L3) ×
#            5 (Reduction: R2/R4/R8/R16/R32)
#    All analyses restricted to FC-layer insertion models only.
# ============================================================================
print("\n" + "=" * 70)
print("3. Running statistical tests (FC-layer models only)...")
print("=" * 70)

report_lines = []  # Collect for final report

def add_report(text):
    report_lines.append(text)
    print(f"   {text}")

from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# --- Restrict to FC-layer insertion models only ---
FC_POSITIONS = ['FC-L1', 'FC-L2', 'FC-L3']

anova_data = df[(df['Condition'] == 'Full') &
                (df['SE_Position_Label'].isin(FC_POSITIONS))].dropna(
    subset=['DiagnosticRatio']).copy()

anova_data['SE_Pos'] = anova_data['SE_Position_Label'].astype(str)
anova_data['BT'] = anova_data['BaseType']
anova_data['Red'] = anova_data['Reduction'].astype(int).astype(str)

add_report(f"\n   FC-layer data: {len(anova_data)} valid rows")
add_report(f"   Models: {anova_data['ModelFile'].nunique()}")
for pos in FC_POSITIONS:
    for bt in ['FaceBased', 'ObjectBased']:
        n = len(anova_data[(anova_data['SE_Pos'] == pos) & (anova_data['BT'] == bt)])
        add_report(f"     {pos} {bt}: {n} images")

# ------------------------------------------------------------------
# 3a.  Three-way ANOVA: DiagnosticRatio ~ SE_Pos × BaseType × Reduction
# ------------------------------------------------------------------
add_report("\n--- 3a. Three-way ANOVA: DiagnosticRatio ~ SE_Pos × BT × Reduction (Full Face) ---")
add_report("      (Type II SS)")

model = ols('DiagnosticRatio ~ C(SE_Pos) + C(BT) + C(Red) + '
            'C(SE_Pos):C(BT) + C(SE_Pos):C(Red) + C(BT):C(Red) + '
            'C(SE_Pos):C(BT):C(Red)',
            data=anova_data).fit()
anova_table = anova_lm(model, typ=2)

add_report(f"\n{anova_table.to_string()}")

anova_path = os.path.join(OUTPUT_DIR, 'ANOVA_Results.csv')
anova_table.to_csv(anova_path, float_format='%.6f')
add_report(f"\n   Saved: {anova_path}")

# Partial eta-squared
ss_error = anova_table.loc['Residual', 'sum_sq']
for effect in anova_table.index:
    if effect != 'Residual':
        ss_effect = anova_table.loc[effect, 'sum_sq']
        eta2 = ss_effect / (ss_effect + ss_error)
        add_report(f"   Partial η² [{effect}]: {eta2:.4f}")

# ------------------------------------------------------------------
# 3b.  Simple main effects & planned contrasts
#      (a) FaceBased vs ObjectBased at each SE_Pos × Reduction
#      (b) SE_Pos effect within FaceBased at each Reduction
#      Focus especially on R=16 (the paper's key reduction ratio)
# ------------------------------------------------------------------
add_report("\n--- 3b. Planned Contrasts: FaceBased vs ObjectBased ---")
add_report("      Per SE_Position × Reduction, Full Face condition")

contrast_results = []
for se_pos in FC_POSITIONS:
    for red in REDUCTION_ORDER:
        subset = anova_data[(anova_data['SE_Pos'] == se_pos) &
                            (anova_data['Reduction'] == red)]
        fb = subset[subset['BT'] == 'FaceBased']['DiagnosticRatio'].dropna()
        ob = subset[subset['BT'] == 'ObjectBased']['DiagnosticRatio'].dropna()

        if len(fb) < 5 or len(ob) < 5:
            continue

        t_stat, p_val = stats.ttest_ind(fb, ob, equal_var=False)  # Welch
        pooled_sd = np.sqrt((np.var(fb, ddof=1) + np.var(ob, ddof=1)) / 2)
        cohens_d = (np.mean(fb) - np.mean(ob)) / pooled_sd if pooled_sd > 0 else 0
        n1, n2 = len(fb), len(ob)
        hedges_g = cohens_d * (1 - 3 / (4 * (n1 + n2) - 9))

        contrast_results.append({
            'SE_Position': se_pos,
            'Reduction': red,
            'FaceBased_Mean': np.mean(fb),
            'FaceBased_SD': np.std(fb, ddof=1),
            'FaceBased_N': n1,
            'ObjectBased_Mean': np.mean(ob),
            'ObjectBased_SD': np.std(ob, ddof=1),
            'ObjectBased_N': n2,
            'Mean_Diff': np.mean(fb) - np.mean(ob),
            't_statistic': t_stat,
            'p_value': p_val,
            'Cohens_d': cohens_d,
            'Hedges_g': hedges_g,
            'd_interpretation': 'large' if abs(cohens_d) >= 0.8 else
                               'medium' if abs(cohens_d) >= 0.5 else
                               'small' if abs(cohens_d) >= 0.2 else 'negligible'
        })

contrast_df = pd.DataFrame(contrast_results)

# FDR correction across all 15 comparisons (3 positions × 5 reductions)
if len(contrast_df) > 1:
    _, p_fdr, _, _ = multipletests(contrast_df['p_value'], method='fdr_bh')
    contrast_df['p_FDR_corrected'] = p_fdr

cols_to_show = ['SE_Position', 'Reduction', 'FaceBased_Mean', 'ObjectBased_Mean',
                'Mean_Diff', 'Cohens_d', 'd_interpretation', 'p_FDR_corrected']
add_report(f"\n{contrast_df[cols_to_show].round(4).to_string()}")

contrast_path = os.path.join(OUTPUT_DIR, 'Pairwise_Comparisons.csv')
contrast_df.to_csv(contrast_path, index=False, float_format='%.6f')
add_report(f"\n   Saved: {contrast_path}")

# Summarize significant
significant = contrast_df[contrast_df['p_FDR_corrected'] < 0.05]
add_report(f"\n   Significant FB > OB after FDR correction:")
if len(significant) > 0:
    for _, row in significant.iterrows():
        add_report(f"     {row['SE_Position']} R{row['Reduction']}: "
                   f"d={row['Cohens_d']:.3f} ({row['d_interpretation']}), "
                   f"p_FDR={row['p_FDR_corrected']:.4f}")
else:
    add_report("     None reached significance after FDR correction.")

# Also report uncorrected (exploratory)
add_report(f"\n   FB > OB with nominally significant p < 0.05 (uncorrected):")
nominal = contrast_df[contrast_df['p_value'] < 0.05]
if len(nominal) > 0:
    for _, row in nominal.iterrows():
        add_report(f"     {row['SE_Position']} R{row['Reduction']}: "
                   f"d={row['Cohens_d']:.3f}, p_uncorrected={row['p_value']:.4f}")

# ------------------------------------------------------------------
# 3c.  Simple main effect of SE_Pos within FaceBased, at each Reduction
# ------------------------------------------------------------------
add_report("\n--- 3c. SE Position effect within FaceBased (per Reduction, Full Face) ---")

tukey_all = []
for red in REDUCTION_ORDER:
    fb_red = anova_data[(anova_data['BT'] == 'FaceBased') &
                        (anova_data['Reduction'] == red)]
    groups = [g['DiagnosticRatio'].dropna().values
              for _, g in fb_red.groupby('SE_Pos')]

    if len(groups) < 2:
        continue

    f_r, p_r = f_oneway(*groups)
    n_total = sum(len(g) for g in groups)
    add_report(f"   R{red}: F({len(groups)-1}, {n_total-len(groups)}) = {f_r:.4f}, "
               f"p = {p_r:.6f}")

    # Means
    for se_pos in FC_POSITIONS:
        vals = fb_red[fb_red['SE_Pos'] == se_pos]['DiagnosticRatio']
        if len(vals) > 0:
            add_report(f"     {se_pos}: M={np.mean(vals):.4f}, SEM={sem(vals):.4f}, n={len(vals)}")

    # Tukey HSD
    if p_r < 0.05:
        tukey = pairwise_tukeyhsd(fb_red['DiagnosticRatio'].values,
                                   fb_red['SE_Pos'].values, alpha=0.05)
        add_report(f"\n   Tukey HSD R{red}:\n{tukey.summary().as_text()}")
        tukey_df_r = pd.DataFrame(data=tukey.summary().data[1:],
                                  columns=tukey.summary().data[0])
        tukey_df_r['Reduction'] = red
        tukey_all.append(tukey_df_r)

if tukey_all:
    tukey_combined = pd.concat(tukey_all, ignore_index=True)
    tukey_path = os.path.join(OUTPUT_DIR, 'TukeyHSD_FaceBased.csv')
    tukey_combined.to_csv(tukey_path, index=False)
    add_report(f"\n   Saved: {tukey_path}")

# Also: simple main effect of Reduction within FaceBased at each SE_Pos
add_report(f"\n   Reduction effect within FaceBased (per SE Position):")
for se_pos in FC_POSITIONS:
    fb_se = anova_data[(anova_data['BT'] == 'FaceBased') &
                       (anova_data['SE_Pos'] == se_pos)]
    groups_r = [g['DiagnosticRatio'].dropna().values
                for _, g in fb_se.groupby('Reduction')]
    if len(groups_r) >= 2:
        f_r2, p_r2 = f_oneway(*groups_r)
        n_total2 = sum(len(g) for g in groups_r)
        add_report(f"   {se_pos}: F({len(groups_r)-1}, {n_total2-len(groups_r)}) = {f_r2:.4f}, "
                   f"p = {p_r2:.6f}")
        for red in REDUCTION_ORDER:
            vals = fb_se[fb_se['Reduction'] == red]['DiagnosticRatio']
            if len(vals) > 0:
                add_report(f"     R{red}: M={np.mean(vals):.4f}, SEM={sem(vals):.4f}, n={len(vals)}")

# ------------------------------------------------------------------
# 3d.  Condition Effects: Full vs Masked
#      Focal model: FC-L3 FaceBased R=16 (the paper's central claim)
#      NOTE: N=1 model × 21 images; Nose Masked has near-zero DR (1/21 valid).
#      Therefore primary analysis uses per-ROI PoS (always valid).
#      DiagnosticRatio comparisons reported where available (Full vs E, Full vs M).
# ------------------------------------------------------------------
add_report("\n--- 3d. Condition Effects: Full vs Masked ---")
add_report("      Focal model: FC-L3 FaceBased R=16 (deepest FC insertion, key reduction)")
add_report("      N = 1 model × 21 images per condition")

key_model_mask = (
    (df['SE_Position_Label'] == 'FC-L3') &
    (df['BaseType'] == 'FaceBased') &
    (df['Reduction'] == 16)
)
key_data = df[key_model_mask].copy()
key_data = key_data[key_data['Condition'].isin(['Full', 'E', 'M', 'N'])]

# --- Primary: Per-ROI PoS paired t-tests (Full vs E, Full vs M) ---
# PoS is always defined (zero is valid data — model allocates no attention there)
add_report(f"\n   === ROI-level PoS: Full vs Masked (FC-L3 R16 FaceBased) ===")

for mask_cond in ['E', 'M', 'N']:
    add_report(f"\n   Full vs {mask_cond}:")
    for roi in ['Eyes_PoS', 'Nose_PoS', 'Mouth_PoS']:
        roi_pivot = key_data.pivot_table(index=['ImageID'],
                                          columns='Condition', values=roi)
        rc = roi_pivot.dropna(subset=['Full', mask_cond])
        n_roi = len(rc)
        if n_roi < 5:
            add_report(f"     {roi}: insufficient pairs (n={n_roi})")
            continue
        diff_r = rc['Full'].values - rc[mask_cond].values
        t_r, p_r = ttest_rel(rc['Full'].values, rc[mask_cond].values)
        d_r = np.mean(diff_r) / np.std(diff_r, ddof=1)
        add_report(f"     {roi} (n={n_roi}, df={n_roi-1}): "
                   f"Full M={np.mean(rc['Full'].values):.4f} → {mask_cond} M={np.mean(rc[mask_cond].values):.4f}, "
                   f"ΔM={np.mean(diff_r):.4f}, t={t_r:.4f}, p={p_r:.6f}, d={d_r:.4f}")

# --- Secondary: DiagnosticRatio where available ---
add_report(f"\n   === DiagnosticRatio: Full vs E, Full vs M (available pairs) ===")
for mask_cond in ['E', 'M']:
    dr_pivot = key_data.pivot_table(index=['ImageID'],
                                     columns='Condition', values='DiagnosticRatio')
    dr_cc = dr_pivot.dropna(subset=['Full', mask_cond])
    n_dr = len(dr_cc)
    if n_dr >= 5:
        diff_dr = dr_cc['Full'].values - dr_cc[mask_cond].values
        t_dr, p_dr = ttest_rel(dr_cc['Full'].values, dr_cc[mask_cond].values)
        d_dr = np.mean(diff_dr) / np.std(diff_dr, ddof=1)
        add_report(f"   Full vs {mask_cond} (n={n_dr}, df={n_dr-1}): "
                   f"Full M={np.mean(dr_cc['Full'].values):.4f}, "
                   f"{mask_cond} M={np.mean(dr_cc[mask_cond].values):.4f}, "
                   f"t={t_dr:.4f}, p={p_dr:.6f}, d={d_dr:.4f}")
    else:
        add_report(f"   Full vs {mask_cond}: insufficient pairs (n={n_dr})")

add_report(f"\n   Note: FC-L3 R16 Nose Masked has only 1/21 valid DiagnosticRatio")
add_report(f"   (all-zeros CAM → undefined ratio), precluding paired DR analysis.")

# ============================================================================
# 4. Publication-ready Key Comparisons Table
# ============================================================================
print("\n" + "=" * 70)
print("4. Generating publication-ready tables...")
print("=" * 70)

# Comprehensive table: FC-L1/FC-L2/FC-L3 × BaseType × Reduction, Full condition
key_table_rows = []
for se_pos in FC_POSITIONS:
    for bt in ['FaceBased', 'ObjectBased']:
        for red in REDUCTION_ORDER:
            sub = anova_data[(anova_data['SE_Pos'] == se_pos) &
                             (anova_data['BT'] == bt) &
                             (anova_data['Reduction'] == red)]

            if len(sub) == 0:
                continue

            dr_vals = sub['DiagnosticRatio']
            row_dict = {
                'SE_Position': se_pos,
                'Pretraining': bt,
                'Reduction': red,
                'N': len(dr_vals),
                'DiagRatio_Mean': np.mean(dr_vals),
                'DiagRatio_SD': np.std(dr_vals, ddof=1),
                'DiagRatio_SEM': sem(dr_vals),
                'Eyes_Mean': np.mean(sub['Eyes_PoS']),
                'Eyes_SEM': sem(sub['Eyes_PoS']),
                'Nose_Mean': np.mean(sub['Nose_PoS']),
                'Nose_SEM': sem(sub['Nose_PoS']),
                'Mouth_Mean': np.mean(sub['Mouth_PoS']),
                'Mouth_SEM': sem(sub['Mouth_PoS']),
            }
            key_table_rows.append(row_dict)

# Also add aggregated rows (collapsed across reductions) for summary
for se_pos in FC_POSITIONS:
    for bt in ['FaceBased', 'ObjectBased']:
        sub = anova_data[(anova_data['SE_Pos'] == se_pos) &
                         (anova_data['BT'] == bt)]
        if len(sub) == 0:
            continue
        dr_vals = sub['DiagnosticRatio']
        row_dict = {
            'SE_Position': se_pos,
            'Pretraining': bt,
            'Reduction': 'ALL',
            'N': len(dr_vals),
            'DiagRatio_Mean': np.mean(dr_vals),
            'DiagRatio_SD': np.std(dr_vals, ddof=1),
            'DiagRatio_SEM': sem(dr_vals),
            'Eyes_Mean': np.mean(sub['Eyes_PoS']),
            'Eyes_SEM': sem(sub['Eyes_PoS']),
            'Nose_Mean': np.mean(sub['Nose_PoS']),
            'Nose_SEM': sem(sub['Nose_PoS']),
            'Mouth_Mean': np.mean(sub['Mouth_PoS']),
            'Mouth_SEM': sem(sub['Mouth_PoS']),
        }
        key_table_rows.append(row_dict)

key_table = pd.DataFrame(key_table_rows)
key_table_path = os.path.join(OUTPUT_DIR, 'Key_Comparisons_Table.csv')
key_table.to_csv(key_table_path, index=False, float_format='%.6f')
add_report(f"   Saved: {key_table_path}")

# ============================================================================
# 5. Enhanced Visualizations (FC-layer models only, with Reduction)
# ============================================================================
print("\n" + "=" * 70)
print("5. Generating publication-ready figures...")
print("=" * 70)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

fb_color = '#2196F3'   # Blue for FaceBased
ob_color = '#FF9800'   # Orange for ObjectBased

# ------------------------------------------------------------------
# Figure 1: Forest Plot — DiagnosticRatio by SE_Pos × BaseType × Reduction
# ------------------------------------------------------------------
print("   Figure 1: DiagnosticRatio Forest Plot (FC-layer, Full Face)...")

forest_data = anova_data.groupby(['SE_Pos', 'BT', 'Reduction']).agg(
    mean=('DiagnosticRatio', 'mean'),
    sem=('DiagnosticRatio', sem),
    sd=('DiagnosticRatio', 'std'),
    n=('DiagnosticRatio', 'count')
).reset_index()
forest_data['ci95'] = forest_data.apply(
    lambda r: stats.t.ppf(0.975, r['n']-1) * r['sem'] if r['n'] > 1 else np.nan, axis=1
)
# Sort: FC-L1, FC-L2, FC-L3; within each, R2-R32; FaceBased then ObjectBased
forest_data['_sort_pos'] = forest_data['SE_Pos'].map({'FC-L1':0, 'FC-L2':1, 'FC-L3':2})
forest_data = forest_data.sort_values(['_sort_pos', 'Reduction', 'BT'])

fig, ax = plt.subplots(figsize=(14, 9))
y_positions = []
y_labels = []

for i, (_, row) in enumerate(forest_data.iterrows()):
    y = len(forest_data) - i - 1
    y_positions.append(y)
    label = f"{row['SE_Pos']} R{row['Reduction']}"
    y_labels.append(label)
    c = fb_color if row['BT'] == 'FaceBased' else ob_color

    ax.errorbar(row['mean'], y, xerr=row['ci95'], fmt='o', color=c,
                capsize=3, markersize=6, elinewidth=1.2, markeredgewidth=0.5,
                markeredgecolor='white', zorder=3)
    ax.errorbar(row['mean'], y, xerr=row['sem'], fmt='none', color=c,
                capsize=0, elinewidth=3, alpha=0.5, zorder=2)

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=8)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.set_xlabel("Diagnostic Ratio (Eyes+Mouth) / Total ROI Saliency")
ax.set_title("ROI Diagnostic Ratio: FC-Layer SE Insertion × Pretraining × Reduction Ratio\n(Full Face Condition)")

legend_elements = [
    Patch(facecolor=fb_color, label='Face-Based Pretraining'),
    Patch(facecolor=ob_color, label='Object-Based Pretraining'),
    plt.Line2D([0], [0], color='black', lw=1.2, label='95% CI (thin)'),
    plt.Line2D([0], [0], color='black', lw=3, alpha=0.5, label='±1 SEM (thick)'),
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

# Significance markers
for _, row_c in contrast_df.iterrows():
    if row_c['p_FDR_corrected'] < 0.05:
        label = f"{row_c['SE_Position']} R{row_c['Reduction']}"
        if label in y_labels:
            fb_label = label
            ob_label = label  # same label for FB and OB positions
            # Find the two y positions for FB and OB
            fb_y = None; ob_y = None
            for j, (_, fr) in enumerate(forest_data.iterrows()):
                l = f"{fr['SE_Pos']} R{fr['Reduction']}"
                if l == label:
                    y_j = len(forest_data) - j - 1
                    if fr['BT'] == 'FaceBased':
                        fb_y = y_j
                    else:
                        ob_y = y_j
            if fb_y is not None and ob_y is not None:
                y_mid = (fb_y + ob_y) / 2
                x_max = max(
                    forest_data[(forest_data['SE_Pos']==row_c['SE_Position'])&
                                (forest_data['BT']=='FaceBased')&
                                (forest_data['Reduction']==row_c['Reduction'])]['mean'].values[0] +
                    forest_data[(forest_data['SE_Pos']==row_c['SE_Position'])&
                                (forest_data['BT']=='FaceBased')&
                                (forest_data['Reduction']==row_c['Reduction'])]['ci95'].values[0],
                    forest_data[(forest_data['SE_Pos']==row_c['SE_Position'])&
                                (forest_data['BT']=='ObjectBased')&
                                (forest_data['Reduction']==row_c['Reduction'])]['mean'].values[0] +
                    forest_data[(forest_data['SE_Pos']==row_c['SE_Position'])&
                                (forest_data['BT']=='ObjectBased')&
                                (forest_data['Reduction']==row_c['Reduction'])]['ci95'].values[0]
                )
                sig_str = '***' if row_c['p_FDR_corrected'] < 0.001 else \
                          '**' if row_c['p_FDR_corrected'] < 0.01 else '*'
                ax.text(x_max + 0.015, y_mid, sig_str, fontsize=12, fontweight='bold',
                        ha='left', va='center')

ax.set_xlim(left=0.3, right=1.0)
fig.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'Fig1_DiagnosticRatio_Forest.png')
fig.savefig(fig1_path)
plt.close()
print(f"      Saved: {fig1_path}")

# ------------------------------------------------------------------
# Figure 2: Effect Size Heatmap — SE_Pos × Reduction (Full Face)
# ------------------------------------------------------------------
print("   Figure 2: Effect Size Heatmap (FC-layer, SE_Pos × Reduction)...")

# Build matrix from contrast_df
es_matrix = contrast_df.pivot(index='SE_Position', columns='Reduction', values='Cohens_d')
es_matrix = es_matrix.reindex(index=FC_POSITIONS, columns=REDUCTION_ORDER)

# Also build p-value matrix for annotations
p_matrix = contrast_df.pivot(index='SE_Position', columns='Reduction', values='p_value')
p_matrix = p_matrix.reindex(index=FC_POSITIONS, columns=REDUCTION_ORDER)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(es_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1.5, vmax=1.5)

ax.set_xticks(range(len(REDUCTION_ORDER)))
ax.set_xticklabels([f'R{r}' for r in REDUCTION_ORDER])
ax.set_yticks(range(len(FC_POSITIONS)))
ax.set_yticklabels(FC_POSITIONS)

for i in range(len(FC_POSITIONS)):
    for j in range(len(REDUCTION_ORDER)):
        val = es_matrix.iloc[i, j]
        if not np.isnan(val):
            pv = p_matrix.iloc[i, j]
            sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
            text = f'{val:.2f}{sig}'
            color = 'white' if abs(val) > 0.7 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=10, color=color,
                    fontweight='bold' if sig else 'normal')

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Cohen's d (FaceBased − ObjectBased)")
ax.set_title("Effect Size of Pretraining Type\n(FC-Layer SE Models, Full Face)")
fig.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'Fig2_EffectSize_Heatmap.png')
fig.savefig(fig2_path)
plt.close()
print(f"      Saved: {fig2_path}")

# ------------------------------------------------------------------
# Figure 3: ROI Activation Profile — FC-L1/L2/L3 FaceBased vs ObjectBased
# ------------------------------------------------------------------
print("   Figure 3: ROI Profile by SE Position (FC-layer, Full Face)...")

profile_data = anova_data.copy()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, se_pos in enumerate(FC_POSITIONS):
    ax = axes[idx]
    sub = profile_data[profile_data['SE_Pos'] == se_pos]

    x = np.arange(3)  # Eyes, Nose, Mouth
    width = 0.35

    for j, bt in enumerate(['FaceBased', 'ObjectBased']):
        bt_sub = sub[sub['BT'] == bt]
        if len(bt_sub) == 0:
            continue
        means = [bt_sub['Eyes_PoS'].mean(), bt_sub['Nose_PoS'].mean(), bt_sub['Mouth_PoS'].mean()]
        sems = [sem(bt_sub['Eyes_PoS']), sem(bt_sub['Nose_PoS']), sem(bt_sub['Mouth_PoS'])]

        offset = (j - 0.5) * width
        ax.bar(x + offset, means, width, yerr=sems, capsize=4,
               color=fb_color if bt == 'FaceBased' else ob_color,
               alpha=0.85, edgecolor='white', linewidth=0.5, label=bt)

    ax.set_xticks(x)
    ax.set_xticklabels(['Eyes', 'Nose', 'Mouth'])
    ax.set_ylabel('Proportion of Saliency')
    ax.set_title(f'{se_pos}')
    ax.legend(fontsize=8)
    ax.set_ylim(0, None)

fig.suptitle('ROI Saliency Distribution: FC-Layer SE Models\n(Full Face, Collapsed Across Reductions)', fontsize=14, y=1.02)
fig.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, 'Fig3_ROI_Profile_ByGroup.png')
fig.savefig(fig3_path)
plt.close()
print(f"      Saved: {fig3_path}")

# ------------------------------------------------------------------
# Figure 4: Condition × Pretraining Interaction (FC-L3 R16, focal model)
# ------------------------------------------------------------------
print("   Figure 4: Condition Interaction (FC-L3 R16, focal model)...")

fig4_data = df[(df['SE_Position_Label'] == 'FC-L3') &
               (df['Reduction'] == 16)].dropna(
    subset=['DiagnosticRatio']).copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: FC-L3 R16 (the focal model)
for p_idx, (panel_label, filter_pos, filter_red) in enumerate([
    ('(a) FC-L3 R=16 (Focal Model)', 'FC-L3', [16]),
    ('(b) FC-L3 All Reductions Combined', 'FC-L3', REDUCTION_ORDER),
]):
    ax = axes[p_idx]
    sub = df[df['SE_Position_Label'] == filter_pos].dropna(subset=['DiagnosticRatio'])
    if filter_red != REDUCTION_ORDER:
        sub = sub[sub['Reduction'].isin(filter_red)]

    x = np.arange(len(CONDITION_ORDER))
    width = 0.35

    for j, bt in enumerate(['FaceBased', 'ObjectBased']):
        bt_sub = sub[sub['BaseType'] == bt]
        means_list, sems_list = [], []
        for cond in CONDITION_ORDER:
            c_sub = bt_sub[bt_sub['Condition'] == cond]['DiagnosticRatio']
            means_list.append(c_sub.mean() if len(c_sub) > 0 else np.nan)
            sems_list.append(sem(c_sub) if len(c_sub) > 0 else np.nan)

        offset = (j - 0.5) * width
        ax.bar(x + offset, means_list, width, yerr=sems_list, capsize=4,
               color=fb_color if bt == 'FaceBased' else ob_color,
               alpha=0.85, edgecolor='white', linewidth=0.5, label=bt)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in CONDITION_ORDER], fontsize=9)
    ax.set_ylabel('Diagnostic Ratio')
    ax.set_title(panel_label, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

fig.suptitle('Diagnostic Ratio Across Masking Conditions: FC-L3 SE Models\n(Focal: Deepest FC Insertion, FaceBased vs ObjectBased)', fontsize=13, y=1.03)
fig.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, 'Fig4_Condition_Interaction.png')
fig.savefig(fig4_path)
plt.close()
print(f"      Saved: {fig4_path}")

# ------------------------------------------------------------------
# Figure 5: Summary — DiagnosticRatio by SE_Pos × Reduction × Condition
# ------------------------------------------------------------------
print("   Figure 5: Enhanced Summary Chart (FC-layer, all conditions)...")

summary_plot = df[df['SE_Position_Label'].isin(FC_POSITIONS)].dropna(
    subset=['DiagnosticRatio']).groupby(
    ['SE_Position_Label', 'Reduction', 'Condition']
).agg(mean=('DiagnosticRatio', 'mean'), sem=('DiagnosticRatio', sem)).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, se_pos in enumerate(FC_POSITIONS):
    ax = axes[idx]
    sp = summary_plot[summary_plot['SE_Position_Label'] == se_pos]

    x = np.arange(len(REDUCTION_ORDER))
    n_conds = len(CONDITION_ORDER)
    width = 0.8 / n_conds
    cond_colors = ['#4CAF50', '#F44336', '#FF9800', '#9C27B0']

    for j, cond in enumerate(CONDITION_ORDER):
        cond_sub = sp[sp['Condition'] == cond]
        means_arr = []
        sems_arr = []
        for red in REDUCTION_ORDER:
            row = cond_sub[cond_sub['Reduction'] == red]
            if len(row) > 0:
                means_arr.append(row['mean'].values[0])
                sems_arr.append(row['sem'].values[0])
            else:
                means_arr.append(0)
                sems_arr.append(0)

        offset = (j - n_conds/2 + 0.5) * width
        ax.bar(x + offset, means_arr, width, yerr=sems_arr, capsize=2,
               color=cond_colors[j], alpha=0.85, edgecolor='white', linewidth=0.3,
               label=CONDITION_LABELS.get(cond, cond))

    ax.set_xticks(x)
    ax.set_xticklabels([f'R{r}' for r in REDUCTION_ORDER])
    ax.set_xlabel('Reduction Ratio')
    if idx == 0:
        ax.set_ylabel('Diagnostic Ratio')
    ax.set_title(f'SE Position: {se_pos}')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.08))
fig.suptitle('Diagnostic Ratio by SE Position, Reduction Ratio, and Masking Condition\n(FC-Layer SE Models, Face+Object Combined)', fontsize=14)
fig.tight_layout(rect=[0, 0.08, 1, 0.95])
fig5_path = os.path.join(FIGURES_DIR, 'Fig5_Enhanced_Summary.png')
fig.savefig(fig5_path)
plt.close()
print(f"      Saved: {fig5_path}")

# ============================================================================
# 6. Generate Final Statistical Report
# ============================================================================
print("\n" + "=" * 70)
print("6. Generating statistical report...")
print("=" * 70)

report_path = os.path.join(OUTPUT_DIR, 'Stats_Report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("Grad-CAM ROI Statistical Analysis Report\n")
    f.write("=" * 70 + "\n\n")

    f.write("ROI Definitions:\n")
    f.write("  Eyes:  x=[52, 171], y=[70, 99]\n")
    f.write("  Nose:  x=[52, 171], y=[120, 149]\n")
    f.write("  Mouth: x=[52, 171], y=[169, 199]\n\n")

    f.write("Heatmap Normalization: Min-Max normalization (0-1) per image\n")
    f.write("ROI Weight Calculation: PoS = sum(ROI) / sum(entire image)\n")
    f.write("DiagnosticRatio: (Eyes+Mouth) / (Eyes+Nose+Mouth)\n\n")

    f.write("-" * 70 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("-" * 70 + "\n\n")

    # Summarize key results
    if len(contrast_df) > 0:
        f.write("1. FaceBased vs ObjectBased Pretraining (Full Face):\n")
        for _, row in contrast_df.iterrows():
            sig_marker = " ***" if row['p_FDR_corrected'] < 0.001 else \
                         " **" if row['p_FDR_corrected'] < 0.01 else \
                         " *" if row['p_FDR_corrected'] < 0.05 else ""
            f.write(f"   {row['SE_Position']:8s}: Δ={row['Mean_Diff']:+.4f}, "
                    f"d={row['Cohens_d']:+.3f} ({row['d_interpretation']}), "
                    f"p_FDR={row['p_FDR_corrected']:.4f}{sig_marker}\n")

    f.write("\n2. Effect Size Interpretation:\n")
    f.write("   d ≥ 0.8 = large effect   |   d ≥ 0.5 = medium   |   d ≥ 0.2 = small\n\n")

    f.write("=" * 70 + "\n")
    f.write("Full report content above was printed to console during execution.\n")

print(f"   Saved: {report_path}")

# ============================================================================
# 7. Summary
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}/")
print(f"  ├── Descriptive_Stats.csv         — Per-model mean ± SEM")
print(f"  ├── ANOVA_Results.csv             — Two-way ANOVA table")
print(f"  ├── Pairwise_Comparisons.csv      — FaceBased vs ObjectBased (Cohen's d)")
print(f"  ├── TukeyHSD_FaceBased.csv        — Post-hoc within FaceBased")
print(f"  ├── Key_Comparisons_Table.csv     — Publication-ready summary")
print(f"  ├── Stats_Report.txt              — Human-readable report")
print(f"  └── Figures/")
print(f"      ├── Fig1_DiagnosticRatio_Forest.png")
print(f"      ├── Fig2_EffectSize_Heatmap.png")
print(f"      ├── Fig3_ROI_Profile_ByGroup.png")
print(f"      ├── Fig4_Condition_Interaction.png")
print(f"      └── Fig5_Enhanced_Summary.png")
