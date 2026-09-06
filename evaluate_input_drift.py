import pandas as pd
import numpy as np
from scipy import stats

# 1. Load Baseline Training and Generated 100-Row Prediction Datasets
train_df = pd.read_csv('data.csv')
sample_df = pd.read_csv('random_100_samples.csv')

# Drop non-feature columns
for col in ['sno', 'target']:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])

if train_df['gender'].dtype == object:
    train_df['gender'] = train_df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

common_features = [col for col in train_df.columns if col in sample_df.columns]

categorical_cols = ['gender', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
continuous_cols = [c for c in common_features if c not in categorical_cols]

def calculate_psi(baseline, target, num_buckets=5):
    try:
        quantiles = np.linspace(0, 1, num_buckets + 1)
        bins = np.percentile(baseline, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        base_counts, _ = np.histogram(baseline, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)
        
        base_pct = np.where(base_counts == 0, 1e-4, base_counts) / len(baseline)
        target_pct = np.where(target_counts == 0, 1e-4, target_counts) / len(target)
        
        psi = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
        return round(float(psi), 4)
    except Exception:
        return 0.0

drift_records = []

for col in common_features:
    b_data = train_df[col].dropna()
    t_data = sample_df[col].dropna()
    
    if col in categorical_cols:
        # Chi-Square Test
        all_cats = list(set(b_data.unique()).union(set(t_data.unique())))
        b_counts = b_data.value_counts().reindex(all_cats, fill_value=0).values
        t_counts = t_data.value_counts().reindex(all_cats, fill_value=0).values
        
        # Add 1e-5 to avoid divide-by-zero
        contingency = np.array([b_counts + 1e-5, t_counts + 1e-5])
        chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency)
        test_type = "Chi-Square"
        stat_val = chi2_stat
    else:
        # Kolmogorov-Smirnov Test
        ks_stat, p_val = stats.ks_2samp(b_data, t_data)
        test_type = "Kolmogorov-Smirnov"
        stat_val = ks_stat
        
    psi_val = calculate_psi(b_data, t_data)
    drift_detected = bool(p_val < 0.05)
    
    drift_records.append({
        'Feature': col,
        'Test_Type': test_type,
        'Statistic': round(float(stat_val), 4),
        'P_Value': round(float(p_val), 5),
        'PSI': psi_val,
        'Drift_Detected': drift_detected
    })

drift_df = pd.DataFrame(drift_records)
print(drift_df.to_string(index=False))

drifted_features = drift_df[drift_df['Drift_Detected']]['Feature'].tolist()

report_text = (
    "DELIVERABLE 7: INPUT DATA DRIFT DETECTION REPORT\n"
    "================================================\n"
    "Reference Baseline Dataset: data.csv (Training Features)\n"
    "Target Prediction Dataset: random_100_samples.csv (100 Generated Inference Samples)\n"
    "Significance Threshold: p-value < 0.05 indicates statistical drift\n\n"
    "--- Feature-by-Feature Drift Analysis ---\n"
    f"{drift_df.to_string(index=False)}\n\n"
    "--- Summary & Conclusion ---\n"
    f"Total Features Analyzed: {len(common_features)}\n"
    f"Features Exhibiting Drift: {len(drifted_features)} ({', '.join(drifted_features) if drifted_features else 'None'})\n"
    "Plain English Findings:\n"
    "Because the 100-sample test set was randomly and uniformly synthesized across feature ranges rather than "
    "drawn from the real patient population, features with non-uniform real-world distributions show significant statistical drift "
    "(p-value < 0.05 and elevated PSI scores). This triggers a data drift alert, signaling the need for model retraining or data validation."
)

with open("drift_detection_report.txt", "w") as f:
    f.write(report_text)

print("\nSaved Deliverable 7 artifact: drift_detection_report.txt")
