import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

# 1. Load dataset
data_path = 'data.csv'
df = pd.read_csv(data_path)

# 2. Preprocess dataset
if 'sno' in df.columns:
    df = df.drop(columns=['sno'])

# Encode target ('yes'/'no' -> 1/0 if string)
if df['target'].dtype == object:
    df['target'] = df['target'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

# Encode gender ('male'/'female' -> 1/0 if string)
if 'gender' in df.columns and df['gender'].dtype == object:
    df['gender'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Keep raw age for fairness grouping
raw_age = X['age'].copy()
sensitive_age_group = (raw_age >= 60).map({True: 'Senior (>=60)', False: 'Younger (<60)'})

# Train/Test Split
X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
    X, y, sensitive_age_group, test_size=0.2, random_state=42, stratify=y
)

# Train Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Baseline Accuracy: {acc:.4f}\n")

# ==========================================
# DELIVERABLE 2: SHAP MODEL EXPLAINABILITY
# ==========================================
print("Generating SHAP Explainability results...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle different shap output shapes across versions
if isinstance(shap_values, list):
    mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
elif len(shap_values.shape) == 3:
    mean_abs_shap = np.abs(shap_values[:, :, 1]).mean(axis=0)
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean_Absolute_SHAP': mean_abs_shap
}).sort_values(by='Mean_Absolute_SHAP', ascending=True)

least_features = importance_df.head(3)['Feature'].tolist()
least_vals = importance_df.head(3)['Mean_Absolute_SHAP'].tolist()

summary_text = (
    "DELIVERABLE 2: MODEL EXPLAINABILITY REPORT (SHAP)\n"
    "================================================\n\n"
    f"Model Test Accuracy: {acc:.4f}\n\n"
    "Features Ranked by Impact (Lowest to Highest):\n"
    f"{importance_df.to_string(index=False)}\n\n"
    "Plain English Explanation:\n"
    f"The features that have the least impact on predicting heart disease are: {', '.join(least_features)}.\n"
    f"Their average SHAP contribution scores are {least_vals[0]:.4f}, {least_vals[1]:.4f}, and {least_vals[2]:.4f} respectively. "
    "Because their SHAP values are closest to zero across test samples, fluctuations in these attributes produce "
    "the smallest change in the model's likelihood estimate for heart disease."
)

with open('shap_least_impact_summary.txt', 'w') as f:
    f.write(summary_text)

# Save SHAP feature importance bar plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Mean_Absolute_SHAP'], color='teal')
plt.xlabel('Mean |SHAP value| (Average impact on model output magnitude)')
plt.title('Feature Impact Ranking (SHAP)')
plt.tight_layout()
plt.savefig('shap_feature_importance.png')
plt.close()

# ==========================================
# DELIVERABLE 3: FAIRLEARN FAIRNESS AUDIT
# ==========================================
print("Generating Fairlearn Audit results...")
metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate,
    'true_positive_rate': true_positive_rate,
    'false_positive_rate': false_positive_rate
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=age_test
)

fairness_text = (
    "DELIVERABLE 3: FAIRNESS TESTING REPORT (FAIRLEARN)\n"
    "=================================================\n"
    f"Sensitive Attribute: Age (Binarized: Senior >= 60 vs Younger < 60)\n\n"
    "--- Overall Performance ---\n"
    f"{metric_frame.overall.to_string()}\n\n"
    "--- Group-Level Metrics ---\n"
    f"{metric_frame.by_group.to_string()}\n\n"
    "--- Disparity Differences (Group Max - Group Min) ---\n"
    f"{metric_frame.difference().to_string()}\n\n"
    "--- Disparity Ratios (Group Min / Group Max) ---\n"
    f"{metric_frame.ratio().to_string()}\n"
)

with open('fairlearn_metrics_report.txt', 'w') as f:
    f.write(fairness_text)

print("\nFinished successfully! Created the following output files:")
print("1. shap_least_impact_summary.txt")
print("2. shap_feature_importance.png")
print("3. fairlearn_metrics_report.txt")
