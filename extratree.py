import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load data
file_path = r'C:\Agam\Work\lanechange\combined_data.csv'
df = pd.read_csv(file_path)



# Feature engineering
df['G_lead'] = df['5_target_lead'] - df['5']
df['G_lag'] = df['5'] - df['5_target_follow']
# 2) Flags for missing target‐lane vehicles
df['no_target_lead']   = (df['5_target_lead']   == -9999)
df['no_target_follow'] = (df['5_target_follow'] == -9999)

# 3) Impute a “large but realistic” gap (99th percentile of observed gaps)
high_gap = df.loc[~df['no_target_lead'], 'G_lead'].quantile(0.99)
high_gap = max(high_gap, df.loc[~df['no_target_follow'], 'G_lag'].quantile(0.99))
df.loc[df['no_target_lead'],   'G_lead'] = high_gap
df.loc[df['no_target_follow'], 'G_lag' ] = high_gap

# 4) Compute relative speeds, zeroing out when no vehicle
#    “22” = v_SV, “22_target_lead” = v_LV_TL, “22_target_follow” = v_FV_TL
df['delta_v_LV_SV'] = (
    df['22_target_lead'].where(~df['no_target_lead'], df['22'])
    - df['22']
)
df['delta_v_SV_FV'] = (
    df['22'].where(~df['no_target_follow'], df['22'])
    - df['22_target_follow'].where(~df['no_target_follow'], df['22'])
)
# and force zero on the flagged rows
df.loc[df['no_target_lead'],   'delta_v_LV_SV'] = 0
df.loc[df['no_target_follow'], 'delta_v_SV_FV'] = 0

# 5) Select features (now including the two new flags)
features = [
    '22',   # v_SV
    '12',   # a_SV
    '22_source_lead',     # v_LV in current lane
    'G_lag',              # lag gap (target lane)
    '22_target_follow',   # v_FV_TL
    '12_target_follow',   # a_FV_TL
    'G_lead',             # lead gap (target lane)
    '22_target_lead',     # v_LV_TL
    '12_target_lead',     # a_LV_TL
    'delta_v_LV_SV',
    'delta_v_SV_FV',
    'Time_lapsed',
    'Previous_decision'
]

X = df[features]
y = df['LC']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance with SMOTE 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Extra Trees Classifier with hyperparameters from the paper
model = ExtraTreesClassifier(
    n_estimators=100,
    criterion='gini',
    max_features='sqrt',
    random_state=42
)

# Train the model
model.fit(X_resampled, y_resampled)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Print metrics
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Extra Trees (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
