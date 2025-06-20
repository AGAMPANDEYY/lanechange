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
columns_to_check = ['0_source_lead', '0_source_follow', '0_target_lead', '0_target_follow']
#df = df[~df[columns_to_check].isin([-9999]).any(axis=1)].reset_index(drop=True)
#print(f"{len(df)} rows after cleaning")

# Replace -9999 in selected columns with the mean
columns_to_check = ['0_source_lead', '0_source_follow', '0_target_lead', '0_target_follow']

# First identify rows that have -9999 in any of the specified columns
rows_to_process = df[df[columns_to_check].isin([-9999]).any(axis=1)].index

# For each column in the dataframe, replace -9999 with mean for the identified rows
for col in df.columns:
    mean_val = df.loc[df[col] != -9999, col].mean()
    df.loc[rows_to_process, col] = df.loc[rows_to_process, col].replace(-9999, mean_val)

# Feature engineering
df['G_lead'] = df['5_target_lead'] - df['5']
df['G_lag'] = df['5'] - df['5_target_follow']
df['delta_v_LV_SV'] = df['22_target_lead'] - df['22']
df['delta_v_SV_FV'] = df['22'] - df['22_target_follow']

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
    'Previous_decision',
    'Time_lapsed'
]

X = df[features]
y = df['LC']

# Print class distribution
print("\nClass distribution in original data:")
print(y.value_counts(normalize=False))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Print class distribution after split
print("\nClass distribution in training data:")
print(y_train.value_counts(normalize=False))

# Handle class imbalance with SMOTE 
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts(normalize=False))

# Extra Trees Classifier with hyperparameters from the paper
model = ExtraTreesClassifier(
    n_estimators=100,
    criterion='gini',
    max_features='sqrt',
    random_state=42
)

from sklearn.feature_selection import RFE
selector = RFE(model, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)
X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)
model.fit(X_train_sel, y_train)
y_pred = model.predict(X_test_sel)

# Train the model
#model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_predict, StratifiedKFold
cv = StratifiedKFold(5, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(model, X_train_sel, y_train, cv=cv, method='predict')


# Print feature importances
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
})
#print("\nFeature importances:")
#print(feature_importance.sort_values('importance', ascending=False))

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
#plt.show()
