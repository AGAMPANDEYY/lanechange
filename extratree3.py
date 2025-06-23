import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

# --- helpers ---
def compute_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    dr  = tp / (tp + fn) * 100 if (tp + fn) else 0
    far = fp / (fp + tn) * 100 if (fp + tn) else 0
    return tp, fp, dr, far

def find_best_threshold(y_true, y_proba):
    """Return (best_threshold, metrics_dict) maximizing Youden's J=TPR-FPR."""
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    youden = tpr - fpr
    i = np.argmax(youden)
    best_t = thr[i]
    # build metrics at best_t
    y_pred = (y_proba >= best_t).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, dr, far = compute_metrics(cm)
    return best_t, {'TP':tp,'FP':fp,'DR':dr,'FAR':far}

# --- load & preprocess ---
df = pd.read_csv(r'C:\Agam\Work\lanechange\combined_data.csv')

# Impute
for col in ['0_source_lead','0_source_follow','0_target_lead','0_target_follow']:
    m = df.loc[df[col] != -9999, col].mean()
    df[col] = df[col].replace(-9999, m)

# Engineered features
df['G_lead']        = df['5_target_lead'] - df['5']
df['G_lag']         = df['5'] - df['5_target_follow']
df['delta_v_LV_SV'] = df['22_target_lead'] - df['22']
df['delta_v_SV_FV'] = df['22'] - df['22_target_follow']

# Convert Previous_decision to integer (if possible)
df['Previous_decision'] = df['Previous_decision'].astype(int)

# Drop all columns that start with 'PrevDec_'
df = df.loc[:, ~df.columns.str.startswith('PrevDec_')]

# One-hot encode Previous_decision, ensure dtype is int
df = pd.get_dummies(df, columns=['Previous_decision'], prefix='PrevDec', dtype=int)

# Ensure all expected dummy columns exist
for i in [0, 1, 2]:
    col = f'PrevDec_{i}'
    if col not in df.columns:
        df[col] = 0

# Time bin
df['Time_bin'] = pd.cut(
    df['Time_lapsed'],
    bins=[-1,5,10,30,np.inf],
    labels=[0,1,2,3]
).astype(int)

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved to 'preprocessed_data.csv'")

# --- define feature sets ---
features_11 = [
    '22','12','22_source_lead','22_target_follow','12_target_follow',
    '22_target_lead','12_target_lead','G_lead','G_lag',
    'delta_v_LV_SV','delta_v_SV_FV'
]
features_13 = features_11 + ['PrevDec_0','PrevDec_1','PrevDec_2','Time_bin']

sets = {
    '11 params': features_11,
    '13 params': features_13
}

# --- train/test split ---
X = df.drop(columns=['LC'])
y = df['LC']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- run experiments ---
results = []
n0_test = (y_test == 0).sum()
n1_test = (y_test == 1).sum()

for name, feats in sets.items():
    # 1) RFE on TRAIN (just to be consistent, though we're keeping all)
    selector = RFE(
        ExtraTreesClassifier(
            n_estimators=100, max_features='sqrt',
            class_weight='balanced', random_state=42
        ),
        n_features_to_select=len(feats), step=1
    )
    selector.fit(X_train[feats], y_train)

    X_tr_sel = selector.transform(X_train[feats])
    X_te_sel = selector.transform(X_test[feats])

    for use_smote in [False, True]:
        # 2) Train
        if use_smote:
            X_rs, y_rs = SMOTE(
                random_state=42, k_neighbors=1
            ).fit_resample(X_tr_sel, y_train)
            model = ExtraTreesClassifier(
                n_estimators=100, max_features='sqrt',
                class_weight='balanced', random_state=42
            )
            model.fit(X_rs, y_rs)
        else:
            model = ExtraTreesClassifier(
                n_estimators=100, max_features='sqrt',
                class_weight='balanced', random_state=42
            )
            model.fit(X_tr_sel, y_train)

        # 3) Get probabilities & find best threshold
        y_proba = model.predict_proba(X_te_sel)[:,1]
        best_t, m = find_best_threshold(y_test, y_proba)

        results.append({
            'Features': name,
            'SMOTE':    'Yes' if use_smote else 'No',
            'Thresh':   f"{best_t:.2f}",
            'N_total':  len(y_test),
            'N_0':      n0_test,
            'N_1':      n1_test,
            'TP':       m['TP'],
            'FP':       m['FP'],
            'DR (%)':   f"{m['DR']:.2f}",
            'FAR (%)':  f"{m['FAR']:.2f}"
        })

# --- final table ---
df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))

df_results.to_csv('extratree3.csv', index=False)
