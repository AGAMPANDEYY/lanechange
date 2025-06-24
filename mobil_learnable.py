import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

# --- MOBIL Feature Engineering ---
def extract_mobil_features(df):
    # Current accelerations
    a_c = df['12']  # Ego vehicle
    a_o = df['12_source_follow']  # Old follower
    a_n = df['12_target_follow']  # New follower
    # Predicted accelerations (not available, set to zero)
    tilde_a_c = 0
    tilde_a_o = 0
    tilde_a_n = 0
    # MOBIL terms
    ego_gain = tilde_a_c - a_c
    old_follower_gain = tilde_a_o - a_o
    new_follower_gain = tilde_a_n - a_n
    return pd.DataFrame({
        'ego_gain': ego_gain,
        'old_follower_gain': old_follower_gain,
        'new_follower_gain': new_follower_gain
    })

# --- Extra Features (examples, can be expanded) ---
def extract_extra_features(df):
    # Not used in this version, but kept for extensibility
    return pd.DataFrame()

def extract_prev_time_features(df):
    prev_dec = df['Previous_decision'] if 'Previous_decision' in df.columns else 0
    time_lapsed = df['Time_lapsed'] if 'Time_lapsed' in df.columns else 0
    return pd.DataFrame({
        'Previous_decision': prev_dec,
        'Time_lapsed': time_lapsed
    })

def feature_engineering(df):
    # Time features
    if 'Time_lapsed' in df.columns:
        df['Time_norm'] = (df['Time_lapsed'] - df['Time_lapsed'].mean()) / df['Time_lapsed'].std()
        df['Time_log'] = np.log1p(df['Time_lapsed'])
        df['Time_bin'] = pd.cut(
            df['Time_lapsed'],
            bins=[-1,5,10,30,np.inf],
            labels=[0,1,2,3]
        ).astype(int)
    # One-hot previous_decision
    if 'Previous_decision' in df.columns:
        df['Previous_decision'] = df['Previous_decision'].astype(int)
        prev_dec_ohe = pd.get_dummies(df['Previous_decision'], prefix='PrevDec', dtype=int)
        for c in [0, 1, 2]:
            col = f'PrevDec_{c}'
            if col not in prev_dec_ohe.columns:
                prev_dec_ohe[col] = 0
        df = pd.concat([df, prev_dec_ohe], axis=1)
    return df

def run_experiment(X, y, desc):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, thr = roc_curve(y_test, y_proba)
    youden = tpr - fpr
    i = np.argmax(youden)
    best_t = thr[i]
    y_pred = (y_proba >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    dr  = tp / (tp + fn) * 100 if (tp + fn) else 0
    far = fp / (fp + tn) * 100 if (fp + tn) else 0
    print(f"\n--- MOBIL Model: {desc} ---")
    print(f"Best threshold: {best_t:.3f}")
    print(f"TP={tp}, FP={fp}, DR={dr:.2f}%, FAR={far:.2f}%")
    print("Learned coefficients:")
    for name, coef in zip(X.columns, model.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    print(f"Intercept (absorbs threshold): {model.intercept_[0]:.4f}")

# --- Main ---
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('combined_data.csv')
    # Replace -9999 with column mean
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if (df[col] == -9999).any():
                m = df.loc[df[col] != -9999, col].mean()
                df[col] = df[col].replace(-9999, m)
    df = feature_engineering(df)
    # Label
    if 'LC' in df.columns:
        y = df['LC']
    elif '22' in df.columns:
        y = df['22']
    else:
        raise ValueError('No lane change label found (expected LC or 22)')
    mobil_feats = extract_mobil_features(df)
    # 1. Base MOBIL features only
    run_experiment(mobil_feats, y, 'Base (MOBIL features only)')
    # 2. Base + Previous_decision + Time_lapsed
    prev_time_feats = df[['Previous_decision','Time_lapsed']] if 'Previous_decision' in df.columns and 'Time_lapsed' in df.columns else pd.DataFrame()
    if not prev_time_feats.empty:
        X2 = pd.concat([mobil_feats, prev_time_feats], axis=1)
        run_experiment(X2, y, 'Base + Previous_decision + Time_lapsed')
    # 3. Base + Previous_decision + Time_norm
    if 'Previous_decision' in df.columns and 'Time_norm' in df.columns:
        X3 = pd.concat([mobil_feats, df[['Previous_decision','Time_norm']]], axis=1)
        run_experiment(X3, y, 'Base + Previous_decision + Time_norm')
    # 4. Base + Previous_decision + Time_log
    if 'Previous_decision' in df.columns and 'Time_log' in df.columns:
        X4 = pd.concat([mobil_feats, df[['Previous_decision','Time_log']]], axis=1)
        run_experiment(X4, y, 'Base + Previous_decision + Time_log')
    # 5. Base + one-hot Previous_decision + Time_bin
    ohe_cols = [f'PrevDec_{i}' for i in [0,1,2] if f'PrevDec_{i}' in df.columns]
    if ohe_cols and 'Time_bin' in df.columns:
        X5 = pd.concat([mobil_feats, df[ohe_cols + ['Time_bin']]], axis=1)
        run_experiment(X5, y, 'Base + one-hot Previous_decision + Time_bin') 