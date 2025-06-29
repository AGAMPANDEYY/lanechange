# run_all_models.py
# Combined script to run all 4 models with and without SMOTE

import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 1. --- Shared Preprocessing ---
def load_and_preprocess():
    df = pd.read_csv('combined_data.csv')
    for col in ['0_source_lead','0_source_follow','0_target_lead','0_target_follow',
                '5_target_lead','5_target_follow','22_target_lead','22_target_follow','22','12','12_target_follow','12_target_lead']:
        if col in df.columns:
            m = df.loc[df[col] != -9999, col].mean()
            df[col] = df[col].replace(-9999, m)

    df['G_lead'] = df['5_target_lead'] - df['5']
    df['G_lag'] = df['5'] - df['5_target_follow']
    df['delta_v_LV_SV'] = df['22_target_lead'] - df['22']
    df['delta_v_SV_FV'] = df['22'] - df['22_target_follow']

    df['Previous_decision'] = df['Previous_decision'].astype(int)
    df = pd.get_dummies(df, columns=['Previous_decision'], prefix='PrevDec', dtype=int)
    for i in [0, 1, 2]:
        col = f'PrevDec_{i}'
        if col not in df.columns:
            df[col] = 0

    if 'Time_lapsed' in df.columns:
        df['Time_norm'] = (df['Time_lapsed'] - df['Time_lapsed'].mean()) / df['Time_lapsed'].std()
        df['Time_log'] = np.log1p(df['Time_lapsed'])
        df['Time_bin'] = pd.cut(df['Time_lapsed'], bins=[-1,5,10,30,np.inf], labels=[0,1,2,3]).astype(int)

    return df

# 2. --- Metric Utilities ---
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    dr  = tp / (tp + fn) * 100 if (tp + fn) else 0
    far = fp / (fp + tn) * 100 if (fp + tn) else 0
    return tp, fp, dr, far

def find_best_threshold(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    youden = tpr - fpr
    i = np.argmax(youden)
    best_t = thr[i]
    y_pred = (y_proba >= best_t).astype(int)
    tp, fp, dr, far = compute_metrics(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_proba)
    return best_t, {'TP':tp,'FP':fp,'DR':dr,'FAR':far,'AUC':auc_score}

# 3. --- ExtraTree Model ---
def run_extratree_model(df):
    features = [
        '22','12','22_source_lead','22_target_follow','12_target_follow',
        '22_target_lead','12_target_lead','G_lead','G_lag',
        'delta_v_LV_SV','delta_v_SV_FV'
    ]
    features_13 = features + ['PrevDec_0','PrevDec_1','PrevDec_2','Time_bin']
    results = []

    for name, feats in tqdm([('11-param', features), ('13-param', features_13)], desc="ExtraTree Models"):
        X = df[feats]
        y = df['LC']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        for smote_flag in [False, True]:
            X_tr, y_tr = X_train, y_train
            if smote_flag:
                X_tr, y_tr = SMOTE(random_state=42, k_neighbors=1).fit_resample(X_train, y_train)
            model = ExtraTreesClassifier(n_estimators=100, max_features='sqrt', class_weight='balanced', random_state=42)
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_test)[:, 1]
            best_t, m = find_best_threshold(y_test, y_proba)
            results.append({
                'Model': f'ExtraTree ({name})',
                'SMOTE': 'Yes' if smote_flag else 'No',
                'TP': m['TP'], 'FP': m['FP'],
                'DR (%)': f"{m['DR']:.2f}", 'FAR (%)': f"{m['FAR']:.2f}",
                'AUC': f"{m['AUC']:.3f}"
            })
    return results

# 4. --- Transformer Model ---
class LaneChangeDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='LC', neg_stride=10):
        self.samples = []
        for _, group in df.groupby('0'):
            group = group.sort_values('1')
            arr = group[feature_cols + [label_col]].values
            lc_indices = np.where(arr[:, -1] == 1)[0]
            for idx in lc_indices:
                if idx == 0: continue
                self.samples.append((arr[:idx, :-1].astype(np.float32), arr[idx, -1]))
            for idx in np.where(arr[:, -1] == 0)[0][::neg_stride]:
                if idx == 0: continue
                self.samples.append((arr[:idx, :-1].astype(np.float32), arr[idx, -1]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True)
    lengths = torch.tensor([x.shape[0] for x in xs])
    return xs_pad, lengths, torch.tensor(ys)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        out = x[torch.arange(x.size(0)), (lengths - 1).long()]
        return torch.sigmoid(self.fc(out)).squeeze(-1)

def run_transformer_model(df):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    
    # Base features (11 parameters)
    base_feats = [
        '22','12','22_source_lead','22_target_follow','12_target_follow',
                '22_target_lead','12_target_lead','G_lead','G_lag',
        'delta_v_LV_SV','delta_v_SV_FV'
    ]
    
    # Define all feature combinations to test
    feature_combinations = []
    
    # 1. 11 base features only
    feature_combinations.append(('11-param (base)', base_feats))
    
    # 2. 11 base + raw previous_decision and time_lapsed
    if 'Previous_decision' in df.columns and 'Time_lapsed' in df.columns:
        feature_combinations.append(('11-param + prev_dec + time_lapsed', 
                                   base_feats + ['Previous_decision', 'Time_lapsed']))
    
    # 3. 11 base + raw previous_decision and normalized/log time
    if 'Previous_decision' in df.columns and 'Time_norm' in df.columns and 'Time_log' in df.columns:
        feature_combinations.append(('11-param + prev_dec + time_norm', 
                                   base_feats + ['Previous_decision', 'Time_norm']))
        feature_combinations.append(('11-param + prev_dec + time_log', 
                                   base_feats + ['Previous_decision', 'Time_log']))
    
    # 4. 11 base + one-hot previous_decision and binned time
    ohe_cols = [f'PrevDec_{i}' for i in [0,1,2] if f'PrevDec_{i}' in df.columns]
    if ohe_cols and 'Time_bin' in df.columns:
        feature_combinations.append(('13-param (ohe + time_bin)', 
                                   base_feats + ohe_cols + ['Time_bin']))
    
    # Run experiments for all feature combinations with and without SMOTE
    for name, feats in tqdm(feature_combinations, desc="Transformer Models"):
        print(f"\nRunning Transformer ({name}) with {len(feats)} features...")
        train_ids, test_ids = train_test_split(df['0'].unique(), test_size=0.3, random_state=42)
        train_df = df[df['0'].isin(train_ids)]
        test_df  = df[df['0'].isin(test_ids)]
        # Test both with and without SMOTE
        for smote_flag in [False, True]:
            smote_label = " (SMOTE)" if smote_flag else ""
            print(f"Training {name}{smote_label}...")
            # Apply SMOTE if needed
            if smote_flag:
                print(f"Applying SMOTE to {name}...")
                # First create the dataset normally
                train_set_original = LaneChangeDataset(train_df, feats)
                # Extract features and labels from the dataset
                X_train = []
                y_train = []
                for x, y in train_set_original:
                    # Use the last timestep features as the feature vector
                    x_last = x[-1].numpy()  # Last timestep
                    # Clean any infinite or NaN values
                    x_clean = np.where(np.isinf(x_last) | np.isnan(x_last), 0.0, x_last)
                    X_train.append(x_clean)
                    y_train.append(y.numpy())
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
                print(f"y_train distribution: {np.bincount(y_train.astype(int))}")
                # Apply SMOTE to the feature vectors
                X_res, y_res = SMOTE(random_state=42, k_neighbors=1).fit_resample(X_train, y_train)
                # Create new dataset with SMOTE samples
                # For SMOTE samples, we'll create single-timestep sequences
                train_set_sm = []
                for i in range(len(X_res)):
                    # Create a single-timestep sequence for SMOTE samples
                    x_sm = torch.tensor(X_res[i].reshape(1, -1), dtype=torch.float32)
                    y_sm = torch.tensor(y_res[i], dtype=torch.float32)
                    train_set_sm.append((x_sm, y_sm))
                train_set = train_set_sm
                print(f"SMOTE applied: {len(train_set)} samples (was {len(train_set_original)})")
            else:
                train_set = LaneChangeDataset(train_df, feats)
            test_set = LaneChangeDataset(test_df, feats)
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
            test_loader  = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)
            model = TransformerClassifier(input_dim=len(feats)).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()
            # Training loop
            for epoch in tqdm(range(5), desc=f"Training {name}{smote_label}"):
                model.train()
                for xb, lengths, yb in train_loader:
                    xb, lengths, yb = xb.to(DEVICE), lengths.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(xb, lengths)
                    loss = criterion(out, yb.float())
                    loss.backward()
                    optimizer.step()
            # Evaluation
            model.eval()
            y_true, y_proba = [], []
            with torch.no_grad():
                for xb, lengths, yb in tqdm(test_loader, desc=f"Evaluating {name}{smote_label}"):
                    xb, lengths = xb.to(DEVICE), lengths.to(DEVICE)
                    out = model(xb, lengths).cpu().numpy()
                    y_true.extend(yb.numpy())
                    y_proba.extend(out)
            best_t, m = find_best_threshold(np.array(y_true), np.array(y_proba))
            results.append({
                'Model': f'Transformer ({name})',
                'SMOTE': 'Yes' if smote_flag else 'No',
                'TP': m['TP'], 'FP': m['FP'],
                'DR (%)': f"{m['DR']:.2f}", 'FAR (%)': f"{m['FAR']:.2f}",
                'AUC': f"{m['AUC']:.3f}"
            })
    
    return results

# 5. --- Kita + Utility Models ---
def run_kita_model(df):
    from kita_merging_model import compute_t_vars, feature_engineering, fit_model_with_features, predict_merge_giveway_with_features
    print("Computing t_vars for Kita model...")
    df_tvars = compute_t_vars(df)
    print("Applying feature engineering...")
    df_tvars = feature_engineering(df_tvars)
    results = []
    
    # Define all feature combinations to test (same as in kita_merging_model.py)
    feature_sets = [
        ([], 'Kita (Base t2, t4)'),
        (['Previous_decision', 'Time_lapsed'], 'Kita (Base + Previous_decision + Time_lapsed)'),
        (['Previous_decision', 'Time_norm'], 'Kita (Base + Previous_decision + Time_norm)'),
        (['Previous_decision', 'Time_log'], 'Kita (Base + Previous_decision + Time_log)'),
        ([f'PrevDec_{i}' for i in [0,1,2] if f'PrevDec_{i}' in df_tvars.columns] + ['Time_bin'], 'Kita (Base + one-hot Previous_decision + Time_bin)')
    ]
    
    for extra_feats, label in tqdm(feature_sets, desc="Kita Models"):
        print(f"\n--- Model Fitting and Validation: {label} ---")
        # Test both with and without SMOTE
        for smote_flag in [False, True]:
            smote_label = " (SMOTE)" if smote_flag else ""
            print(f"Building dataset for {label}{smote_label}...")
            dataset = []
            for _, row in tqdm(df_tvars.iterrows(), total=len(df_tvars), desc="Processing rows"):
                t2 = row['t2']
                t4 = row['t4']
                extras = [row[feat] for feat in extra_feats] if extra_feats else []
                merge_flag = row['LC']
                giveway_flag = row['Final_follower_decision'] if 'Final_follower_decision' in row else 1
                dataset.append(((t2, t4, *extras), (merge_flag, giveway_flag)))
            indices = np.arange(len(dataset))
            train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
            train_set = [dataset[i] for i in train_idx]
            test_set  = [dataset[i] for i in test_idx]
            # Apply SMOTE if needed
            if smote_flag:
                print(f"Applying SMOTE to {label}...")
                # Extract features and labels for SMOTE
                X_train = []
                y_train = []
                for (t2, t4, *extras), (merge_flag, _) in train_set:
                    # Clean infinite values - replace with large finite values
                    t2_clean = 100.0 if np.isinf(t2) or np.isnan(t2) else t2
                    t4_clean = 100.0 if np.isinf(t4) or np.isnan(t4) else t4
                    # Clean extras as well
                    extras_clean = []
                    for extra in extras:
                        if np.isinf(extra) or np.isnan(extra):
                            extras_clean.append(0.0)
                        else:
                            extras_clean.append(extra)
                    X_train.append([t2_clean, t4_clean] + extras_clean)
                    y_train.append(merge_flag)
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                # Apply SMOTE
                X_res, y_res = SMOTE(random_state=42, k_neighbors=1).fit_resample(X_train, y_train)
                # Reconstruct dataset with SMOTE
                train_set_sm = []
                for i in range(len(X_res)):
                    t2, t4 = X_res[i][0], X_res[i][1]
                    extras = X_res[i][2:].tolist()
                    merge_flag = y_res[i]
                    giveway_flag = 1  # Default value
                    train_set_sm.append(((t2, t4, *extras), (merge_flag, giveway_flag)))
                train_set = train_set_sm
                print(f"SMOTE applied: {len(train_set)} samples (was {len([dataset[i] for i in train_idx])})")
            print(f"Fitting {label}{smote_label} model...")
            # Better initial theta for better convergence (similar to good results)
            if len(extra_feats) == 0:
                # Base model (t2, t4 only) - use good initial values
                initial_theta = [0.08, 0.995, -0.007, 1.496, -0.033, 0.343, 5.45e-07]
            elif len(extra_feats) >= 3:  # Extended model with one-hot + time_bin
                # 13-param model - use good initial values
                initial_theta = [0.434, 0.758, 0.883, 0.094, 0.107, 0.498, 1.169, 
                               0.714, 0.165, 0.199, 0.119]
            else:
                # Other combinations - use random but reasonable values
                initial_theta = np.random.rand(7 + len(extra_feats)).tolist()
            print(f"Initial theta: {initial_theta}")
            theta = fit_model_with_features(train_set, extra_feats, initial_theta=initial_theta)
            print(f"Fitted theta: {theta}")
            y_true, y_proba = [], []
            print(f"Evaluating {label}{smote_label} model...")
            for (t2, t4, *extras), (merge_flag, _) in tqdm(test_set, desc="Making predictions"):
                prob, _ = predict_merge_giveway_with_features(t2, t4, extras, theta)
                y_true.append(merge_flag)
                y_proba.append(prob)
            best_t, m = find_best_threshold(np.array(y_true), np.array(y_proba))
            results.append({
                'Model': f"{label}{smote_label}",
                'SMOTE': 'Yes' if smote_flag else 'No',
                'TP': m['TP'], 'FP': m['FP'],
                'DR (%)': f"{m['DR']:.2f}", 'FAR (%)': f"{m['FAR']:.2f}",
                'AUC': f"{m['AUC']:.3f}"
            })
    
    return results

def run_utility_model(df):
    from utility_model_add_params import fit_mle_11, fit_mle_13, recompute_preds_11, recompute_preds_13
    results = []
    
    print("="*82)
    print("UTILITY MODEL ANALYSIS")
    print("="*82)
    
    # Check if we have the original Previous_decision column or the one-hot encoded version
    if 'Previous_decision' in df.columns:
        # Use original column
        required_cols = ['22','22_target_lead','22_target_follow','5','5_target_lead','5_target_follow','Previous_decision','Time_lapsed','LC']
    else:
        # Use one-hot encoded columns
        required_cols = ['22','22_target_lead','22_target_follow','5','5_target_lead','5_target_follow','Time_lapsed','LC']
        # Add the one-hot encoded columns if they exist
        for i in [0, 1, 2]:
            col = f'PrevDec_{i}'
            if col in df.columns:
                required_cols.append(col)
    
    # Filter to only include columns that exist
    existing_cols = [col for col in required_cols if col in df.columns]
    df_clean = df.dropna(subset=existing_cols).reset_index(drop=True)
    
    # For the utility model, we need to reconstruct Previous_decision if it doesn't exist
    if 'Previous_decision' not in df_clean.columns:
        # Reconstruct from one-hot encoded columns
        df_clean['Previous_decision'] = 0  # default
        for i in [0, 1, 2]:
            col = f'PrevDec_{i}'
            if col in df_clean.columns:
                df_clean.loc[df_clean[col] == 1, 'Previous_decision'] = i
    
    # Ensure Previous_decision is numeric
    if df_clean['Previous_decision'].dtype == object:
        df_clean['Previous_decision'] = df_clean['Previous_decision'].astype('category').cat.codes
    
    print(f"Loaded {len(df_clean)} samples for utility model analysis")
    print(f"Lane change events: {int(df_clean['LC'].sum())}")
    print(f"Non-lane change events: {len(df_clean) - int(df_clean['LC'].sum())}")
    
    # Train/test split (70/30 stratified on LC)
    df_train, df_test = train_test_split(
        df_clean, test_size=0.3, stratify=df_clean['LC'], random_state=42
    )
    
    # For SMOTE, we only need the following numeric columns from df_train:
    smote_feats = [
        '22','22_target_lead','22_target_follow',
        '5','5_target_lead','5_target_follow',
        'Previous_decision','Time_lapsed'
    ]
    X_train_sm = df_train[smote_feats].values
    y_train_sm = df_train['LC'].values.astype(int)
    
    print(f"\nTraining set: {len(df_train)} samples")
    print(f"Test set: {len(df_test)} samples")
    
    # =============================================================================
    # RUNNING THE FOUR SCENARIOS:
    # (A) 11-param WITHOUT SMOTE
    # (B) 11-param WITH SMOTE
    # (C) 13-param WITHOUT SMOTE
    # (D) 13-param WITH SMOTE
    # =============================================================================
    
    print("\n" + "="*82)
    print(f"{'Model / Sampling':<22} | {'#Rows':>6} | {'LC':>4} | {'NLC':>4} | {'TP':>4} | {'FP':>4} | {'DR (%)':>7} | {'FAR (%)':>7} | {'AUC':>6}")
    print("-"*82)
    
    # 3.2) Scenario A: 11-param WITHOUT SMOTE
    print("\nFitting 11-parameter model WITHOUT SMOTE...")
    print("Starting MLE optimization...")
    try:
        theta11_ws = fit_mle_11(df_train)
        print("MLE optimization completed successfully!")
        print("Computing predictions...")
        df_test_11_ws = recompute_preds_11(df_test, theta11_ws)
        print("Predictions computed successfully!")
        
        cm11_ws = confusion_matrix(df_test['LC'], df_test_11_ws['LC_pred_11'])
        tn11_ws, fp11_ws, fn11_ws, tp11_ws = cm11_ws.ravel()
        N11_ws   = len(df_test)
        LC11_ws  = int(df_test['LC'].sum())
        NLC11_ws = N11_ws - LC11_ws
        DR11_ws  = 100 * tp11_ws / (tp11_ws + fn11_ws) if (tp11_ws + fn11_ws) > 0 else 0.0
        FAR11_ws = 100 * fp11_ws / (fp11_ws + tn11_ws) if (fp11_ws + tn11_ws) > 0 else 0.0
        
        # Calculate AUC
        y_true_11_ws = df_test['LC'].values
        y_proba_11_ws = df_test_11_ws['P_lead_11'].values * df_test_11_ws['P_lag_11'].values
        auc11_ws = roc_auc_score(y_true_11_ws, y_proba_11_ws)
        
        print(f"{'11 params (WS)':<22} | "
              f"{N11_ws:>6} | "
              f"{LC11_ws:>4} | "
              f"{NLC11_ws:>4} | "
              f"{tp11_ws:>4} | "
              f"{fp11_ws:>4} | "
              f"{DR11_ws:>7.2f} | "
              f"{FAR11_ws:>7.2f} | "
              f"{auc11_ws:>6.3f}")
        
        results.append({'Model': 'Utility (11 param)', 'SMOTE': 'No', 'TP': tp11_ws, 'FP': fp11_ws, 'DR (%)': f"{DR11_ws:.2f}", 'FAR (%)': f"{FAR11_ws:.2f}", 'AUC': f"{auc11_ws:.3f}"})
    except Exception as e:
        print(f"Error in 11-param WITHOUT SMOTE: {e}")
        results.append({'Model': 'Utility (11 param)', 'SMOTE': 'No', 'TP': 0, 'FP': 0, 'DR (%)': '0.00', 'FAR (%)': '0.00', 'AUC': '0.000'})
    
    # 3.3) Scenario B: 11-param WITH SMOTE
    print("\nFitting 11-parameter model WITH SMOTE...")
    print("Applying SMOTE...")
    try:
        sm = SMOTE(random_state=42, k_neighbors=1)
        X_res11, y_res11 = sm.fit_resample(X_train_sm, y_train_sm)
        train_sm_11 = pd.DataFrame(X_res11, columns=smote_feats)
        train_sm_11['LC'] = y_res11
        print(f"SMOTE applied: {len(train_sm_11)} samples (was {len(df_train)})")
        
        print("Starting MLE optimization with SMOTE...")
        theta11_s = fit_mle_11(train_sm_11)
        print("MLE optimization with SMOTE completed successfully!")
        print("Computing predictions...")
        df_test_11_s = recompute_preds_11(df_test, theta11_s)
        print("Predictions computed successfully!")
        
        cm11_s = confusion_matrix(df_test['LC'], df_test_11_s['LC_pred_11'])
        tn11_s, fp11_s, fn11_s, tp11_s = cm11_s.ravel()
        N11_s   = len(df_test)
        LC11_s  = int(df_test['LC'].sum())
        NLC11_s = N11_s - LC11_s
        DR11_s  = 100 * tp11_s / (tp11_s + fn11_s) if (tp11_s + fn11_s) > 0 else 0.0
        FAR11_s = 100 * fp11_s / (fp11_s + tn11_s) if (fp11_s + tn11_s) > 0 else 0.0
        
        # Calculate AUC
        y_true_11_s = df_test['LC'].values
        y_proba_11_s = df_test_11_s['P_lead_11'].values * df_test_11_s['P_lag_11'].values
        auc11_s = roc_auc_score(y_true_11_s, y_proba_11_s)
        
        print(f"{'11 params (S)':<22} | "
              f"{N11_s:>6} | "
              f"{LC11_s:>4} | "
              f"{NLC11_s:>4} | "
              f"{tp11_s:>4} | "
              f"{fp11_s:>4} | "
              f"{DR11_s:>7.2f} | "
              f"{FAR11_s:>7.2f} | "
              f"{auc11_s:>6.3f}")
        
        results.append({'Model': 'Utility (11 param)', 'SMOTE': 'Yes', 'TP': tp11_s, 'FP': fp11_s, 'DR (%)': f"{DR11_s:.2f}", 'FAR (%)': f"{FAR11_s:.2f}", 'AUC': f"{auc11_s:.3f}"})
    except Exception as e:
        print(f"Error in 11-param WITH SMOTE: {e}")
        results.append({'Model': 'Utility (11 param)', 'SMOTE': 'Yes', 'TP': 0, 'FP': 0, 'DR (%)': '0.00', 'FAR (%)': '0.00', 'AUC': '0.000'})
    
    # 3.4) Scenario C: 13-param WITHOUT SMOTE
    print("\nFitting 13-parameter model WITHOUT SMOTE...")
    print("Starting MLE optimization...")
    try:
        theta13_ws = fit_mle_13(df_train)
        print("MLE optimization completed successfully!")
        print("Computing predictions...")
        df_test_13_ws = recompute_preds_13(df_test, theta13_ws)
        print("Predictions computed successfully!")
        
        cm13_ws = confusion_matrix(df_test['LC'], df_test_13_ws['LC_pred_13'])
        tn13_ws, fp13_ws, fn13_ws, tp13_ws = cm13_ws.ravel()
        N13_ws   = len(df_test)
        LC13_ws  = int(df_test['LC'].sum())
        NLC13_ws = N13_ws - LC13_ws
        DR13_ws  = 100 * tp13_ws / (tp13_ws + fn13_ws) if (tp13_ws + fn13_ws) > 0 else 0.0
        FAR13_ws = 100 * fp13_ws / (fp13_ws + tn13_ws) if (fp13_ws + tn13_ws) > 0 else 0.0
        
        # Calculate AUC
        y_true_13_ws = df_test['LC'].values
        y_proba_13_ws = df_test_13_ws['P_lead_13'].values * df_test_13_ws['P_lag_13'].values
        auc13_ws = roc_auc_score(y_true_13_ws, y_proba_13_ws)
        
        print(f"{'13 params (WS)':<22} | "
              f"{N13_ws:>6} | "
              f"{LC13_ws:>4} | "
              f"{NLC13_ws:>4} | "
              f"{tp13_ws:>4} | "
              f"{fp13_ws:>4} | "
              f"{DR13_ws:>7.2f} | "
              f"{FAR13_ws:>7.2f} | "
              f"{auc13_ws:>6.3f}")
        
        results.append({'Model': 'Utility (13 param)', 'SMOTE': 'No', 'TP': tp13_ws, 'FP': fp13_ws, 'DR (%)': f"{DR13_ws:.2f}", 'FAR (%)': f"{FAR13_ws:.2f}", 'AUC': f"{auc13_ws:.3f}"})
    except Exception as e:
        print(f"Error in 13-param WITHOUT SMOTE: {e}")
        results.append({'Model': 'Utility (13 param)', 'SMOTE': 'No', 'TP': 0, 'FP': 0, 'DR (%)': '0.00', 'FAR (%)': '0.00', 'AUC': '0.000'})
    
    # 3.5) Scenario D: 13-param WITH SMOTE
    print("\nFitting 13-parameter model WITH SMOTE...")
    try:
        # Use the same train_sm_11 (it has all 8 numeric columns + 'LC')
        train_sm_13 = train_sm_11.copy()
        print("Starting MLE optimization with SMOTE...")
        theta13_s = fit_mle_13(train_sm_13)
        print("MLE optimization with SMOTE completed successfully!")
        print("Computing predictions...")
        df_test_13_s = recompute_preds_13(df_test, theta13_s)
        print("Predictions computed successfully!")
        
        cm13_s = confusion_matrix(df_test['LC'], df_test_13_s['LC_pred_13'])
        tn13_s, fp13_s, fn13_s, tp13_s = cm13_s.ravel()
        N13_s   = len(df_test)
        LC13_s  = int(df_test['LC'].sum())
        NLC13_s = N13_s - LC13_s
        DR13_s  = 100 * tp13_s / (tp13_s + fn13_s) if (tp13_s + fn13_s) > 0 else 0.0
        FAR13_s = 100 * fp13_s / (fp13_s + tn13_s) if (fp13_s + tn13_s) > 0 else 0.0
        
        # Calculate AUC
        y_true_13_s = df_test['LC'].values
        y_proba_13_s = df_test_13_s['P_lead_13'].values * df_test_13_s['P_lag_13'].values
        auc13_s = roc_auc_score(y_true_13_s, y_proba_13_s)
        
        print(f"{'13 params (S)':<22} | "
              f"{N13_s:>6} | "
              f"{LC13_s:>4} | "
              f"{NLC13_s:>4} | "
              f"{tp13_s:>4} | "
              f"{fp13_s:>4} | "
              f"{DR13_s:>7.2f} | "
              f"{FAR13_s:>7.2f} | "
              f"{auc13_s:>6.3f}")
        
        results.append({'Model': 'Utility (13 param)', 'SMOTE': 'Yes', 'TP': tp13_s, 'FP': fp13_s, 'DR (%)': f"{DR13_s:.2f}", 'FAR (%)': f"{FAR13_s:.2f}", 'AUC': f"{auc13_s:.3f}"})
    except Exception as e:
        print(f"Error in 13-param WITH SMOTE: {e}")
        results.append({'Model': 'Utility (13 param)', 'SMOTE': 'Yes', 'TP': 0, 'FP': 0, 'DR (%)': '0.00', 'FAR (%)': '0.00', 'AUC': '0.000'})
    
    print("="*82)
    print("UTILITY MODEL ANALYSIS COMPLETED")
    print("="*82 + "\n")
    
    return results

# 6. --- Main Execution ---
if __name__ == '__main__':
    print("Loading and preprocessing data...")
    df = load_and_preprocess()
    results = []
    
    print("\n" + "="*50)
    print("Running All Models")
    print("="*50)
    
    # Run ExtraTree models
    print("\n1. Running ExtraTree Models...")
    results.extend(run_extratree_model(df))
    
    # Run Transformer model
    print("\n2. Running Transformer Model...")
    results.extend(run_transformer_model(df))
    
    # Run Kita models
    print("\n3. Running Kita Models...")
    results.extend(run_kita_model(df))
    
    # Run Utility models
    print("\n4. Running Utility Models...")
    results.extend(run_utility_model(df))
    
    print("\nSaving results...")
    df_results = pd.DataFrame(results)
    df_results.to_csv('all_model_comparison.csv', index=False)
    print("\n=== Combined Results ===")
    print("Model | SMOTE | TP | FP | DR (%) | FAR (%) | AUC")
    print("------|-------|----|----|--------|---------|-----")
    for _, row in df_results.iterrows():
        print(f"{row['Model']} | {row['SMOTE']} | {row['TP']} | {row['FP']} | {row['DR (%)']} | {row['FAR (%)']} | {row['AUC']}")
    print("\nDetailed results saved to 'all_model_comparison.csv'")