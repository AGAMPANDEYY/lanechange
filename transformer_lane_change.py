import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data Preparation ---
class LaneChangeDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='LC', neg_stride=10):
        self.samples = []
        grouped = df.groupby('0')
        for veh_id, group in grouped:
            group = group.sort_values('1')
            arr = group[feature_cols + [label_col]].values
            # Positive samples (LC==1)
            lc_indices = np.where(arr[:, -1] == 1)[0]
            for idx in lc_indices:
                if idx == 0: continue
                x = arr[:idx, :-1].astype(np.float32)
                y = arr[idx, -1].astype(np.float32)
                self.samples.append((x, y))
            # Negative samples (LC==0, sampled every neg_stride frames)
            neg_indices = np.where(arr[:, -1] == 0)[0]
            for idx in neg_indices[::neg_stride]:
                if idx == 0: continue
                x = arr[:idx, :-1].astype(np.float32)
                y = arr[idx, -1].astype(np.float32)
                self.samples.append((x, y))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True)  # (batch, max_seq, features)
    lengths = torch.tensor([x.shape[0] for x in xs])
    ys = torch.tensor(ys)
    return xs_padded, lengths, ys

# --- Model ---
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x, lengths):
        # x: (batch, seq, features)
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        # Use the last valid time step for each sequence
        last_indices = (lengths - 1).long()
        out = x[torch.arange(x.size(0)), last_indices]
        return torch.sigmoid(self.fc(out)).squeeze(-1)

# --- Metrics ---
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
    return best_t, {'TP':tp,'FP':fp,'DR':dr,'FAR':far}

def run_experiment(df, feature_cols, label_col, desc):
    veh_ids = df['0'].unique()
    train_ids, test_ids = train_test_split(veh_ids, test_size=0.3, random_state=42)
    train_df = df[df['0'].isin(train_ids)]
    test_df  = df[df['0'].isin(test_ids)]
    train_set = LaneChangeDataset(train_df, feature_cols, label_col)
    test_set  = LaneChangeDataset(test_df, feature_cols, label_col)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = TransformerClassifier(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    for epoch in range(EPOCHS):
        model.train()
        for xb, lengths, yb in train_loader:
            xb, lengths, yb = xb.to(DEVICE), lengths.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb, lengths)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    y_true, y_proba = [], []
    with torch.no_grad():
        for xb, lengths, yb in test_loader:
            xb, lengths = xb.to(DEVICE), lengths.to(DEVICE)
            out = model(xb, lengths).cpu().numpy()
            y_true.extend(yb.numpy())
            y_proba.extend(out)
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    best_t, m = find_best_threshold(y_true, y_proba)
    print(f"\n[{desc}]\nBest threshold: {best_t:.2f}\nTP: {m['TP']}, FP: {m['FP']}, DR: {m['DR']:.2f}%, FAR: {m['FAR']:.2f}%")

# --- Main ---
if __name__ == "__main__":
    # --- Load Data ---
    df = pd.read_csv('combined_data.csv')
    # Feature engineering: add G_lead, G_lag, delta_v_LV_SV, delta_v_SV_FV if not present
    if 'G_lead' not in df.columns:
        df['G_lead']        = df['5_target_lead'] - df['5']
    if 'G_lag' not in df.columns:
        df['G_lag']         = df['5'] - df['5_target_follow']
    if 'delta_v_LV_SV' not in df.columns:
        df['delta_v_LV_SV'] = df['22_target_lead'] - df['22']
    if 'delta_v_SV_FV' not in df.columns:
        df['delta_v_SV_FV'] = df['22'] - df['22_target_follow']
    # Replace -9999 with column mean (excluding -9999)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if (df[col] == -9999).any():
                m = df.loc[df[col] != -9999, col].mean()
                df[col] = df[col].replace(-9999, m)
    # Normalized and log time
    if 'Time_lapsed' in df.columns:
        df['Time_norm'] = (df['Time_lapsed'] - df['Time_lapsed'].mean()) / df['Time_lapsed'].std()
        df['Time_log'] = np.log1p(df['Time_lapsed'])
    # One-hot previous_decision
    if 'Previous_decision' in df.columns:
        df['Previous_decision'] = df['Previous_decision'].astype(int)
        prev_dec_ohe = pd.get_dummies(df['Previous_decision'], prefix='PrevDec', dtype=int)
        for c in [0, 1, 2]:
            col = f'PrevDec_{c}'
            if col not in prev_dec_ohe.columns:
                prev_dec_ohe[col] = 0
        df = pd.concat([df, prev_dec_ohe], axis=1)
    # Time bin
    if 'Time_lapsed' in df.columns:
        df['Time_bin'] = pd.cut(
            df['Time_lapsed'],
            bins=[-1,5,10,30,np.inf],
            labels=[0,1,2,3]
        ).astype(int)
    label_col = 'LC'
    # 11 base features
    base_feats = [
        '22','12','22_source_lead','22_target_follow','12_target_follow',
        '22_target_lead','12_target_lead','G_lead','G_lag',
        'delta_v_LV_SV','delta_v_SV_FV'
    ]
    # 1. 11 base features
    run_experiment(df, base_feats, label_col, '11 base features')
    # 2. 11 base + raw previous_decision and time_lapsed
    if 'Previous_decision' in df.columns and 'Time_lapsed' in df.columns:
        run_experiment(df, base_feats + ['Previous_decision', 'Time_lapsed'], label_col, '11 + prev_decision + time_lapsed')
    # 3. 11 base + raw previous_decision and normalized/log time
    if 'Previous_decision' in df.columns and 'Time_norm' in df.columns and 'Time_log' in df.columns:
        run_experiment(df, base_feats + ['Previous_decision', 'Time_norm'], label_col, '11 + prev_decision + time_norm')
        run_experiment(df, base_feats + ['Previous_decision', 'Time_log'], label_col, '11 + prev_decision + time_log')
    # 4. 11 base + one-hot previous_decision and binned time
    ohe_cols = [f'PrevDec_{i}' for i in [0,1,2] if f'PrevDec_{i}' in df.columns]
    if ohe_cols and 'Time_bin' in df.columns:
        run_experiment(df, base_feats + ohe_cols + ['Time_bin'], label_col, '11 + one-hot prev_decision + time_bin')

    # --- Save Results ---
    #pd.DataFrame({'y_true': y_true, 'y_proba': y_proba}).to_csv('transformer_preds.csv', index=False)