import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy.optimize import minimize
import nashpy as nash
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

# ========================================
# 1. Compute explanatory variables (t1–t5)
# ========================================

def compute_t_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event (row), compute t2 and t4 using the available columns.
    Returns a DataFrame with columns: event_id, t2, t4, merging_id, lead_id, lag_id, LC, Final_follower_decision, Previous_decision, Time_lapsed
    - t2: TTC of merging car to lead in target lane
    - t4: Time headway between lag and lead in target lane
    """
    results = []
    for idx, row in df.iterrows():
        merging_id = row['0']
        lead_id = row['0_target_lead'] if '0_target_lead' in row else None
        lag_id = row['0_target_follow'] if '0_target_follow' in row else None
        
        # t2: TTC of merging car to lead in target lane
        try:
            x_lead = row['4_target_lead']
            x_subj = row['5']
            v_subj = row['22']  # Use column 22 for subject vehicle speed
            v_lead = row['22_target_lead']
            denom = v_subj - v_lead
            t2 = (x_lead - x_subj) / denom if denom != 0 else float('inf')
        except Exception:
            t2 = None
        
        # t4: Time headway between lag and lead in target lane
        try:
            x_lag = row['4_target_follow']
            v_lag = row['22_target_follow']
            x_lead = row['4_target_lead']
            t4 = (x_lead - x_lag) / v_lag if v_lag != 0 else float('inf')
        except Exception:
            t4 = None
        
        results.append({
            'event_id': idx,
            't2': t2,
            't4': t4,
            'merging_id': merging_id,
            'lead_id': lead_id,
            'lag_id': lag_id,
            'LC': row.get('LC', np.nan),
            'Final_follower_decision': row.get('Final_follower_decision', np.nan),
            'Previous_decision': row.get('Previous_decision', np.nan),
            'Time_lapsed': row.get('Time_lapsed', np.nan)
        })
    results_df = pd.DataFrame(results)
    # Replace NaN with mean of the column if any
    for col in ['t2', 't4']:
        valid = results_df[col][~results_df[col].isna()]
        if not valid.empty:
            mean_val = valid.mean()
            results_df[col] = results_df[col].fillna(mean_val)
    return results_df

def compute_t_features(
    distance_to_lane_end: float,
    speed_merging: float,
    ttc_m3: float,
    ttc_m4: float,
    headway_23: float,
    ttc_35: float
) -> Tuple[float, float, float, float, float]:
    """
    Alternative function for computing t1-t5 from individual components
    t1: time to end of acceleration lane
    t2: TTC (merging vehicle to through car [3])
    t3: TTC (merging vehicle to through car [4])
    t4: Time headway between [2] and [3]
    t5: TTC (through car [3] to adjacent lane car [5])
    """
    t1 = distance_to_lane_end / speed_merging if speed_merging > 0 else float('inf')
    t2 = ttc_m3
    t3 = ttc_m4
    t4 = headway_23
    t5 = ttc_35
    return t1, t2, t3, t4, t5

# ========================================
# 2. Define Payoff Matrices (using theta)
# ========================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Normalized and log time
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

def payoff_matrices(t: Tuple, theta: List[float], extra_features: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized payoff matrix to include extra features and theta parameters.
    t: (t1, t2, t3, t4, t5)
    extra_features: list of extra feature values (same order as extra theta)
    theta: [theta1, ..., theta7, theta8, ...]
    """
    t1, t2, t3, t4, t5 = t
    θ1, θ2, θ3, θ4, θ5, θ6, θ7 = theta[:7]
    extra = 0
    if extra_features is not None and len(theta) > 7:
        for i, val in enumerate(extra_features):
            extra += theta[7 + i] * val
    # P1 (merging vehicle) payoffs
    a11 = θ1 * t3 + θ2 + extra        # merge, giveway
    a12 = θ1 * t2 + θ2 + extra        # merge, no giveway
    a21 = a22 = θ3 * t1 + extra       # wait
    A = np.array([[a11, a12], [a21, a22]])
    # P2 (through vehicle) payoffs
    b11 = b21 = θ4 * t5 + θ5 + extra  # giveway
    b12 = θ6 * t2 + extra             # no giveway, merge
    b22 = θ7 * t4 + extra             # no giveway, pass
    B = np.array([[b11, b12], [b21, b22]])
    return A, B

# ========================================
# 3. Solve Nash Equilibrium
# ========================================

def compute_equilibrium_nashpy(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Nash equilibrium using NashPy library
    """
    try:
        game = nash.Game(A, B)
        eq = list(game.support_enumeration())
        if eq:
            return eq[0]  # (p*, q*)
    except:
        pass
    return np.array([0.5, 0.5]), np.array([0.5, 0.5])  # default fallback

def compute_equilibrium_closed_form(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    """
    Closed-form Nash equilibrium for 2x2 game
    P1: row player, P2: column player
    p* = prob P1 plays row 0 (merge), q* = prob P2 plays col 0 (giveway)
    """
    a11, a12, a21, a22 = A.flatten()
    b11, b12, b21, b22 = B.flatten()
    
    # q* = (a22 - a12) / (a11 - a12 - a21 + a22)
    # p* = (b22 - b21) / (b11 - b12 - b21 + b22)
    denom1 = (a11 - a12 - a21 + a22)
    denom2 = (b11 - b12 - b21 + b22)
    
    if denom1 == 0 or not np.isfinite(denom1):
        q_star = 0.5
    else:
        q_star = (a22 - a12) / denom1
    if denom2 == 0 or not np.isfinite(denom2):
        p_star = 0.5
    else:
        p_star = (b22 - b21) / denom2
    
    # Clamp to [0,1]
    q_star = np.clip(q_star, 0, 1)
    p_star = np.clip(p_star, 0, 1)
    
    return p_star, q_star

def compute_equilibrium(A: np.ndarray, B: np.ndarray, method: str = 'closed_form') -> Tuple[float, float]:
    """
    Solve Nash equilibrium using specified method
    """
    if method == 'nashpy':
        p_vec, q_vec = compute_equilibrium_nashpy(A, B)
        return p_vec[0], q_vec[0]
    else:  # closed_form
        return compute_equilibrium_closed_form(A, B)

# ========================================
# 4. Log-likelihood for parameter optimization
# ========================================

def log_likelihood_loss(theta: List[float], data: List[Tuple[Tuple[float, float, float, float, float], Tuple[int, int]]]) -> float:
    """
    theta: list of θ1–θ7
    data: list of (t1–t5, (merge_flag, giveway_flag)) observations
    """
    epsilon = 1e-6
    neg_log_likelihood = 0.0

    for features, (merge_obs, giveway_obs) in data:
        A, B = payoff_matrices(features, theta)
        p_merge, q_yield = compute_equilibrium(A, B)

        # Assume binary observations
        prob = (
            p_merge if merge_obs else (1 - p_merge)
        ) * (
            q_yield if giveway_obs else (1 - q_yield)
        )

        neg_log_likelihood -= np.log(prob + epsilon)

    return neg_log_likelihood

# ========================================
# 5. Fit parameters from data
# ========================================

def fit_model(data: List[Tuple[Tuple[float, float], Tuple[int, int]]], 
              initial_theta: Optional[List[float]] = None) -> List[float]:
    """
    Fit model parameters using maximum likelihood estimation
    Only t2 and t4 are used; t1, t3, t5 are set to 0.
    """
    def wrap_data_for_full_tvars(data):
        # Convert (t2, t4) to (t1, t2, t3, t4, t5) with t1, t3, t5 = 0
        return [((0, t2, 0, t4, 0), obs) for (t2, t4), obs in data]
    if initial_theta is None:
        initial_theta = [1.0] * 7  # Default initialization
    full_data = wrap_data_for_full_tvars(data)
    result = minimize(log_likelihood_loss, initial_theta, args=(full_data,), method="Nelder-Mead")
    return result.x.tolist()

def fit_model_with_features(data, extra_features_list, initial_theta=None):
    """
    Fit model with extra features. data: [((t2, t4, extras...), (merge_flag, giveway_flag))]
    """
    def wrap_data_for_full_tvars(data):
        # Convert (t2, t4, extras...) to (t1, t2, t3, t4, t5, extras...) with t1, t3, t5 = 0
        return [((0, t2, 0, t4, 0, *extras), obs) for (t2, t4, *extras), obs in data]
    if initial_theta is None:
        initial_theta = [1.0] * (7 + len(extra_features_list))
    full_data = wrap_data_for_full_tvars(data)
    def log_likelihood_loss_theta(theta, data):
        epsilon = 1e-6
        neg_log_likelihood = 0.0
        for features, (merge_obs, giveway_obs) in data:
            t1, t2, t3, t4, t5, *extras = features
            A, B = payoff_matrices((t1, t2, t3, t4, t5), theta, extras)
            p_merge, q_yield = compute_equilibrium(A, B)
            prob = (
                p_merge if merge_obs else (1 - p_merge)
            ) * (
                q_yield if giveway_obs else (1 - q_yield)
            )
            neg_log_likelihood -= np.log(prob + epsilon)
        return neg_log_likelihood
    result = minimize(log_likelihood_loss_theta, initial_theta, args=(full_data,), method="Nelder-Mead")
    return result.x.tolist()

# ========================================
# 6. Predict using fitted model
# ========================================

def predict_merge_giveway(
    t_features: Tuple[float, float],
    theta: List[float]
) -> Tuple[float, float]:
    """
    Predict merge and giveway probabilities for given t2, t4 and theta parameters
    t1, t3, t5 are set to 0.
    """
    t1, t2, t3, t4, t5 = 0, t_features[0], 0, t_features[1], 0
    A, B = payoff_matrices((t1, t2, t3, t4, t5), theta)
    p_merge, q_yield = compute_equilibrium(A, B)
    return p_merge, q_yield

def predict_merge_giveway_with_features(t2, t4, extras, theta):
    t1, t3, t5 = 0, 0, 0
    A, B = payoff_matrices((t1, t2, t3, t4, t5), theta, extras)
    p_merge, q_yield = compute_equilibrium(A, B)
    return p_merge, q_yield

def payoff_and_nash(t1: float, t2: float, t3: float, t4: float, t5: float, theta: List[float]) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Legacy function from kita_game_payoff_nash.py for compatibility
    """
    t_features = (t1, t2, t3, t4, t5)
    A, B = payoff_matrices(t_features, theta)
    p_star, q_star = compute_equilibrium(A, B)
    
    a11, a12, a21, a22 = A.flatten()
    b11, b12, b21, b22 = B.flatten()
    
    return p_star, q_star, a11, a12, a21, a22, b11, b12, b21, b22

# ========================================
# 7. Data processing utilities
# ========================================

def process_data_from_csv(csv_file: str, output_file: str = 'kita_tvars.csv') -> pd.DataFrame:
    """
    Process raw data from CSV and compute t-variables
    """
    df = pd.read_csv(csv_file)
    tvars_df = compute_t_vars(df)
    tvars_df.to_csv(output_file, index=False)
    print(f'Saved t1-t5 variables to {output_file}')
    return tvars_df

def compute_nash_results_from_tvars(tvars_file: str, theta: List[float], output_file: str = 'kita_nash_results.csv') -> pd.DataFrame:
    """
    Compute Nash equilibrium results from t-variables file
    Sets t1, t3, t5 to 0 since only t2 and t4 are available.
    """
    df = pd.read_csv(tvars_file)
    results = []
    
    for _, row in df.iterrows():
        t1 = 0
        t2 = row['t2']
        t3 = 0
        t4 = row['t4']
        t5 = 0
        p_star, q_star, a11, a12, a21, a22, b11, b12, b21, b22 = payoff_and_nash(t1, t2, t3, t4, t5, theta)
        
        results.append({
            'event_id': row['event_id'],
            'p_star': p_star,
            'q_star': q_star,
            'a11': a11, 'a12': a12, 'a21': a21, 'a22': a22,
            'b11': b11, 'b12': b12, 'b21': b21, 'b22': b22,
            't1': t1, 't2': t2, 't3': t3, 't4': t4, 't5': t5,
            'merging_id': row.get('merging_id', -9999),
            'lag_id': row.get('lag_id', -9999),
            'lead_id': row.get('lead_id', -9999)
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f'Saved Nash equilibrium results to {output_file}')
    return results_df

def extract_dataset_for_fitting(tvars_file: str, merge_col: str, giveway_col: str):
    """
    Extracts dataset for model fitting from tvars_file.
    Each sample is ((t2, t4), (merge_flag, giveway_flag)).
    """
    df = pd.read_csv(tvars_file)
    dataset = []
    for _, row in df.iterrows():
        t2 = row['t2']
        t4 = row['t4']
        merge_flag = row[merge_col]
        giveway_flag = row[giveway_col]
        dataset.append(((t2, t4), (merge_flag, giveway_flag)))
    return dataset

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

# ========================================
# Example usage
# ========================================

if __name__ == "__main__":
    # Example 1: Process raw data and compute t-variables
    try:
        tvars_df = process_data_from_csv('combined_data.csv')
        print("Successfully processed data and computed t-variables")
    except FileNotFoundError:
        print("combined_data.csv not found, skipping data processing")
    
    # Feature engineering
    tvars_df = pd.read_csv('kita_tvars.csv')
    tvars_df = feature_engineering(tvars_df)
    tvars_df.to_csv('kita_tvars_fe.csv', index=False)
    
    # Feature sets to compare
    feature_sets = [
        ([], 'Base (t2, t4)'),
        (['Previous_decision', 'Time_lapsed'], 'Base + Previous_decision + Time_lapsed'),
        (['Previous_decision', 'Time_norm'], 'Base + Previous_decision + Time_norm'),
        (['Previous_decision', 'Time_log'], 'Base + Previous_decision + Time_log'),
        ([f'PrevDec_{i}' for i in [0,1,2] if f'PrevDec_{i}' in tvars_df.columns] + ['Time_bin'], 'Base + one-hot Previous_decision + Time_bin')
    ]
    
    for extra_feats, desc in feature_sets:
        print(f"\n--- Model Fitting and Validation: {desc} ---")
        # Prepare dataset
        dataset = []
        for _, row in tvars_df.iterrows():
            t2 = row['t2']
            t4 = row['t4']
            extras = [row[feat] for feat in extra_feats] if extra_feats else []
            merge_flag = row['LC']
            giveway_flag = row['Final_follower_decision']
            dataset.append(((t2, t4, *extras), (merge_flag, giveway_flag)))
        # Split train/val
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]
        # Random initial theta
        initial_theta = np.random.rand(7 + len(extra_feats)).tolist()
        print("Initial theta:", initial_theta)
        # Fit model
        fitted_theta = fit_model_with_features(train_set, extra_feats, initial_theta=initial_theta)
        print("Fitted theta:", fitted_theta)
        # Predict on train set for threshold selection
        y_true_train = []
        y_proba_train = []
        for (t2, t4, *extras), (merge_flag, _) in train_set:
            prob, _ = predict_merge_giveway_with_features(t2, t4, extras, fitted_theta)
            y_true_train.append(merge_flag)
            y_proba_train.append(prob)
        best_t, train_metrics = find_best_threshold(np.array(y_true_train), np.array(y_proba_train))
        print(f"Best threshold on train set: {best_t:.3f}")
        print(f"Train metrics: TP={train_metrics['TP']}, FP={train_metrics['FP']}, DR={train_metrics['DR']:.2f}%, FAR={train_metrics['FAR']:.2f}%")
        # Predict on validation set using best threshold
        y_true = []
        y_proba = []
        for (t2, t4, *extras), (merge_flag, _) in val_set:
            prob, _ = predict_merge_giveway_with_features(t2, t4, extras, fitted_theta)
            y_true.append(merge_flag)
            y_proba.append(prob)
        y_pred = [int(p >= best_t) for p in y_proba]
        tp, fp, dr, far = compute_metrics(y_true, y_pred)
        print(f"Validation metrics (merge decision, threshold={best_t:.3f}): TP={tp}, FP={fp}, DR={dr:.2f}%, FAR={far:.2f}%") 