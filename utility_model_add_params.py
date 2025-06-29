import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# =============================================================================
# 1) Utility functions for "11-parameter" version (no Previous_decision/Time_lapsed)
#    We estimate 10 parameters total:
#      lead:   [β0, β1, β2, α, logσ_lead]
#      lag:    [γ0, γ1, γ2, α', logσ_lag]
# =============================================================================

def prob_gap_11(
    gap, delta_v, v_sub,
    b0, b1, b2, alpha, sigma
):
    """
    11-param: 
      P(gap acceptable) = Φ( [ ln(gap) − (b0 + b1·max(Δv,0) + b2·min(Δv,0) + α·v_sub ) ] / σ ).
      If gap <= 0 => return 0.
    """
    if gap <= 0:
        return 0.0
    lin = b0 + b1 * max(delta_v, 0) + b2 * min(delta_v, 0) + alpha * v_sub
    z = (np.log(gap + 1e-8) - lin) / sigma
    return norm.cdf(z)

def neg_log_likelihood_11(theta, df):
    """
    Negative log-likelihood for the 11-param model.
    theta vector (length = 10):
      [b0_l, b1_l, b2_l, alpha_l, logσ_l,
       b0_g, b1_g, b2_g, alpha_g, logσ_g]
    """
    # Unpack
    b0_l, b1_l, b2_l, a_l, logσ_l = theta[:5]
    b0_g, b1_g, b2_g, a_g,  logσ_g = theta[5:10]

    σ_l = np.exp(logσ_l)
    σ_g = np.exp(logσ_g)

    # Pre-extract arrays
    gap_lead_arr   = (df['5_target_lead'].values - df['5'].values)
    gap_lag_arr    = (df['5'].values - df['5_target_follow'].values)
    dvel_lead_arr  = (df['22_target_lead'].values - df['22'].values)
    dvel_lag_arr   = (df['22_target_follow'].values - df['22'].values)
    v_sub_arr      = df['22'].values
    LC_true        = df['LC'].values.astype(int)

    ll_sum = 0.0
    eps = 1e-12
    N = len(df)

    for i in range(N):
        gl = gap_lead_arr[i]
        dl = dvel_lead_arr[i]
        vs = v_sub_arr[i]

        P_l_i = prob_gap_11(gl, dl, vs, b0_l, b1_l, b2_l, a_l, σ_l)
        P_g_i = prob_gap_11(
            gap_lag_arr[i], dvel_lag_arr[i], vs,
            b0_g, b1_g, b2_g, a_g, σ_g
        )

        joint = P_l_i * P_g_i
        joint = np.clip(joint, eps, 1 - eps)

        if LC_true[i] == 1:
            ll_sum += np.log(joint)
        else:
            ll_sum += np.log(1 - joint)

    return -ll_sum  # negative log-likelihood

def recompute_preds_11(df, theta):
    """
    Given df and fitted theta (length=10), compute
    P_lead_11, P_lag_11, and LC_pred_11 for each row.
    Returns a new DataFrame with these three extra columns.
    """
    print(f"Computing predictions for {len(df)} test samples...")
    
    (b0_l, b1_l, b2_l, a_l, logσ_l,
     b0_g, b1_g, b2_g, a_g,  logσ_g) = theta

    σ_l = np.exp(logσ_l)
    σ_g = np.exp(logσ_g)

    P_l_list = []
    P_g_list = []
    LCp_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing 11-param predictions"):
        vs = float(row['22'])
        gl = float(row['5_target_lead'] - row['5'])
        dl = float(row['22_target_lead'] - row['22'])
        gg = float(row['5'] - row['5_target_follow'])
        dg = float(row['22_target_follow'] - row['22'])

        P_l_i = prob_gap_11(gl, dl, vs, b0_l, b1_l, b2_l, a_l, σ_l)
        P_g_i = prob_gap_11(gg, dg, vs, b0_g, b1_g, b2_g, a_g, σ_g)

        joint = P_l_i * P_g_i
        lc_pred = int(joint > 0.5)

        P_l_list.append(P_l_i)
        P_g_list.append(P_g_i)
        LCp_list.append(lc_pred)

    out = df.copy()
    out['P_lead_11']  = P_l_list
    out['P_lag_11']   = P_g_list
    out['LC_pred_11'] = LCp_list
    return out

# =============================================================================
# 2) Utility functions for "13-parameter" version (includes PrevDecision & Time_lapsed)
#    We estimate 14 parameters:
#      lead: [β0, β1, β2, β3, β4, α, logσ]
#      lag:  [γ0, γ1, γ2, γ3, γ4, α', logσ]
# =============================================================================

def prob_gap_13(
    gap, delta_v, prev_dec, time_lapsed, v_sub,
    b0, b1, b2, b3, b4, alpha, sigma
):
    """
    13-param gap acceptance formula:
      If gap <= 0 => return 0.
      Else P = Φ(
         [ ln(gap) − (b0 + b1·max(Δv,0) + b2·min(Δv,0)
                       + b3·prev_dec + b4·time_lapsed + alpha·v_sub ) ] / sigma
      )
    """
    if gap <= 0:
        return 0.0

    lin = (
        b0
        + b1 * max(delta_v, 0)
        + b2 * min(delta_v, 0)
        + b3 * prev_dec
        + b4 * time_lapsed
        + alpha * v_sub
    )
    z = (np.log(gap + 1e-8) - lin) / sigma
    return norm.cdf(z)

def neg_log_likelihood_13(theta, df):
    """
    Negative log-likelihood for the 13-param model.
    theta: length-14:
      [b0_l, b1_l, b2_l, b3_l, b4_l, alpha_l, logσ_l,
       b0_g, b1_g, b2_g, b3_g, b4_g, alpha_g, logσ_g]
    """
    b0_l, b1_l, b2_l, b3_l, b4_l, a_l, logσ_l = theta[:7]
    b0_g, b1_g, b2_g, b3_g, b4_g, a_g, logσ_g = theta[7:14]

    σ_l = np.exp(logσ_l)
    σ_g = np.exp(logσ_g)

    gap_lead_arr    = (df['5_target_lead'].values - df['5'].values)
    gap_lag_arr     = (df['5'].values - df['5_target_follow'].values)
    dvel_lead_arr   = (df['22_target_lead'].values - df['22'].values)
    dvel_lag_arr    = (df['22_target_follow'].values - df['22'].values)
    prev_arr        = df['Previous_decision'].values.astype(float)
    time_arr        = df['Time_lapsed'].values.astype(float)
    v_sub_arr       = df['22'].values
    LC_true         = df['LC'].values.astype(int)

    ll_sum = 0.0
    eps = 1e-12
    N = len(df)

    for i in range(N):
        gl = gap_lead_arr[i]
        dl = dvel_lead_arr[i]
        pd = prev_arr[i]
        tl = time_arr[i]
        vs = v_sub_arr[i]

        P_l_i = prob_gap_13(gl, dl, pd, tl, vs,
                            b0_l, b1_l, b2_l, b3_l, b4_l, a_l, σ_l)

        gg = gap_lag_arr[i]
        dg = dvel_lag_arr[i]

        P_g_i = prob_gap_13(gg, dg, pd, tl, vs,
                            b0_g, b1_g, b2_g, b3_g, b4_g, a_g, σ_g)

        joint = P_l_i * P_g_i
        joint = np.clip(joint, eps, 1 - eps)

        if LC_true[i] == 1:
            ll_sum += np.log(joint)
        else:
            ll_sum += np.log(1 - joint)

    return -ll_sum

def recompute_preds_13(df, theta):
    """
    Given df and fitted theta (length=14), compute
    P_lead_13, P_lag_13, and LC_pred_13 for each row.
    Returns a new DataFrame with these three extra columns.
    """
    print(f"Computing predictions for {len(df)} test samples...")
    
    (b0_l, b1_l, b2_l, b3_l, b4_l, a_l, logσ_l,
     b0_g, b1_g, b2_g, b3_g, b4_g, a_g, logσ_g) = theta

    σ_l = np.exp(logσ_l)
    σ_g = np.exp(logσ_g)

    P_l_list = []
    P_g_list = []
    LCp_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing 13-param predictions"):
        vs = float(row['22'])
        gl = float(row['5_target_lead'] - row['5'])
        dl = float(row['22_target_lead'] - row['22'])
        gg = float(row['5'] - row['5_target_follow'])
        dg = float(row['22_target_follow'] - row['22'])
        pd = float(row['Previous_decision'])
        tl = float(row['Time_lapsed'])

        P_l_i = prob_gap_13(gl, dl, pd, tl, vs,
                            b0_l, b1_l, b2_l, b3_l, b4_l, a_l, σ_l)
        P_g_i = prob_gap_13(gg, dg, pd, tl, vs,
                            b0_g, b1_g, b2_g, b3_g, b4_g, a_g, σ_g)

        joint = P_l_i * P_g_i
        lc_pred = int(joint > 0.5)

        P_l_list.append(P_l_i)
        P_g_list.append(P_g_i)
        LCp_list.append(lc_pred)

    out = df.copy()
    out['P_lead_13']  = P_l_list
    out['P_lag_13']   = P_g_list
    out['LC_pred_13'] = LCp_list
    return out

# =============================================================================
# 3) DRIVER BLOCK: load data, split, run four scenarios, print table
# =============================================================================

# 3.1) Load CSV
df = pd.read_csv(r"/content/lanechange/combined_data.csv")

# Ensure Previous_decision is numeric
if df['Previous_decision'].dtype == object:
    df['Previous_decision'] = df['Previous_decision'].astype('category').cat.codes

# Drop rows missing any of these columns
needed_cols = [
    '22','22_target_lead','22_target_follow',
    '5','5_target_lead','5_target_follow',
    'Previous_decision','Time_lapsed','LC'
]
df = df.dropna(subset=needed_cols).reset_index(drop=True)

# 3.2) Train/test split (70/30 stratified on LC)
df_train, df_test = train_test_split(
    df, test_size=0.3, stratify=df['LC'], random_state=42
)

# For SMOTE, we only need the following numeric columns from df_train:
smote_feats = [
    '22','22_target_lead','22_target_follow',
    '5','5_target_lead','5_target_follow',
    'Previous_decision','Time_lapsed'
]
X_train_sm = df_train[smote_feats].values
y_train_sm = df_train['LC'].values.astype(int)

# 3.3) Functions to fit MLE under bounds

def fit_mle_11(train_df):
    """
    Fit the 11-param model on train_df via MLE with bounded parameters.
    Returns the fitted theta (length=10).
    """
    print(f"Fitting 11-parameter MLE model on {len(train_df)} training samples...")
    
    # Initial guess (near Toledo's WS values)
    init10 = np.array([
        # lead:    b0,    b1,    b2,   alpha, logσ
         5.12,  0.065,  0.044,  0.12,  np.log(1.51),
        # lag:     b0,     b1,     b2,    alpha, logσ
         4.23, -0.115,  0.0,    0.128, np.log(1.03)
    ])

    # Bounds for each of the 10 parameters:
    #  b0,γ0 in [−10, +10]
    #  b1,b2,γ1,γ2 in [−5, +5]
    #  alpha, alpha' in [−1, +1]
    #  logσ in [ln(0.1), ln(5)]
    bnds = [
        (-10, 10),   # b0_lead
        (-5, 5),     # b1_lead
        (-5, 5),     # b2_lead
        (-1, 1),     # alpha_lead
        (np.log(0.1), np.log(5)),  # logσ_lead

        (-10, 10),   # b0_lag
        (-5, 5),     # b1_lag
        (-5, 5),     # b2_lag
        (-1, 1),     # alpha_lag
        (np.log(0.1), np.log(5))   # logσ_lag
    ]

    print("Starting optimization with L-BFGS-B method...")
    res = minimize(
        fun=neg_log_likelihood_11,
        x0=init10,
        args=(train_df,),
        method='L-BFGS-B',
        bounds=bnds,
        options={'disp': True, 'maxiter': 500}  # Reduced maxiter and enabled disp
    )
    if not res.success:
        print("WARNING: 11-param MLE did not converge:", res.message)
        print("Using best available result...")
    else:
        print(f"11-param MLE converged successfully in {res.nit} iterations")
    return res.x

def fit_mle_13(train_df):
    """
    Fit the 13-param model on train_df via MLE with bounded parameters.
    Returns the fitted theta (length=14).
    """
    print(f"Fitting 13-parameter MLE model on {len(train_df)} training samples...")
    
    init14 = np.array([
        # lead:    b0,    b1,    b2,   b3,    b4,    alpha,  logσ
         5.12,  0.065,  0.044, 0.1,  -0.005,  0.12,   np.log(1.51),
        # lag:     b0,    b1,    b2,   b3,    b4,    alpha,  logσ
         4.23, -0.115,  0.0,   0.1,  -0.005,  0.128,  np.log(1.03)
    ])

    # Bounds for each of the 14 parameters:
    #  b0,γ0       in [−10, +10]
    #  b1,b2,γ1,γ2 in [−5, +5]
    #  b3,γ3       in [−5, +5]
    #  b4,γ4       in [−5, +5]
    #  alpha,alpha' in [−1, +1]
    #  logσ       in [ln(0.1), ln(5)]
    bnds = [
        (-10, 10),  # b0_lead
        (-5, 5),    # b1_lead
        (-5, 5),    # b2_lead
        (-5, 5),    # b3_lead (PrevDec)
        (-5, 5),    # b4_lead (Time)
        (-1, 1),    # alpha_lead
        (np.log(0.1), np.log(5)),  # logσ_lead

        (-10, 10),  # b0_lag
        (-5, 5),    # b1_lag
        (-5, 5),    # b2_lag
        (-5, 5),    # b3_lag (PrevDec)
        (-5, 5),    # b4_lag (Time)
        (-1, 1),    # alpha_lag
        (np.log(0.1), np.log(5))   # logσ_lag
    ]

    print("Starting optimization with L-BFGS-B method...")
    res = minimize(
        fun=neg_log_likelihood_13,
        x0=init14,
        args=(train_df,),
        method='L-BFGS-B',
        bounds=bnds,
        options={'disp': True, 'maxiter': 500}  # Reduced maxiter and enabled disp
    )
    if not res.success:
        print("WARNING: 13-param MLE did not converge:", res.message)
        print("Using best available result...")
    else:
        print(f"13-param MLE converged successfully in {res.nit} iterations")
    return res.x

# =============================================================================
# 3) RUNNING THE FOUR SCENARIOS:
#    (A) 11-param WITHOUT SMOTE
#    (B) 11-param WITH SMOTE
#    (C) 13-param WITHOUT SMOTE
#    (D) 13-param WITH SMOTE
# =============================================================================

# 3.1) Prepare SMOTE inputs from df_train
X_train_sm = df_train[smote_feats].values
y_train_sm = df_train['LC'].values.astype(int)

# 3.2) Scenario A: 11-param WITHOUT SMOTE
theta11_ws = fit_mle_11(df_train)
df_test_11_ws = recompute_preds_11(df_test, theta11_ws)

cm11_ws = confusion_matrix(df_test['LC'], df_test_11_ws['LC_pred_11'])
tn11_ws, fp11_ws, fn11_ws, tp11_ws = cm11_ws.ravel()
N11_ws   = len(df_test)
LC11_ws  = int(df_test['LC'].sum())
NLC11_ws = N11_ws - LC11_ws
DR11_ws  = 100 * tp11_ws / (tp11_ws + fn11_ws) if (tp11_ws + fn11_ws) > 0 else 0.0
FAR11_ws = 100 * fp11_ws / (fp11_ws + tn11_ws) if (fp11_ws + tn11_ws) > 0 else 0.0

# 3.3) Scenario B: 11-param WITH SMOTE
sm = SMOTE(random_state=42, k_neighbors=1)
X_res11, y_res11 = sm.fit_resample(X_train_sm, y_train_sm)
train_sm_11 = pd.DataFrame(X_res11, columns=smote_feats)
train_sm_11['LC'] = y_res11

theta11_s = fit_mle_11(train_sm_11)
df_test_11_s = recompute_preds_11(df_test, theta11_s)

cm11_s = confusion_matrix(df_test['LC'], df_test_11_s['LC_pred_11'])
tn11_s, fp11_s, fn11_s, tp11_s = cm11_s.ravel()
N11_s   = len(df_test)
LC11_s  = int(df_test['LC'].sum())
NLC11_s = N11_s - LC11_s
DR11_s  = 100 * tp11_s / (tp11_s + fn11_s) if (tp11_s + fn11_s) > 0 else 0.0
FAR11_s = 100 * fp11_s / (fp11_s + tn11_s) if (fp11_s + tn11_s) > 0 else 0.0

# 3.4) Scenario C: 13-param WITHOUT SMOTE
theta13_ws = fit_mle_13(df_train)
df_test_13_ws = recompute_preds_13(df_test, theta13_ws)

cm13_ws = confusion_matrix(df_test['LC'], df_test_13_ws['LC_pred_13'])
tn13_ws, fp13_ws, fn13_ws, tp13_ws = cm13_ws.ravel()
N13_ws   = len(df_test)
LC13_ws  = int(df_test['LC'].sum())
NLC13_ws = N13_ws - LC13_ws
DR13_ws  = 100 * tp13_ws / (tp13_ws + fn13_ws) if (tp13_ws + fn13_ws) > 0 else 0.0
FAR13_ws = 100 * fp13_ws / (fp13_ws + tn13_ws) if (fp13_ws + tn13_ws) > 0 else 0.0

# 3.5) Scenario D: 13-param WITH SMOTE
# Use the same train_sm_11 (it has all 8 numeric columns + 'LC')
train_sm_13 = train_sm_11.copy()

theta13_s = fit_mle_13(train_sm_13)
df_test_13_s = recompute_preds_13(df_test, theta13_s)

cm13_s = confusion_matrix(df_test['LC'], df_test_13_s['LC_pred_13'])
tn13_s, fp13_s, fn13_s, tp13_s = cm13_s.ravel()
N13_s   = len(df_test)
LC13_s  = int(df_test['LC'].sum())
NLC13_s = N13_s - LC13_s
DR13_s  = 100 * tp13_s / (tp13_s + fn13_s) if (tp13_s + fn13_s) > 0 else 0.0
FAR13_s = 100 * fp13_s / (fp13_s + tn13_s) if (fp13_s + tn13_s) > 0 else 0.0

# =============================================================================
# 3.6) Print the consolidated table
# =============================================================================
print("\n" + "="*82)
print(f"{'Model / Sampling':<22} | {'#Rows':>6} | {'LC':>4} | {'NLC':>4} | {'TP':>4} | {'FP':>4} | {'DR (%)':>7} | {'FAR (%)':>7}")
print("-"*82)

print(f"{'11 params (WS)':<22} | "
      f"{N11_ws:>6} | "
      f"{LC11_ws:>4} | "
      f"{NLC11_ws:>4} | "
      f"{tp11_ws:>4} | "
      f"{fp11_ws:>4} | "
      f"{DR11_ws:>7.2f} | "
      f"{FAR11_ws:>7.2f}")

print(f"{'11 params (S)':<22} | "
      f"{N11_s:>6} | "
      f"{LC11_s:>4} | "
      f"{NLC11_s:>4} | "
      f"{tp11_s:>4} | "
      f"{fp11_s:>4} | "
      f"{DR11_s:>7.2f} | "
      f"{FAR11_s:>7.2f}")

print(f"{'13 params (WS)':<22} | "
      f"{N13_ws:>6} | "
      f"{LC13_ws:>4} | "
      f"{NLC13_ws:>4} | "
      f"{tp13_ws:>4} | "
      f"{fp13_ws:>4} | "
      f"{DR13_ws:>7.2f} | "
      f"{FAR13_ws:>7.2f}")

print(f"{'13 params (S)':<22} | "
      f"{N13_s:>6} | "
      f"{LC13_s:>4} | "
      f"{NLC13_s:>4} | "
      f"{tp13_s:>4} | "
      f"{fp13_s:>4} | "
      f"{DR13_s:>7.2f} | "
      f"{FAR13_s:>7.2f}")

print("="*82 + "\n")
