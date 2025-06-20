import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# Try SMOTE
try:
    from imblearn.over_sampling import SMOTE
    has_smote = True
except ImportError:
    has_smote = False
# Tabulate for clear tables
from tabulate import tabulate

# ----------------------------
# 1) VECTORIZED GAP-ACCEPTANCE PROBABILITY
# ----------------------------
def vectorized_prob_gap(gap, delta_v, prev_dec, time_lapsed, v_sub, params):
    lin = (
        params['b0']
        + params['b1'] * np.maximum(delta_v, 0)
        + params['b2'] * np.minimum(delta_v, 0)
        + params.get('b3', 0) * prev_dec
        + params.get('b4', 0) * time_lapsed
        + params['alpha'] * v_sub
    )
    safe_gap = np.maximum(gap, 1e-8)
    z = (np.log(safe_gap) - lin) / params['sigma']
    p = norm.cdf(z)
    p[gap <= 0] = 0.0
    return p

# ----------------------------
# 2) WEIGHTED NEG LOG-LIKELIHOOD
# ----------------------------
def neg_log_likelihood(theta, df, version='11-param', w_pos=5.0):
    if version == '11-param':
        init_len, lead_len = 10, 5
        keys = ['b0','b1','b2','alpha','logσ']
    else:
        init_len, lead_len = 14, 7
        keys = ['b0','b1','b2','b3','b4','alpha','logσ']
    lag_len = lead_len
    assert len(theta) == init_len
    lead = dict(zip(keys, theta[:lead_len]))
    lag  = dict(zip(keys, theta[lead_len:lead_len+lag_len]))
    lead['sigma'], lag['sigma'] = np.exp(lead.pop('logσ')), np.exp(lag.pop('logσ'))
    gl = df['5_target_lead'] - df['5']
    gg = df['5'] - df['5_target_follow']
    dl = df['22_target_lead'] - df['22']
    dg = df['22_target_follow'] - df['22']
    vs = df['22']
    prev = df['Previous_decision'] if 'Previous_decision' in df.columns else 0
    time = df['Time_lapsed']      if 'Time_lapsed'    in df.columns else 0
    P_l = vectorized_prob_gap(gl.values, dl.values,
                                pd.Series(prev).astype(float).values,
                                pd.Series(time).astype(float).values,
                                vs.values, lead)
    P_g = vectorized_prob_gap(gg.values, dg.values,
                                pd.Series(prev).astype(float).values,
                                pd.Series(time).astype(float).values,
                                vs.values, lag)
    joint = np.clip(P_l * P_g, 1e-12, 1-1e-12)
    LC = df['LC'].astype(int).values
    ll = np.sum(w_pos * LC * np.log(joint) + (1 - LC) * np.log(1 - joint))
    return -ll

# ----------------------------
# 3) FITTING FUNCTION
# ----------------------------
def fit_mle(df, version='11-param', w_pos=5.0):
    if version == '11-param':
        init = np.array([5.12,0.065,0.044,0.12,np.log(1.51),
                         4.23,-0.115,0.0,0.128,np.log(1.03)])
        bounds = [(-10,10),(-5,5),(-5,5),(-1,1),(np.log(0.1),np.log(5))] * 2
    else:
        init = np.array([5.12,0.065,0.044,0.1,-0.005,0.12,np.log(1.51),
                         4.23,-0.115,0.0,0.1,-0.005,0.128,np.log(1.03)])
        bounds = [(-10,10),(-5,5),(-5,5),(-5,5),(-5,5),(-1,1),(np.log(0.1),np.log(5))] * 2
    res = minimize(lambda t: neg_log_likelihood(t, df, version, w_pos), init,
                   method='L-BFGS-B', bounds=bounds)
    if not res.success:
        print(f"MLE failed for {version}: {res.message}")
    return res.x

# ----------------------------
# 4) RUN SCENARIOS & DISPLAY TABLE
# ----------------------------
if __name__ == '__main__':
    df = pd.read_csv(r'C:\Agam\Work\lanechange\combined_data.csv')
    needed = ['22','22_target_lead','22_target_follow','5','5_target_lead','5_target_follow','LC']
    if 'Previous_decision' in df.columns:
        needed += ['Previous_decision','Time_lapsed']
    df = df.dropna(subset=needed)
    df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['LC'], random_state=42)
    scenarios = [
        ('11-param', False),
        ('11-param', True),
        ('13-param', False),
        ('13-param', True),
    ]
    table = []
    for version, use_smote in scenarios:
        train = df_train.copy()
        if use_smote and has_smote:
            feats = ['22','22_target_lead','22_target_follow','5','5_target_lead','5_target_follow']
            if 'Previous_decision' in train:
                feats += ['Previous_decision','Time_lapsed']
            sm = SMOTE(random_state=42)
            Xr, yr = sm.fit_resample(train[feats], train['LC'])
            train = pd.concat([pd.DataFrame(Xr, columns=feats), pd.Series(yr, name='LC')], axis=1)
        theta = fit_mle(train, version, w_pos=5.0)
        # Predictions
        gl = df_test['5_target_lead'] - df_test['5']
        gg = df_test['5'] - df_test['5_target_follow']
        dl = df_test['22_target_lead'] - df_test['22']
        dg = df_test['22_target_follow'] - df_test['22']
        vs = df_test['22']
        prev = df_test['Previous_decision'] if 'Previous_decision' in df_test else 0
        time = df_test['Time_lapsed']      if 'Time_lapsed' in df_test else 0
        lead_len = 5 if version=='11-param' else 7
        keys = ['b0','b1','b2','alpha','logσ'] if version=='11-param' else ['b0','b1','b2','b3','b4','alpha','logσ']
        lead = dict(zip(keys, theta[:lead_len]))
        lag  = dict(zip(keys, theta[lead_len:lead_len+lead_len]))
        lead['sigma'], lag['sigma'] = np.exp(lead.pop('logσ')), np.exp(lag.pop('logσ'))
        P_l = vectorized_prob_gap(gl.values, dl.values, pd.Series(prev).astype(float).values, pd.Series(time).astype(float).values, vs.values, lead)
        P_g = vectorized_prob_gap(gg.values, dg.values, pd.Series(prev).astype(float).values, pd.Series(time).astype(float).values, vs.values, lag)
        joint = P_l * P_g
        y_pred = (joint > 0.2).astype(int)
        tn, fp, fn, tp = confusion_matrix(df_test['LC'], y_pred).ravel()
        dr = tp/(tp+fn)*100 if tp+fn else 0
        far = fp/(fp+tn)*100 if fp+tn else 0
        coeffs = np.round(theta,4)
        table.append([version, 'SMOTE' if use_smote else 'No SMOTE', coeffs.tolist(), tp, fp, f"{dr:.2f}", f"{far:.2f}"])
    headers = ['Model', 'SMOTE', 'Coefficients', 'TP', 'FP', 'DR%', 'FAR%']
    print(tabulate(table, headers=headers, tablefmt='github'))
