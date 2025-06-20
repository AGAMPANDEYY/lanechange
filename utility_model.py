import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Critical gap model using probabilistic gap acceptance ---
def compute_critical_gap_parameters():
    return {
        'lead': {
            'beta0': 5.12,
            'beta1': 0.065,  # max(\u0394V)
            'beta2': 0.044,  # min(\u0394V)
            'alpha': 0.12,
            'sigma': 1.51
        },
        'lag': {
            'beta0': 4.23,
            'beta1': -0.115,  # max(\u0394V)
            'alpha': 0.128,
            'sigma': 1.03
        }
    }

def compute_critical_gap(delta_v, params, nu=0):
    mean_log_gap = (
        params['beta0'] +
        params['beta1'] * max(delta_v, 0) +
        (params.get('beta2', 0) * min(delta_v, 0)) +
        params['alpha'] * nu
    )
    return np.exp(mean_log_gap)  # Convert back from log-space

def compute_gap_acceptance_probability(gap, delta_v, params, nu=0):
    if gap <= 0:
        return 0.0  # definitely not acceptable
    mean = (
        params['beta0'] +
        params['beta1'] * max(delta_v, 0) +
        (params.get('beta2', 0) * min(delta_v, 0)) +
        params['alpha'] * nu
    )
    z = (np.log(gap) - mean) / params['sigma']
    return norm.cdf(z) if gap > 0 else 0

# --- Load dataset ---
df = pd.read_csv(r"C:\\Agam\\Work\\lanechange\\combined_data.csv")

# --- Calculate acceptance probability row-wise ---
params = compute_critical_gap_parameters()

def apply_gap_acceptance(row):
    v_sub = row["22"]
    v_lead = row["22_target_lead"]
    v_lag = row["22_target_follow"]

    gap_lead = row['5_target_lead'] - row['5']
    gap_lag = row['5'] - row['5_target_follow']

    delta_v_lead = v_lead - v_sub
    delta_v_lag = v_lag - v_sub

    critical_lead = compute_critical_gap(delta_v_lead, params['lead'])
    critical_lag = compute_critical_gap(delta_v_lag, params['lag'])

    P_lead = compute_gap_acceptance_probability(gap_lead, delta_v_lead, params['lead'])
    P_lag = compute_gap_acceptance_probability(gap_lag, delta_v_lag, params['lag'])

    LC_prob = P_lead * P_lag
    LC = int(LC_prob > 0.5)

    return pd.Series({
        "gap_lead": gap_lead,
        "gap_lag": gap_lag,
        "G_lead": critical_lead,
        "G_lag": critical_lag,
        "P_lead": P_lead,
        "P_lag": P_lag,
        "LC": LC
    })

df[["gap_lead", "gap_lag", "G_lead", "G_lag", "P_lead", "P_lag", "LC"]] = df.apply(apply_gap_acceptance, axis=1)

# --- Save result ---
df.to_csv("with_prob_gap_acceptance.csv", index=False)

# --- Evaluation ---
ground_truth = pd.read_csv(r"C:\\Agam\\Work\\lanechange\\combined_data.csv")['LC']

accuracy = accuracy_score(ground_truth, df['LC'])
precision = precision_score(ground_truth, df['LC'])
recall = recall_score(ground_truth, df['LC'])
f1 = f1_score(ground_truth, df['LC'])
conf_matrix = confusion_matrix(ground_truth, df['LC'])

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClass Distribution:")
print("Ground Truth:")
print(ground_truth.value_counts(normalize=True))
print("\nModel Predictions:")
print(df['LC'].value_counts(normalize=True))
