import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.metrics import f1_score

# Load dataset
df = pd.read_csv(r"C:\Agam\Work\lanechange\combined_data.csv")
ground_truth = df['LC'].values

# 1. Replace -9999 with column-wise mean
df.replace(-9999, np.nan, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# 2. Normalize selected features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["Time_lapsed"] = scaler.fit_transform(df[["Time_lapsed"]])
df["Previous_decision"] = df["Previous_decision"] / 2.0  # normalize from [0,1,2] → [0, 0.5, 1]


# Define the log-likelihood function
def neg_log_likelihood(params, df, lc_true):
    # Unpack parameters
    b0, b1, b2, alpha, sigma = params
    
    log_likelihoods = []
    
    for _, row in df.iterrows():
        v_sub = row["22"]
        v_lead = row["22_target_lead"]
        v_lag = row["22_target_follow"]
        prev_dec = row["Previous_decision"]
        time_lapsed = row["Time_lapsed"]
        
        gap_lead = row["5_target_lead"] - row["5"]
        gap_lag = row["5"] - row["5_target_follow"]

        delta_v_lead = v_lead - v_sub
        delta_v_lag = v_lag - v_sub

        for gap, delta_v in [(gap_lead, delta_v_lead), (gap_lag, delta_v_lag)]:
            if gap <= 0:
                prob = 1e-6
            else:
                mu = (
                    b0 +
                    b1 * max(delta_v, 0) +
                    b2 * min(delta_v, 0) +
                    #b3 * prev_dec +
                    #b4 * time_lapsed +
                    alpha * 0  # nu = 0
                )
                z = (np.log(gap + 1e-6) - mu) / sigma
                prob = norm.cdf(z)
                prob = np.clip(prob, 1e-6, 1 - 1e-6)

            lc = row['LC']
            if lc == 1:
                log_likelihoods.append(np.log(prob))
            else:
                log_likelihoods.append(np.log(1 - prob))
    
    return -np.sum(log_likelihoods)

# Initial guesses for [b0, b1, b2, alpha, sigma]
initial_params = [5.0, 0.05, 0.05, 0.1, 1.0]

# Bounds to ensure sigma > 0
#bounds = [(None, None)] * 4 + [(1e-3, None)]  # 4 unbounded parameters + 1 bounded parameter for sigma

bounds = [
    (0, 6),  # beta0
    (-5, 5),  # beta1
    (-5, 5),  # beta2
    #(-10, 10),  # beta3
    #(-10, 10),  # beta4
    (0, 2),     # alpha
    (0.1, 5)   # sigma
]


# Fit model
res = minimize(neg_log_likelihood, initial_params, args=(df, ground_truth), bounds=bounds)

# Extract estimated parameters
estimated_params = res.x
print("\nEstimated Parameters:")
param_names = ["beta0", "beta1", "beta2", "alpha", "sigma"]#["beta0", "beta1", "beta2", "beta3", "beta4", "alpha", "sigma"]
for name, val in zip(param_names, estimated_params):
    print(f"{name}: {val:.4f}")

# Evaluate model using F1-score on prediction
def predict_LC(row, params):
    b0, b1, b2, alpha, sigma = params
    
    v_sub = row["22"]
    v_lead = row["22_target_lead"]
    v_lag = row["22_target_follow"]
    prev_dec = row["Previous_decision"]
    time_lapsed = row["Time_lapsed"]

    gap_lead = row["5_target_lead"] - row["5"]
    gap_lag = row["5"] - row["5_target_follow"]

    delta_v_lead = v_lead - v_sub
    delta_v_lag = v_lag - v_sub

    def prob(gap, delta_v):
        if gap <= 0:
            return 0
        mu = b0 + b1 * max(delta_v, 0) + b2 * min(delta_v, 0) + alpha * 0
        z = (np.log(gap + 1e-6) - mu) / sigma
        return norm.cdf(z)
    
    P_lead = prob(gap_lead, delta_v_lead)
    P_lag  = prob(gap_lag, delta_v_lag)
    return int(P_lead * P_lag > 0.5)

# Apply prediction
df['LC_pred'] = df.apply(lambda row: predict_LC(row, estimated_params), axis=1)

# Evaluate
f1 = f1_score(ground_truth, df['LC_pred'])
print(f"\nF1 Score on predictions: {f1:.4f}")
