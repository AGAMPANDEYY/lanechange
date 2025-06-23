import pandas as pd
import numpy as np

def payoff_and_nash(t1, t2, t3, t4, t5, theta):
    # Unpack theta
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = theta
    # Payoff matrix for merging vehicle (P1)
    a11 = theta1 * t3 + theta2         # merge, giveway
    a12 = theta1 * t2 + theta2         # merge, no giveway
    a21 = a22 = theta3 * t1            # pass (wait)
    # Payoff matrix for through vehicle (P2)
    b11 = b21 = theta4 * t5 + theta5   # giveway
    b12 = theta6 * t2                  # no giveway, merge
    b22 = theta7 * t4                  # no giveway, pass
    # 2x2 payoff matrices
    A = np.array([[a11, a12], [a21, a22]])
    B = np.array([[b11, b12], [b21, b22]])
    # Closed-form Nash equilibrium for 2x2 game
    # P1: row player, P2: column player
    # p* = prob P1 plays row 0 (merge), q* = prob P2 plays col 0 (giveway)
    # q* = (a22 - a12) / (a11 - a12 - a21 + a22)
    # p* = (b22 - b21) / (b11 - b12 - b21 + b22)
    denom1 = (a11 - a12 - a21 + a22)
    denom2 = (b11 - b12 - b21 + b22)
    q_star = (a22 - a12) / denom1 if denom1 != 0 else 0.5
    p_star = (b22 - b21) / denom2 if denom2 != 0 else 0.5
    # Clamp to [0,1]
    q_star = np.clip(q_star, 0, 1)
    p_star = np.clip(p_star, 0, 1)
    return p_star, q_star, a11, a12, a21, a22, b11, b12, b21, b22

if __name__ == "__main__":
    # Load t1-t5 data
    df = pd.read_csv('kita_tvars.csv')
    # Placeholder theta values (to be estimated later)
    theta = [1, 1, 1, 1, 1, 1, 1]
    results = []
    for _, row in df.iterrows():
        t1, t2, t3, t4, t5 = row['t1'], row['t2'], row['t3'], row['t4'], row['t5']
        p_star, q_star, a11, a12, a21, a22, b11, b12, b21, b22 = payoff_and_nash(t1, t2, t3, t4, t5, theta)
        results.append({
            'event_id': row['event_id'],
            'p_star': p_star,
            'q_star': q_star,
            'a11': a11, 'a12': a12, 'a21': a21, 'a22': a22,
            'b11': b11, 'b12': b12, 'b21': b21, 'b22': b22,
            't1': t1, 't2': t2, 't3': t3, 't4': t4, 't5': t5
        })
    pd.DataFrame(results).to_csv('kita_nash_results.csv', index=False)
    print('Saved Nash equilibrium results to kita_nash_results.csv') 