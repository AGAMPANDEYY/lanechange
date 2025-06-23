import pandas as pd
import numpy as np

def compute_t_vars(df):
    """
    For each event (row), compute t1-t5 using the available columns and neighbor logic.
    Returns a DataFrame with columns: event_id, t1, t2, t3, t4, t5, merging_id, lag_id, second_lag_id, adj_id
    """
    results = []
    for idx, row in df.iterrows():
        merging_id = row['0']
        frame_id = row['1']
        # t1: time to end of acceleration lane (need to define distance_to_lane_end)
        # For now, set distance_to_lane_end as a constant (e.g., 100m = 328ft)
        distance_to_lane_end = 328.0
        speed_merging = row['12'] if row['12'] > 0 else 0.1  # avoid div by zero
        t1 = distance_to_lane_end / speed_merging
        # Find lag vehicle in target lane (use columns if available, else -9999)
        lag_id = row['0_target_follow'] if '0_target_follow' in row else -9999
        second_lag_id = row['0_target_follow_2'] if '0_target_follow_2' in row else -9999
        # Get positions and speeds
        x_merging = row['5']
        v_merging = row['12']
        # Lag vehicle
        if lag_id != -9999 and lag_id in df['0'].values:
            lag_row = df[(df['0'] == lag_id) & (df['1'] == frame_id)]
            if not lag_row.empty:
                x_lag = lag_row.iloc[0]['5']
                v_lag = lag_row.iloc[0]['12']
                lag_lane = lag_row.iloc[0]['13'] if '13' in lag_row.columns else np.nan
            else:
                x_lag = np.nan
                v_lag = np.nan
                lag_lane = np.nan
        else:
            x_lag = np.nan
            v_lag = np.nan
            lag_lane = np.nan
        # Second lag vehicle
        if '0_target_follow_2' in row and row['0_target_follow_2'] != -9999 and row['0_target_follow_2'] in df['0'].values:
            lag2_row = df[(df['0'] == row['0_target_follow_2']) & (df['1'] == frame_id)]
            if not lag2_row.empty:
                x_lag2 = lag2_row.iloc[0]['5']
                v_lag2 = lag2_row.iloc[0]['12']
            else:
                x_lag2 = np.nan
                v_lag2 = np.nan
        else:
            x_lag2 = np.nan
            v_lag2 = np.nan
        # t2: time to collision (merging & lag)
        if not np.isnan(x_lag) and not np.isnan(v_lag) and v_merging != v_lag:
            t2 = (x_lag - x_merging) / (v_merging - v_lag) if (v_merging - v_lag) != 0 else np.inf
        else:
            t2 = np.nan
        # t3: time to collision (merging & second lag)
        if not np.isnan(x_lag2) and not np.isnan(v_lag2) and v_merging != v_lag2:
            t3 = (x_lag2 - x_merging) / (v_merging - v_lag2) if (v_merging - v_lag2) != 0 else np.inf
        else:
            t3 = np.nan
        # t4: time headway between lag and second lag
        if not np.isnan(x_lag2) and not np.isnan(x_lag) and not np.isnan(v_lag) and v_lag != 0:
            t4 = (x_lag2 - x_lag) / v_lag
        else:
            t4 = np.nan
        # t5: time to collision of lag with adjacent lane vehicle
        adj_id = -9999
        t5 = np.nan
        if not np.isnan(x_lag) and not np.isnan(v_lag) and not np.isnan(lag_lane):
            # Find vehicles in adjacent lanes at the same frame
            possible_lanes = [lag_lane - 1, lag_lane + 1]
            adj_vehicles = df[(df['1'] == frame_id) & (df['13'].isin(possible_lanes))]
            if not adj_vehicles.empty:
                # Find the closest in position to lag vehicle
                adj_vehicles = adj_vehicles.copy()
                adj_vehicles['dist'] = np.abs(adj_vehicles['5'] - x_lag)
                adj_row = adj_vehicles.nsmallest(1, 'dist').iloc[0]
                x_adj = adj_row['5']
                v_adj = adj_row['12']
                adj_id = adj_row['0']
                if v_lag != v_adj:
                    t5 = (x_adj - x_lag) / (v_lag - v_adj) if (v_lag - v_adj) != 0 else np.inf
        results.append({
            'event_id': idx,
            't1': t1,
            't2': t2,
            't3': t3,
            't4': t4,
            't5': t5,
            'merging_id': merging_id,
            'lag_id': lag_id,
            'second_lag_id': second_lag_id,
            'adj_id': adj_id
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('combined_data.csv')
    tvars_df = compute_t_vars(df)
    tvars_df.to_csv('kita_tvars.csv', index=False)
    print('Saved t1-t5 variables to kita_tvars.csv') 