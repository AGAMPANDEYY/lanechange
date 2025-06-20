import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import os

def load_data(i80_file: str, lane_change_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the I80 dataset and lane change data
    """
    # Read the I80 dataset in chunks to manage memory
    i80_chunks = pd.read_csv(i80_file, chunksize=100000)
    i80_data = pd.concat(i80_chunks)
    
    # Read the lane change data and filter for data_type = 2
    lane_change_data = pd.read_csv(lane_change_file)
    lane_change_data = lane_change_data[lane_change_data['data_type'] == 3]
    
    return i80_data, lane_change_data

def get_vehicle_data(df: pd.DataFrame, vehicle_id: int, frame_id: int) -> pd.Series:
    """
    Get all data for a specific vehicle at a specific frame
    """
    vehicle_data = df[(df['0'] == vehicle_id) & (df['1'] == frame_id)]
    return vehicle_data.iloc[0] if not vehicle_data.empty else None

def get_vehicle_trajectory(df: pd.DataFrame, vehicle_id: int, frame_id: int) -> pd.DataFrame:
    """
    Get the vehicle's trajectory for all frames before the given frame
    """
    vehicle_data = df[(df['0'] == vehicle_id) & 
                     (df['1'] <= frame_id)]
    
    # Sort in descending order of frame_id to get reverse chronological order
    return vehicle_data.sort_values('1', ascending=False)

def find_nearest_neighbors(df: pd.DataFrame, vehicle_id: int, frame_id: int, 
                          lane_id: int) -> Tuple[Dict, Dict]:
    """
    Find the nearest leading and following vehicles in the same lane
    """
    # Get the vehicle's position at the given frame
    vehicle_data = df[(df['0'] == vehicle_id) & (df['1'] == frame_id)]
    if vehicle_data.empty:
        return None, None
    
    vehicle_x = vehicle_data['4'].iloc[0]
    vehicle_y = vehicle_data['5'].iloc[0]
    
    # Get all vehicles in the same lane at the same frame
    same_lane_vehicles = df[(df['1'] == frame_id) & 
                           (df['13'] == lane_id) &
                           (df['0'] != vehicle_id)]
    
    if same_lane_vehicles.empty:
        return None, None
    
    # Calculate distances
    """
    same_lane_vehicles['distance'] = np.sqrt(
        (same_lane_vehicles['4'] - vehicle_x)**2 + 
        (same_lane_vehicles['5'] - vehicle_y)**2
    )
    """
    same_lane_vehicles['distance'] = same_lane_vehicles['5'] - vehicle_y
    
    # Find leading and following vehicles
    leading = same_lane_vehicles[same_lane_vehicles['4'] > vehicle_x]
    following = same_lane_vehicles[same_lane_vehicles['4'] < vehicle_x]
    
    lead_vehicle = leading.nsmallest(1, 'distance') if not leading.empty else None
    follow_vehicle = following.nsmallest(1, 'distance') if not following.empty else None
    
    lead_info = {
        'vehicle_id': lead_vehicle['0'].iloc[0] if lead_vehicle is not None else None,
        'spacing': lead_vehicle['distance'].iloc[0] if lead_vehicle is not None else None,
        'speed_diff': lead_vehicle['18'].iloc[0] - vehicle_data['18'].iloc[0] if lead_vehicle is not None else None,
        'data': lead_vehicle.iloc[0].to_dict() if lead_vehicle is not None else None
    } if lead_vehicle is not None else None
    
    follow_info = {
        'vehicle_id': follow_vehicle['0'].iloc[0] if follow_vehicle is not None else None,
        'spacing': follow_vehicle['distance'].iloc[0] if follow_vehicle is not None else None,
        'speed_diff': follow_vehicle['19'].iloc[0] - vehicle_data['19'].iloc[0] if follow_vehicle is not None else None,
        'data': follow_vehicle.iloc[0].to_dict() if follow_vehicle is not None else None
    } if follow_vehicle is not None else None
    
    return lead_info, follow_info

def process_lane_changes(i80_file: str, lane_change_file: str, output_file: str, averaged_output_file: str):
    """
    Main function to process lane changes and create the analysis
    """
    print("Loading data...")
    i80_data, lane_change_data = load_data(i80_file, lane_change_file)
    
    
    all_rows = []
    averaged_rows = []
    
    print("Processing lane changes...")
    for idx, row in lane_change_data.iterrows():
        vehicle_id = row['0_lane_change']
        frame_id = row['1_lane_change']
        source_lane = row['lane_no_lane_change']
        target_lane = row['from_lane_lane_change']
   
        # Get lane change vehicle data
        lc_vehicle_data = get_vehicle_data(i80_data, vehicle_id, frame_id)
            
        # Find nearest neighbors before lane change (in source lane)
        pre_lead, pre_follow = find_nearest_neighbors(i80_data, vehicle_id, frame_id, source_lane)
        
        # Find nearest neighbors after lane change (in target lane)
        post_lead, post_follow = find_nearest_neighbors(i80_data, vehicle_id, frame_id + 1, target_lane)
        
        # Get trajectories for all vehicles
        lc_trajectory = get_vehicle_trajectory(i80_data, vehicle_id, frame_id)
        
        # Create base result dictionary with lane change vehicle data
        result = {
            # Lane change vehicle data
            '0_lane_change': vehicle_id,
            '1_lane_change': frame_id,
            '2_lane_change': lc_vehicle_data['2'],  # Total Frames
            '3_lane_change': lc_vehicle_data['3'],  # Epoch Time
            '4_lane_change': lc_vehicle_data['4'],  # Local X
            '5_lane_change': lc_vehicle_data['5'],  # Local Y
            '6_lane_change': lc_vehicle_data['6'],  # Global X
            '7_lane_change': lc_vehicle_data['7'],  # Global Y
            '8_lane_change': lc_vehicle_data['8'],  # Vehicle Length
            '9_lane_change': lc_vehicle_data['9'],  # Vehicle Width
            '10_lane_change': lc_vehicle_data['10'],  # Vehicle Class
            '11_lane_change': lc_vehicle_data['11'],  # Vehicle Velocity
            '12_lane_change': lc_vehicle_data['12'],  # Vehicle Acceleration
            '13_lane_change': lc_vehicle_data['13'],  # Lane Identification
            '14_lane_change': lc_vehicle_data['14'],  # Preceding Vehicle
            '15_lane_change': lc_vehicle_data['15'],  # Following Vehicle
            '16_lane_change': lc_vehicle_data['16'],  # Spacing
            '17_lane_change': lc_vehicle_data['17'],  # Headway
            '18_lane_change': lc_vehicle_data['18'],  # Speed diff with preceding
            '19_lane_change': lc_vehicle_data['19'],  # Speed diff with following
            '20_lane_change': lc_vehicle_data['20'],  # Not sure
            '21_lane_change': lc_vehicle_data['21'],  # Lane change indicator
            '22_lane_change': lc_vehicle_data['22'],  # Corrected speed
            # Add columns from lane change data
            'Time_lapsed': row.get('Time_lapsed', None),
            'Previous_decision': row.get('Previous_decision', None),
            'Final_follower_decision': row.get('Final_follower_decision', None)
        }
        
        # Add source lane lead vehicle data
        if pre_lead and pre_lead['data']:
            for i in range(23):
                result[f'{i}_source_lead'] = pre_lead['data'].get(str(i), None)
        else:
            for i in range(23):
                result[f'{i}_source_lead'] = -9999
        
        # Add source lane follow vehicle data
        if pre_follow and pre_follow['data']:
            for i in range(23):
                result[f'{i}_source_follow'] = pre_follow['data'].get(str(i), None)
        else:
            for i in range(23):
                result[f'{i}_source_follow'] = -9999
        
        # Add target lane lead vehicle data
        if post_lead and post_lead['data']:
            for i in range(23):
                result[f'{i}_target_lead'] = post_lead['data'].get(str(i), None)
        else:
            for i in range(23):
                result[f'{i}_target_lead'] = -9999
        
        # Add target lane follow vehicle data
        if post_follow and post_follow['data']:
            for i in range(23):
                result[f'{i}_target_follow'] = post_follow['data'].get(str(i), None)
        else:
            for i in range(23):
                result[f'{i}_target_follow'] = -9999
        
        # Add the lane change event row
        all_rows.append(result)
        
        # Process trajectory data for averaging
        if not lc_trajectory.empty:
            # Get frames for T-20 to T (LC=1)
            recent_frames = lc_trajectory[lc_trajectory['1'] >= frame_id - 20]
            if not recent_frames.empty:
                avg_row = recent_frames.mean(numeric_only=True)
                avg_row['LC'] = 1
                avg_row['0_lane_change'] = vehicle_id
                avg_row['Time_lapsed'] = row.get('Time_lapsed', None)
                avg_row['Previous_decision'] = row.get('Previous_decision', None)
                avg_row['Final_follower_decision'] = row.get('Final_follower_decision', None)
                
                # Add source lane lead/follow vehicle data
                if pre_lead and pre_lead['data']:
                    for i in range(23):
                        avg_row[f'{i}_source_lead'] = pre_lead['data'].get(str(i), None)
                else:
                    for i in range(23):
                        avg_row[f'{i}_source_lead'] = -9999
                
                if pre_follow and pre_follow['data']:
                    for i in range(23):
                        avg_row[f'{i}_source_follow'] = pre_follow['data'].get(str(i), None)
                else:
                    for i in range(23):
                        avg_row[f'{i}_source_follow'] = -9999
                
                # Add target lane lead/follow vehicle data
                if post_lead and post_lead['data']:
                    for i in range(23):
                        avg_row[f'{i}_target_lead'] = post_lead['data'].get(str(i), None)
                else:
                    for i in range(23):
                        avg_row[f'{i}_target_lead'] = -9999
                
                if post_follow and post_follow['data']:
                    for i in range(23):
                        avg_row[f'{i}_target_follow'] = post_follow['data'].get(str(i), None)
                else:
                    for i in range(23):
                        avg_row[f'{i}_target_follow'] = -9999
                
                averaged_rows.append(avg_row)
            
            # Get all frames before T-20 and process in 20-frame bins
            older_frames = lc_trajectory[lc_trajectory['1'] < frame_id - 20]
            if not older_frames.empty:
                # Sort frames in descending order
                older_frames = older_frames.sort_values('1', ascending=False)
                
                # Process in 20-frame bins
                for i in range(0, len(older_frames), 20):
                    bin_frames = older_frames.iloc[i:i+20]
                    if not bin_frames.empty:
                        avg_row = bin_frames.mean(numeric_only=True)
                        avg_row['LC'] = 0
                        avg_row['0_lane_change'] = vehicle_id
                        avg_row['Time_lapsed'] = row.get('Time_lapsed', None)
                        avg_row['Previous_decision'] = row.get('Previous_decision', None)
                        avg_row['Final_follower_decision'] = row.get('Final_follower_decision', None)
                        
                        # Add source lane lead/follow vehicle data
                        if pre_lead and pre_lead['data']:
                            for i in range(23):
                                avg_row[f'{i}_source_lead'] = pre_lead['data'].get(str(i), None)
                        else:
                            for i in range(23):
                                avg_row[f'{i}_source_lead'] = -9999
                        
                        if pre_follow and pre_follow['data']:
                            for i in range(23):
                                avg_row[f'{i}_source_follow'] = pre_follow['data'].get(str(i), None)
                        else:
                            for i in range(23):
                                avg_row[f'{i}_source_follow'] = -9999
                        
                        # Add target lane lead/follow vehicle data
                        if post_lead and post_lead['data']:
                            for i in range(23):
                                avg_row[f'{i}_target_lead'] = post_lead['data'].get(str(i), None)
                        else:
                            for i in range(23):
                                avg_row[f'{i}_target_lead'] = -9999
                        
                        if post_follow and post_follow['data']:
                            for i in range(23):
                                avg_row[f'{i}_target_follow'] = post_follow['data'].get(str(i), None)
                        else:
                            for i in range(23):
                                avg_row[f'{i}_target_follow'] = -9999
                        
                        averaged_rows.append(avg_row)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_rows)
    
    # Remove duplicate rows
    results_df = results_df.drop_duplicates()
    
    # Convert averaged rows to DataFrame
    averaged_df = pd.DataFrame(averaged_rows)
    
    
    # Save to CSV files
    print("Saving results...")
    results_df.to_csv(output_file, index=False)
    averaged_df.to_csv(averaged_output_file, index=False)
    

if __name__ == "__main__":
    i80_file = r"C:\Users\USER\Downloads\I80dataset3.csv"
    lane_change_file = r"C:\Agam\Work\lanechange\Final_354_whole_data_csv_Corrected_AP.csv"
    output_file = "lane_change_analysis.csv"
    averaged_output_file = "lane_change_averaged_analysis_3.csv"
    
    process_lane_changes(i80_file, lane_change_file, output_file, averaged_output_file) 