import pandas as pd
import os

def combine_csv_files():
    # List of CSV files to combine
    csv_files = [
        r'C:\Agam\Work\lanechange\lane_change_averaged_analysis_1.csv',
        r'C:\Agam\Work\lanechange\lane_change_averaged_analysis_2.csv',
        r'C:\Agam\Work\lanechange\lane_change_averaged_analysis_3.csv',
    ]
    
    # Create an empty list to store dataframes
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded {file} with {len(df)} rows")
        else:
            print(f"Warning: {file} not found")
    
    if not dfs:
        print("No CSV files were found to combine")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save the combined dataframe to a new CSV file
    output_file = 'combined_data.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined data saved to {output_file}")
    print(f"Total rows in combined file: {len(combined_df)}")

if __name__ == "__main__":
    combine_csv_files() 