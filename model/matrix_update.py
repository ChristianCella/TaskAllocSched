import numpy as np
import pandas as pd


"""
Update Human Time Matrix

This module is specifically designed for integrating simulated task durations with realistically generated observational times.
The key objective is to create a robust time prediction model that simulates potential real-world variations across different times of the day.
The process involves:
1. Generating "real" time data based on existing simulation times, simulating how long tasks might actually take under various daily conditions.
2. Updating the simulated times with these generated "real" times using a weighted average approach to create a more realistic forecast of task durations.

The generated times are not purely random but are based on the simulated human times, adjusted to reflect possible real-world variations due to factors like worker fatigue, differing conditions at different times of the day, etc.
"""

# Generate random times based on simulated human times to model potential real-world variability.
# Note: In actual deployment, this simulated data will be replaced with actual real-world times from human task performance.

def generate_random_times_based_on_existing(human_task):

    # Time bands to simulate different times of the day
    time_bands = ['0-3h', '3-6h', '6h+']
    # Multipliers for each time band to simulate longer tasks during the day
    base_multipliers = [1.0, 1.1, 1.2]  
    # Variations for each time band to simulate random fluctuations
    variations = [0.1, 0.1, 0.1]  

    rows = []
    for _, row in human_task.iterrows():
        base_time = row['th']
        
        for time_band, base_multiplier, variation in zip(time_bands, base_multipliers, variations):
            # Generate two random times for each time band
            for _ in range(2):  
                modified_time = base_time * base_multiplier * (1 + np.random.uniform(-variation, variation))
                rows.append({
                    'Task_ID': row['ID'],
                    'Task': row['TASK'],
                    'Time_Band': time_band,
                    'Time': round(modified_time, 2)
                })
    
    return pd.DataFrame(rows)

# Update the human_task matrix by integrating real-world human performance times with simulated human times.
# This function uses weighted averages to combine actual observed times with simulated base times, enhancing accuracy.
# In practice, this approach will help adjust simulated predictions based on actual performance metrics.

def initialize_time_bands(human_task):

    # Initialize time bands for each task
    time_bands = ['0-3h', '3-6h', '6h+']
    for band in time_bands:
        human_task[f'th_{band}'] = human_task['th']

    return human_task

# Reshape the human task times to match the structure of the real times
# This function reshapes the human task times to match the structure of the real times, with separate columns for each time band.
def reshape_human_task_times(human_task):
    
    reshaped = []

    for time_band in ['0-3h', '3-6h', '6h+']:

        temp = human_task[['Task_ID', f'th_{time_band}']].rename(columns={f'th_{time_band}': 'Simulated_Time'})
        temp['Time_Band'] = time_band
        reshaped.append(temp)

    return pd.concat(reshaped, ignore_index=True)


def aggregate_real_times(real_times, column='Time'):

    """
    Calculates the mean real times for each Task_ID and Time_Band, removing outliers.
    
    Args:
    real_times (pd.DataFrame): DataFrame containing real times for each Task_ID and Time_Band.
    column (str): Name of the column containing the times to be analyzed.

    Returns:
    pd.DataFrame: Aggregated DataFrame with mean times computed excluding outliers.
    """
    # Applying IQR filter to remove outliers

    # Calculate the first quartile (25th percentile)
    Q1 = real_times[column].quantile(0.25) 

    # Calculate the third quartile (75th percentile) 
    Q3 = real_times[column].quantile(0.75)  

    # Interquartile Range (IQR)
    IQR = Q3 - Q1  

    # Lower boundary for outlier detection
    lower_bound = Q1 - 1.5 * IQR  

    # Upper boundary for outlier detection
    upper_bound = Q3 + 1.5 * IQR 

    # Filter out outliers
    filtered_real_times = real_times[(real_times[column] >= lower_bound) & (real_times[column] <= upper_bound)]  

    # Aggregating the clean times
    # Group by Task_ID and Time_Band, then calculate mean
    aggregated = filtered_real_times.groupby(['Task_ID', 'Time_Band'])[column].mean().reset_index()  
    # Rename the aggregated column to 'Real_Time'
    aggregated.rename(columns={column: 'Real_Time'}, inplace=True)  
    return aggregated


def sensitivity_analysis(simulated_times, real_times):
    """
    Analizza la discrepanza tra i tempi simulati e quelli reali per determinare i pesi appropriati per ogni task e fascia oraria.
    """
    merged_df = pd.merge(simulated_times, real_times, on=['Task_ID', 'Time_Band'])
    
    # Evaluate the discrepancy between simulated and real times
    merged_df['Discrepancy'] = abs(merged_df['Simulated_Time'] - merged_df['Real_Time']) / merged_df['Simulated_Time']
    
    # Evalutate the weights based on the discrepancy
    merged_df['Weight_Real'] = merged_df['Discrepancy'].apply(lambda x: min(0.8, 0.4 + 0.4 * x))
    merged_df['Weight_Simulated'] = 1 - merged_df['Weight_Real']
    
    return merged_df[['Task_ID', 'Time_Band', 'Weight_Real', 'Weight_Simulated']]



# Function to update the human task matrix
def update_human_tasks_with_weighted_times(simulated_times, weights_df, real_times, original_data):
    # Merge the simulated times, weights, and real times
    merged = pd.merge(simulated_times, weights_df, on=['Task_ID', 'Time_Band'])
    merged = pd.merge(merged, real_times, on=['Task_ID', 'Time_Band'])
    
    # Evalutate the updated times based on the weighted average
    merged['Updated_Time'] = (
        merged['Simulated_Time'] * merged['Weight_Simulated'] +
        merged['Real_Time'] * merged['Weight_Real']
    )
    
    # Pivot to get the updated times in separate columns
    pivot = merged.pivot_table(index='Task_ID', columns='Time_Band', values='Updated_Time').reset_index()
    pivot.columns = ['Task_ID', 'th_0-3h', 'th_3-6h', 'th_6h+']

    # Merge of the original data with the updated times
    final = pd.merge(original_data, pivot, on='Task_ID', how='left')
    
    # Substituting NaN values with the original simulated times
    final['th_0-3h'] = final['th_0-3h_y'].fillna(final['th_0-3h_x'])
    final['th_3-6h'] = final['th_3-6h_y'].fillna(final['th_3-6h_x'])
    final['th_6h+'] = final['th_6h+_y'].fillna(final['th_6h+_x'])

    # Cancel the unnecessary columns
    final.drop(['th_0-3h_x', 'th_0-3h_y', 'th_3-6h_x', 'th_3-6h_y', 'th_6h+_x', 'th_6h+_y'], axis=1, inplace=True)

    return final

