import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path to the CSV file in the Documents folder
file_path = '/Users/armuaa/Documents/site-a.csv'  # Update to your correct path

# Load the data
df = pd.read_csv(file_path)

# Convert 'Date (Europe/Stockholm)' to datetime and set as index for common usage
df['Date'] = pd.to_datetime(df['Date (Europe/Stockholm)'])
df.set_index('Date', inplace=True)

upper_threshold = 250  # Threshold value to remove outliers

def filter_outliers(df, column, threshold):
    """
    Filters out outliers from the specified column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column to filter outliers.
    threshold (float): The upper threshold above which values are considered outliers.

    Returns:
    pd.Series: Series with outliers removed.
    """
    if column in df.columns:
        return df[column][df[column] <= threshold].dropna()
    return pd.Series()

def analyze_seven_day_pattern(df, sensor_column, lookback_weeks=4, start_date='2022-12-20', end_date='2024-04-22'):
    """
    Analyze the seven-day pattern for a specific sensor.
    """
    # Filter non-null and valid sensor data and remove outliers
    sensor_data = filter_outliers(df, sensor_column, upper_threshold)
    sensor_data = sensor_data.resample('H').mean()

    # Focus on a specific date range
    sensor_data = sensor_data.loc[start_date:end_date]

    # Create weekday index
    weekday_index = sensor_data.index.to_series().dt.weekday

    # Calculate the average pattern
    grouped_means = sensor_data.groupby([weekday_index, sensor_data.index.hour]).mean()

    # Rolling average
    rolling_avg = sensor_data.rolling(window=7*24*lookback_weeks, min_periods=1).mean()

    # Identify deviations
    deviations = sensor_data - rolling_avg

    plt.figure(figsize=(15, 6))
    plt.plot(sensor_data.index, sensor_data, label='Actual Data')
    plt.plot(rolling_avg.index, rolling_avg, label=f'Rolling 7-Day Average (Lookback {lookback_weeks} Weeks)')
    plt.xlabel('Date')
    plt.ylabel(sensor_column)
    plt.title(f'Seven-Day Pattern Analysis for {sensor_column}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return deviations

def analyze_base_load(df, sensor_column, rolling_window=30*24, start_date='2022-12-20', end_date='2024-04-22'):
    """
    Analyze base load for a specific sensor using a rolling average-based approach.
    """
    # Filter and resample, remove outliers
    sensor_data = filter_outliers(df, sensor_column, upper_threshold).resample('H').mean()
    
    # Focus on a specific date range
    sensor_data = sensor_data.loc[start_date:end_date]
    
    # Determine the base load level using a rolling minimum
    rolling_min = sensor_data.rolling(window=rolling_window, min_periods=1).min()
    
    # Identify base load, idle, and production periods
    base_load_periods = sensor_data <= rolling_min
    idle_threshold = (sensor_data.max() + rolling_min) / 2
    idle_periods = (sensor_data > rolling_min) & (sensor_data <= idle_threshold)
    production_periods = sensor_data > idle_threshold

    # Create 0-1 patterns
    base_load_pattern = base_load_periods.astype(int)
    idle_pattern = idle_periods.astype(int)
    production_pattern = production_periods.astype(int)

    # Calculate average values
    avg_rolling_min = rolling_min.mean()
    avg_base_load = sensor_data[base_load_periods].mean()
    avg_idle = sensor_data[idle_periods].mean()
    avg_production = sensor_data[production_periods].mean()
    
    print(f"Average Rolling Minimum (30 days): {avg_rolling_min}")
    print(f"Average Base Load: {avg_base_load}")
    print(f"Average Idle: {avg_idle}")
    print(f"Average Production: {avg_production}")

    # Plot the main results with explicit state separation
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(15, 12))

    ax[0].plot(sensor_data.index, sensor_data, label='Actual Data')
    ax[0].plot(rolling_min.index, rolling_min, label='Base Load', linestyle='--')
    ax[0].fill_between(sensor_data.index, rolling_min, sensor_data, where=base_load_periods, facecolor='blue', alpha=0.3, label='Base Load')
    ax[0].fill_between(sensor_data.index, rolling_min, idle_threshold, where=idle_periods, facecolor='orange', alpha=0.3, label='Idle')
    ax[0].fill_between(sensor_data.index, idle_threshold, sensor_data.max(), where=production_periods, facecolor='green', alpha=0.3, label='Production')

    annotation_text = (f'Average Base Load: {avg_base_load:.2f} kWh\n'
                       f'Average Idle: {avg_idle:.2f} kWh\n'
                       f'Average Production: {avg_production:.2f} kWh')
    ax[0].annotate(annotation_text, 
                   xy=(0.95, 0.95), 
                   xycoords='axes fraction', 
                   fontsize=12, 
                   ha='right', 
                   va='top',
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    ax[0].legend(loc='upper left')
    ax[0].set_ylabel(sensor_column)
    ax[0].set_title('Base Load and Operational States for {}'.format(sensor_column))

    ax[1].plot(sensor_data.index, base_load_pattern, label='Base Load Pattern (0-1)', color='blue')
    ax[1].legend(loc='upper left')
    ax[1].set_ylabel('Base Load')

    ax[2].plot(sensor_data.index, idle_pattern, label='Idle Pattern (0-1)', color='orange')
    ax[2].legend(loc='upper left')
    ax[2].set_ylabel('Idle')

    ax[3].plot(sensor_data.index, production_pattern, label='Production Pattern (0-1)', color='green')
    ax[3].legend(loc='upper left')
    ax[3].set_ylabel('Production')
    ax[3].set_xlabel('Date')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return base_load_pattern, idle_pattern, production_pattern

def compare_sensor_patterns(df, sensors, rolling_window=30*24, start_date='2022-12-20', end_date='2024-04-22'):
    """
    Compare the operational patterns across multiple sensors.
    """
    patterns = {}
    
    for sensor in sensors:
        # Filter and analyze, continue if sensor data is non-empty
        if not df[sensor].empty:
            base_load_pattern, idle_pattern, production_pattern = analyze_base_load(df, sensor, rolling_window, start_date, end_date)
            patterns[sensor] = {
                'base_load': base_load_pattern,
                'idle': idle_pattern,
                'production': production_pattern
            }

    if not patterns:
        print("No valid data for the specified sensors within the threshold limits.")
        return

    # Combine data for comparison
    combined = pd.DataFrame(index=patterns[sensors[0]]['base_load'].index)
    
    for sensor in sensors:
        combined[f'{sensor}_base_load'] = patterns[sensor]['base_load']
        combined[f'{sensor}_idle'] = patterns[sensor]['idle']
        combined[f'{sensor}_production'] = patterns[sensor]['production']

    # Plot comparison
    plt.figure(figsize=(15, 10))

    for i, sensor in enumerate(sensors):
        plt.subplot(len(sensors), 1, i + 1)
        plt.plot(combined.index, combined[f'{sensor}_base_load'], label=f'{sensor} Base Load', alpha=0.6)
        plt.plot(combined.index, combined[f'{sensor}_idle'], label=f'{sensor} Idle', alpha=0.6)
        plt.plot(combined.index, combined[f'{sensor}_production'], label=f'{sensor} Production', alpha=0.6)
        plt.xlabel('Date')
        plt.ylabel('0-1 Patterns')
        plt.title(f'Operational Patterns for {sensor}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage:
sensors = ['Värme T1 (kWh)', '3210 - Fiberlaser (kWh)', '3211 - Laser (kWh)', '3222 - Laser (kWh)', 
           '3223 - Laser (kWh)', '3226 - Laserstans (kWh)', '3212 - laser (kWh)', '3230 - 3D laser (kWh)', 
           'GIvare ej aktiv (kWh)', '3250 - Press (kWh)', '3252 - press (kWh)', '3430 - P-stag (kWh)', 
           '3248 - Hydraulico (kWh)', 'Avfuktare T2 (kWh)', 'Avfuktare runda huset (kWh)', 
           'Avfuktare T1 (kWh)', 'Avfuktare T4 (kWh)', 'Gestamp gamla (kWh)', 'Gestamp nya (kWh)', 
           'Kompressor - S2PP (kWh)', 'Kompressor - S2QQ (kWh)', 'Kompressor - S2RR (kWh)', 
           'Kontor Berget (kWh)', 'Kontor produktion (kWh)', 'Planrikt (kWh)', 'Cataneo (kWh)', 
           'Slipline (kWh)', 'Tvätt maskin (kWh)', 'Tvätt Tranemo (kWh)', 'Värme fabrik (kWh)', 
           'Värme T2 (kWh)', 'Varmvatten vvb (kWh)']
compare_sensor_patterns(df, sensors)