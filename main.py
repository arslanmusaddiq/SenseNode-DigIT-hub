import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path to the CSV file in the Documents folder
file_path = '/Users/armuaa/Documents/site-a.csv'  # Update to your correct path

# Load the data
df = pd.read_csv(file_path)

# 1. Seven-day period analysis
def analyze_seven_day_pattern(df, sensor_column, lookback_weeks=4):
    # Convert 'Date (Europe/Stockholm)' to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date (Europe/Stockholm)'])
    df.set_index('Date', inplace=True)
    
    # Filter non-null sensor data
    sensor_data = df[sensor_column].dropna()

    # Exclude data points with values higher than 200
    sensor_data = sensor_data[sensor_data <= 200]

    # Resampling the data to hourly frequency to decrease clutter (if needed)
    sensor_data = sensor_data.resample('H').mean()

    # Focus on a realistic date range (if applicable)
    start_date = '2022-12-20'
    end_date = '2023-01-31'
    sensor_data = sensor_data.loc[start_date:end_date]

    # Create weekday index
    weekday_index = sensor_data.index.to_series().dt.weekday

    # Calculate the average pattern for a 7-day period using the last `lookback_weeks` weeks
    grouped_means = sensor_data.groupby([weekday_index, sensor_data.index.hour]).mean()

    # Rolling average over lookback weeks
    rolling_avg = sensor_data.rolling(window=7*24*lookback_weeks, min_periods=1).mean()

    # Identify deviations
    deviations = sensor_data - rolling_avg

    plt.figure(figsize=(15, 6))
    plt.plot(sensor_data.index, sensor_data, label='Actual Data')
    plt.plot(rolling_avg.index, rolling_avg, label='Rolling 7-Day Average (Lookback {} Weeks)'.format(lookback_weeks))
    plt.xlabel('Date')
    plt.ylabel(sensor_column)
    plt.title('Seven-Day Pattern Analysis for {}'.format(sensor_column))
    plt.legend()
    plt.grid(True)
    plt.show()

    return deviations

# 2. Base load analysis
def analyze_base_load(df, sensor_column, threshold=200, start_date='2022-12-20', end_date='2023-01-31'):
    # Convert 'Date (Europe/Stockholm)' to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date (Europe/Stockholm)'])
    df.set_index('Date', inplace=True)
    
    # Filter and resample
    sensor_data = df[sensor_column].dropna()
    sensor_data = sensor_data[sensor_data <= threshold]
    sensor_data = sensor_data.resample('H').mean()
    
    # Focus on a specific date range
    sensor_data = sensor_data.loc[start_date:end_date]
    
    # Detect base load
    rolling_min = sensor_data.rolling(window=30*24, min_periods=1).min()
    
    # Define idle threshold
    idle_threshold = (sensor_data.max() + rolling_min) / 2
    
    # Identify periods
    base_load_periods = sensor_data <= rolling_min
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
    
    # Plotting the results (with more explicit state separation)
    plt.figure(figsize=(15, 6))
    plt.plot(sensor_data.index, sensor_data, label='Actual Data')
    plt.plot(rolling_min.index, rolling_min, label='Base Load (30 Days Rolling Min)', linestyle='--')
    plt.fill_between(sensor_data.index, sensor_data.min(), rolling_min, where=base_load_periods, facecolor='blue', alpha=0.3, label='Base Load')
    plt.fill_between(sensor_data.index, rolling_min, idle_threshold, where=idle_periods, facecolor='orange', alpha=0.3, label='Idle')
    plt.fill_between(sensor_data.index, idle_threshold, sensor_data.max(), where=production_periods, facecolor='green', alpha=0.3, label='Production')
    plt.xlabel('Date')
    plt.ylabel(sensor_column)
    plt.title('Base Load and Operational States for {}'.format(sensor_column))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return base_load_pattern, idle_pattern, production_pattern

# 3. Identify state of operation
def identify_operational_states(sensor_data, rolling_period=30*24):
    # Exclude data points with values higher than 200
    sensor_data = sensor_data[sensor_data <= 200]

    base_load = sensor_data.rolling(window=rolling_period, min_periods=1).min()

    idle_threshold = (sensor_data.max() + base_load) / 2
    idle_periods = (sensor_data <= idle_threshold) & (sensor_data > base_load)
    production_periods = sensor_data > idle_threshold

    return base_load, idle_periods, production_periods

def visualize_states(sensor_data, base_load, idle_periods, production_periods, sensor_column):
    plt.figure(figsize=(15, 6))
    plt.plot(sensor_data.index, sensor_data, label='Actual Data')
    plt.plot(base_load.index, base_load, label='Base Load (30 Days Rolling Min)', linestyle='--')
    plt.fill_between(sensor_data.index, base_load, idle_periods, where=idle_periods, facecolor='orange', alpha=0.5, label='Idle')
    plt.fill_between(sensor_data.index, idle_threshold, sensor_data.max(), where=production_periods, facecolor='green', alpha=0.5, label='Production')
    plt.xlabel('Date')
    plt.ylabel(sensor_column)
    plt.title('Operational States for {}'.format(sensor_column))
    plt.legend()
    plt.grid(True)
    plt.show()

# 4. Compare Patterns Across Sensors
def compare_sensor_patterns(df, sensors, threshold=200, start_date='2022-12-20', end_date='2023-01-31'):
    patterns = {}
    
    for sensor in sensors:
        base_load, idle, production = analyze_base_load(df, sensor, threshold, start_date, end_date)
        patterns[sensor] = {
            'base_load': base_load,
            'idle': idle,
            'production': production
        }
    
    # Combine data for comparison
    combined = pd.DataFrame(index=patterns[sensors[0]]['base_load'].index)
    
    for sensor in sensors:
        combined[f'{sensor}_base_load'] = patterns[sensor]['base_load']
        combined[f'{sensor}_idle'] = patterns[sensor]['idle']
        combined[f'{sensor}_production'] = patterns[sensor]['production']
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    for sensor in sensors:
        plt.subplot(len(sensors), 1, sensors.index(sensor) + 1)
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
sensors = ['3210 - Fiberlaser (kWh)', '3211 - Laser (kWh)', '3222 - Laser (kWh)']
compare_sensor_patterns(df, sensors)

# Specifically analyze and plot the seven-day pattern for one sensor
deviations_3210 = analyze_seven_day_pattern(df, '3210 - Fiberlaser (kWh)')
