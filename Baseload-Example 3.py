import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = '/Users/armuaa/Documents/site-a.csv'  # Update to your correct path
df = pd.read_csv(file_path)

# Convert 'Date (Europe/Stockholm)' to datetime and set as index
df['Date'] = pd.to_datetime(df['Date (Europe/Stockholm)'], format='%d/%m/%Y %H:%M')
df.set_index('Date', inplace=True)

def filter_outliers_iqr(df, column):
    if column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[column][(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return pd.Series(dtype=float)

def analyze_base_load(df, sensor_column, rolling_window=30*24, start_date='2023-03-01',
                      end_date='2023-03-30', base_load_percentile=10):
    # Filter outliers and make into a DataFrame
    filtered_data = filter_outliers_iqr(df, sensor_column)
    sensor_data = filtered_data.to_frame()

    # Resample data to an hourly frequency to ensure time continuity, filling gaps with NaN
    sensor_data = sensor_data.resample('H').mean()
    sensor_data = sensor_data.loc[start_date:end_date]

    # Detect missing data (existing timestamps with NaN values)
    missing_data = sensor_data[sensor_column].isna()

    # Detect "No Energy Consumption" (consumption 0 or below)
    no_energy_consumption = sensor_data[sensor_column] <= 0

    # Calculate percentile-based base load, excluding zero or negative values
    valid_data_for_baseload = sensor_data[~no_energy_consumption & ~missing_data][sensor_column]
    percentile_base_load = valid_data_for_baseload.quantile(base_load_percentile / 100)
    tolerance = 0.5  # Tolerance value for calculating baseload

    # Define operational periods
    base_load_periods = (sensor_data[sensor_column] <= (percentile_base_load + tolerance)) & (~no_energy_consumption)
    production_threshold = percentile_base_load + (sensor_data[sensor_column].max() - percentile_base_load) / 3
    idle_periods = (sensor_data[sensor_column] > (percentile_base_load + tolerance)) & (sensor_data[sensor_column] <= production_threshold) & (~no_energy_consumption)
    production_periods = (sensor_data[sensor_column] > production_threshold) & (~no_energy_consumption)

    avg_percentile_base_load = percentile_base_load
    avg_base_load = valid_data_for_baseload[base_load_periods].mean()
    avg_idle = valid_data_for_baseload[idle_periods].mean()
    avg_production = valid_data_for_baseload[production_periods].mean()

    print(f"Percentile-Based Base Load ({base_load_percentile}th percentile): {avg_percentile_base_load:.2f} kWh")
    print(f"Average Base Load: {avg_base_load:.2f} kWh")
    print(f"Average Idle: {avg_idle:.2f} kWh")
    print(f"Average Production: {avg_production:.2f} kWh")

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(15, 14))

    ax[0].plot(sensor_data.index, sensor_data[sensor_column], label='Actual Data', alpha=0.7)
    ax[0].axhline(y=percentile_base_load, color='r', linestyle='--', label='Base Load Level')
    ax[0].axhline(y=production_threshold, color='orange', linestyle='--', label='Idle Level')  # Idle level line
    ax[0].fill_between(sensor_data.index, 0, sensor_data[sensor_column].max(), where=base_load_periods, 
                       facecolor='purple', alpha=0.1, label='Base Load')
    ax[0].fill_between(sensor_data.index, 0, sensor_data[sensor_column], where=idle_periods, facecolor='orange', alpha=0.3, label='Idle')
    ax[0].fill_between(sensor_data.index, 0, sensor_data[sensor_column], where=production_periods, facecolor='green', alpha=0.3, label='Production')    
    ax[0].fill_between(sensor_data.index, 0, sensor_data[sensor_column].max(), where=no_energy_consumption, 
                       facecolor='red', alpha=0.3, label='No Energy Consumption')

    # Highlight missing data
    ax[0].fill_between(sensor_data.index, 0, sensor_data[sensor_column].max(), where=missing_data,
                       facecolor='gray', alpha=0.5, label='Missing Data')

    ax[0].legend(loc='upper left')
    ax[0].set_ylabel(sensor_column)
    ax[0].set_title(f'Base Load and Operational States for {sensor_column}')

    annotation_text = (f'Base Load Level: {avg_percentile_base_load:.2f} kWh\n'
                       f'Idle Level: {production_threshold:.2f} kWh\n'  # Add this line for Idle Level
                       f'Average Base Load: {avg_base_load:.2f} kWh\n'
                       f'Average Idle: {avg_idle:.2f} kWh\n'
                       f'Average Production: {avg_production:.2f} kWh')
    
    ax[0].annotate(annotation_text, 
                   xy=(0.95, 0.95), 
                   xycoords='axes fraction', 
                   fontsize=12, 
                   ha='right', 
                   va='top',
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    ax[1].plot(sensor_data.index, base_load_periods.astype(int), label='Base Load Pattern (0-1)', color='blue')
    ax[1].legend(loc='upper left')
    ax[1].set_ylabel('Base Load')

    ax[2].plot(sensor_data.index, idle_periods.astype(int), label='Idle Pattern (0-1)', color='orange')
    ax[2].legend(loc='upper left')
    ax[2].set_ylabel('Idle')

    ax[3].plot(sensor_data.index, production_periods.astype(int), label='Production Pattern (0-1)', color='green')
    ax[3].legend(loc='upper left')
    ax[3].set_ylabel('Production')
    ax[3].set_xlabel('Date')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return base_load_periods.astype(int), idle_periods.astype(int), production_periods.astype(int)


def compare_sensor_patterns(df, sensors, rolling_window=15*24, start_date='2023-03-01', end_date='2023-3-30'):
    for sensor in sensors:
        if sensor in df.columns and not df[sensor].empty:
            print(f"Analyzing {sensor}")
            analyze_base_load(df, sensor, rolling_window, start_date, end_date)

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
