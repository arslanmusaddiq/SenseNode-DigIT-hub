# Baseload Energy Analysis Scripts

These scripts are used to analyze energy consumption data from sensors. They load energy data, process it, and compute key metrics such as base load levels, idle periods, and production periods. The scripts help in understanding energy usage patterns by categorizing different states of operation and plotting these for visualization.

## Example 1

**Overall Functionality:**

- **Data Loading and Preprocessing:** Load CSV data, convert the date column to a datetime object, and use it as an index.
- **Outlier Filtering:** Remove outliers from the data using the Interquartile Range (IQR) method.
- **Analysis:** Calculate percentile-based base load levels, detect missing data, and categorize periods into base load, idle, and production.
- **Visualization:** Plot actual energy data, indicating base load levels and periods of different operational states.

This script uses a predefined tolerance to set operational thresholds and visualizes the results with plots that mark base load, idle, and production periods. It annotates plots with average values for these periods.

## Example 2

**Overall Functionality:**

- Similar to Example 1, Example 2 also loads and preprocesses the energy data.
- This script identifies missing data and "no energy consumption" periods.
- It excludes zero or negative values when determining percentile-based base load levels.
- Visualization includes a separate fill for "No Energy Consumption" periods.

In this example, special attention is given to handling zero consumption, considered indicative of inactivity, and these periods are marked distinctly in plots.

## Example 3

**Overall Functionality:**

- This script is similar to Example 2 but adds a new visualization feature, plotting an idle level line to represent transition thresholds.
- Like the previous examples, it preprocesses data and filters outliers.
- It includes an additional line in the visualization to distinguish between base load and idle periods clearly.

Example 3 introduces an idle level line to visually differentiate between base load and idle states, providing more detailed insights into operational states based on energy consumption thresholds.

---
