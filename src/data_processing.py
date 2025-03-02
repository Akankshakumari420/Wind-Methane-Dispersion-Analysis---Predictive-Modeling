import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(methane_path, wind_path):
    """
    Load methane and wind data from CSV files.
    
    Parameters:
    -----------
    methane_path : str
        Path to the methane sensor data CSV file
    wind_path : str
        Path to the wind data CSV file
    
    Returns:
    --------
    tuple
        (methane_df, wind_df) : Loaded pandas DataFrames
    """
    print(f"Loading methane data from: {methane_path}")
    methane_df = pd.read_csv(methane_path)
    
    print(f"Loading wind data from: {wind_path}")
    wind_df = pd.read_csv(wind_path)
    
    # Convert timestamps to datetime
    methane_df['Timestamp'] = pd.to_datetime(methane_df['Timestamp'])
    wind_df['Timestamp'] = pd.to_datetime(wind_df['Timestamp'])
    
    # Basic data info
    print(f"Loaded {len(methane_df)} methane readings from {methane_df['Sensor_ID'].nunique()} sensors")
    print(f"Loaded {len(wind_df)} wind measurements")
    print(f"Time range: {methane_df['Timestamp'].min()} to {methane_df['Timestamp'].max()}")
    
    return methane_df, wind_df

def preprocess_methane_data(methane_df):
    """
    Preprocess methane sensor data.
    
    Parameters:
    -----------
    methane_df : pandas.DataFrame
        DataFrame containing methane sensor data
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Preprocessed GeoDataFrame with geometry
    """
    print("Preprocessing methane sensor data...")
    
    # Create a copy to avoid modifying the original
    df = methane_df.copy()
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values in methane data")
        # Fill missing values or drop rows as needed
        df = df.dropna()
        print(f"After handling missing values: {len(df)} records remain")
    
    # Create shapely Point geometries from coordinates
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Sort by timestamp and sensor ID
    gdf = gdf.sort_values(['Timestamp', 'Sensor_ID'])
    
    # Check for outliers in methane concentration
    q1 = gdf['Methane_Concentration (ppm)'].quantile(0.25)
    q3 = gdf['Methane_Concentration (ppm)'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    outliers = gdf[gdf['Methane_Concentration (ppm)'] > upper_bound]
    
    if len(outliers) > 0:
        print(f"Found {len(outliers)} potential outliers in methane concentrations")
        # We keep

    return gdf

def preprocess_wind_data(wind_df):
    """
    Preprocess the wind data:
    1. Convert timestamp to datetime
    2. Check for missing values
    3. Add wind vector components for visualization
    
    Parameters:
    -----------
    wind_df : pandas.DataFrame
        Raw wind data
    
    Returns:
    --------
    pandas.DataFrame
        Processed wind data with additional columns
    """
    print("Preprocessing wind data...")
    
    # Make a copy to avoid modifying the original dataframe
    df = wind_df.copy()
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Missing values detected:\n{missing_values[missing_values > 0]}")
        # Fill missing values if needed
        df = df.dropna()  # or use appropriate imputation method
    
    # Convert wind direction from degrees to radians for vector calculations
    # Note: Wind direction in meteorology is where the wind is coming FROM
    # For vector calculations, we need where the wind is going TO (opposite direction)
    df['Wind_Direction_Rad'] = np.radians((df['Wind_Direction (°)'] + 180) % 360)
    
    # Calculate wind vector components (U: West-East, V: South-North)
    # Note: In meteorology, U is positive for eastward wind, V is positive for northward wind
    df['U'] = -df['Wind_Speed (m/s)'] * np.sin(df['Wind_Direction_Rad'])
    df['V'] = -df['Wind_Speed (m/s)'] * np.cos(df['Wind_Direction_Rad'])
    
    print(f"Processed {len(df)} wind records with vector components")
    
    return df

def merge_data(methane_gdf, wind_df):
    """
    Merge methane and wind data based on timestamp.
    
    Parameters:
    -----------
    methane_gdf : geopandas.GeoDataFrame
        Processed methane sensor data
    wind_df : pandas.DataFrame
        Processed wind data
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Merged dataframe containing both methane and wind data
    """
    print("Merging methane and wind data...")
    
    # Merge on timestamp
    merged_gdf = methane_gdf.merge(wind_df, on='Timestamp')
    
    # Check that merge was successful
    n_methane_times = methane_gdf['Timestamp'].nunique()
    n_wind_times = wind_df['Timestamp'].nunique()
    n_merged_times = merged_gdf['Timestamp'].nunique()
    
    print(f"Unique timestamps in methane data: {n_methane_times}")
    print(f"Unique timestamps in wind data: {n_wind_times}")
    print(f"Unique timestamps in merged data: {n_merged_times}")
    
    if n_merged_times < min(n_methane_times, n_wind_times):
        print("Warning: Some timestamps were lost in the merge!")
    
    return merged_gdf

def basic_data_summary(methane_gdf, wind_df, merged_gdf=None):
    """
    Print a basic summary of the data and generate exploratory plots.
    
    Parameters:
    -----------
    methane_gdf : geopandas.GeoDataFrame
        Processed methane sensor data
    wind_df : pandas.DataFrame
        Processed wind data
    merged_gdf : geopandas.GeoDataFrame, optional
        Merged dataframe containing both methane and wind data
    """
    print("\n--- Methane Data Summary ---")
    print(f"Number of sensors: {methane_gdf['Sensor_ID'].nunique()}")
    print(f"Time range: {methane_gdf['Timestamp'].min()} to {methane_gdf['Timestamp'].max()}")
    print(f"Methane concentration range: {methane_gdf['Methane_Concentration (ppm)'].min():.2f} to {methane_gdf['Methane_Concentration (ppm)'].max():.2f} ppm")
    
    # Summary statistics for methane data
    methane_stats = methane_gdf.groupby('Sensor_ID')['Methane_Concentration (ppm)'].agg(['mean', 'std', 'min', 'max'])
    print("\nMethane concentration statistics by sensor:")
    print(methane_stats)
    
    print("\n--- Wind Data Summary ---")
    print(f"Time range: {wind_df['Timestamp'].min()} to {wind_df['Timestamp'].max()}")
    print(f"Wind speed range: {wind_df['Wind_Speed (m/s)'].min():.2f} to {wind_df['Wind_Speed (m/s)'].max():.2f} m/s")
    print(f"Wind direction range: {wind_df['Wind_Direction (°)'].min():.2f}° to {wind_df['Wind_Direction (°)'].max():.2f}°")
    
    # Basic plots
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Methane concentration over time for all sensors
    plt.subplot(1, 2, 1)
    for sensor in methane_gdf['Sensor_ID'].unique():
        sensor_data = methane_gdf[methane_gdf['Sensor_ID'] == sensor]
        plt.plot(sensor_data['Timestamp'], sensor_data['Methane_Concentration (ppm)'], 
                 marker='.', linestyle='-', alpha=0.7, label=sensor if sensor in ['S1', 'S2', 'S3'] else "")
    
    plt.title('Methane Concentration Over Time')
    plt.xlabel('Time')
    plt.ylabel('Methane Concentration (ppm)')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    
    # Plot 2: Wind speed and direction over time
    ax = plt.subplot(1, 2, 2)
    ax.plot(wind_df['Timestamp'], wind_df['Wind_Speed (m/s)'], marker='o', linestyle='-', color='blue')
    plt.title('Wind Speed Over Time')
    plt.xlabel('Time')
    plt.ylabel('Wind Speed (m/s)', color='blue')
    plt.xticks(rotation=45)
    
    # Add wind direction on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(wind_df['Timestamp'], wind_df['Wind_Direction (°)'], marker='x', linestyle='--', color='red')
    ax2.set_ylabel('Wind Direction (°)', color='red')
    
    plt.tight_layout()
    
    # Save the plots
    output_dir = '../outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'basic_data_summary.png'))
    print(f"\nBasic summary plots saved to {os.path.join(output_dir, 'basic_data_summary.png')}")
    
    # Close to free up memory
    plt.close()

def save_processed_data(methane_gdf, wind_df, merged_gdf, output_dir='../data/processed'):
    """
    Save the processed dataframes to files.
    
    Parameters:
    -----------
    methane_gdf : geopandas.GeoDataFrame
        Processed methane sensor data
    wind_df : pandas.DataFrame
        Processed wind data
    merged_gdf : geopandas.GeoDataFrame
        Merged dataframe containing both methane and wind data
    output_dir : str
        Directory to save the processed data files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed files
    methane_gdf.to_file(os.path.join(output_dir, 'methane_processed.geojson'), driver='GeoJSON')
    wind_df.to_csv(os.path.join(output_dir, 'wind_processed.csv'), index=False)
    merged_gdf.to_file(os.path.join(output_dir, 'merged_data.geojson'), driver='GeoJSON')
    
    print(f"\nProcessed data saved to {output_dir}")

def main():
    """
    Main function to run the data processing pipeline.
    """
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    methane_path = os.path.join(project_dir, 'data', 'methane_sensors.csv')
    wind_path = os.path.join(project_dir, 'data', 'wind_data.csv')
    
    # For testing with direct paths (if needed)
    if not os.path.exists(methane_path):
        methane_path = r"C:\Users\pradeep dubey\Downloads\methane_sensors.csv"
        wind_path = r"C:\Users\pradeep dubey\Downloads\wind_data.csv"
    
    # Load data
    methane_df, wind_df = load_data(methane_path, wind_path)
    
    # Process data
    methane_gdf = preprocess_methane_data(methane_df)
    wind_df_processed = preprocess_wind_data(wind_df)
    
    # Merge datasets
    merged_gdf = merge_data(methane_gdf, wind_df_processed)
    
    # Generate summary
    basic_data_summary(methane_gdf, wind_df_processed, merged_gdf)
    
    # Save processed data
    processed_dir = os.path.join(project_dir, 'data', 'processed')
    save_processed_data(methane_gdf, wind_df_processed, merged_gdf, processed_dir)
    
    return methane_gdf, wind_df_processed, merged_gdf

if __name__ == "__main__":
    print("Running data processing pipeline...")
    methane_gdf, wind_df, merged_gdf = main()
    print("Data processing completed successfully!")
