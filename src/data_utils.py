import os
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_methane_data():
    """
    Load and preprocess methane data.
    
    Returns:
    --------
    geopandas.GeoDataFrame or None
        Processed methane data or None if loading fails
    """
    try:
        # Define paths relative to this script
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try processed data first
        processed_data_path = os.path.join(project_dir, 'data', 'processed', 'methane_processed.csv')
        
        if os.path.exists(processed_data_path):
            # Load pre-processed data
            methane_df = pd.read_csv(processed_data_path)
            
            # Convert timestamp to datetime
            methane_df['Timestamp'] = pd.to_datetime(methane_df['Timestamp'])
            
            # Convert to GeoDataFrame
            methane_gdf = gpd.GeoDataFrame(
                methane_df,
                geometry=gpd.points_from_xy(methane_df['Longitude'], methane_df['Latitude']),
                crs="EPSG:4326"
            )
            
            return methane_gdf
        
        # If processed data doesn't exist, try raw data
        methane_path = os.path.join(project_dir, 'data', 'methane_sensors.csv')
        
        # For testing with direct paths (if needed)
        if not os.path.exists(methane_path):
            methane_path = r"C:\Users\pradeep dubey\Downloads\methane_sensors.csv"
        
        # Load and preprocess data
        from src.data_processing import preprocess_methane_data
        methane_df = pd.read_csv(methane_path)
        methane_gdf = preprocess_methane_data(methane_df)
        
        return methane_gdf
        
    except Exception as e:
        print(f"Error loading methane data: {e}")
        return None

def load_wind_data():
    """
    Load and preprocess wind data.
    
    Returns:
    --------
    pandas.DataFrame or None
        Processed wind data or None if loading fails
    """
    try:
        # Define paths relative to this script
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try processed data first
        processed_data_path = os.path.join(project_dir, 'data', 'processed', 'wind_processed.csv')
        
        if os.path.exists(processed_data_path):
            # Load pre-processed data
            wind_df = pd.read_csv(processed_data_path)
            
            # Convert timestamp to datetime
            wind_df['Timestamp'] = pd.to_datetime(wind_df['Timestamp'])
            
            return wind_df
        
        # If processed data doesn't exist, try raw data
        wind_path = os.path.join(project_dir, 'data', 'wind_data.csv')
        
        # For testing with direct paths (if needed)
        if not os.path.exists(wind_path):
            wind_path = r"C:\Users\pradeep dubey\Downloads\wind_data.csv"
        
        # Load and preprocess data
        from src.data_processing import load_data, preprocess_wind_data
        _, wind_df = load_data(None, wind_path)
        wind_df_processed = preprocess_wind_data(wind_df)
        
        return wind_df_processed
        
    except Exception as e:
        print(f"Error loading wind data: {e}")
        return None

def load_all_data():
    """
    Load and process all data.
    
    Returns:
    --------
    tuple
        (methane_gdf, wind_df, merged_gdf) : Processed data
    """
    # Get methane and wind data
    methane_gdf = load_methane_data()
    wind_df = load_wind_data()
    
    # Merge data
    from src.data_processing import merge_data
    merged_gdf = merge_data(methane_gdf, wind_df)
    
    return methane_gdf, wind_df, merged_gdf

def convert_wind_to_uv(speed, direction):
    """
    Convert wind speed and direction to U and V components.
    
    Parameters:
    -----------
    speed : float or array-like
        Wind speed in m/s
    direction : float or array-like
        Wind direction in degrees (meteorological convention)
        
    Returns:
    --------
    tuple
        (u, v) components of wind vector
    """
    # Convert direction from meteorological convention (0=N, 90=E)
    # to mathematical convention (0=E, 90=N)
    rad_direction = np.radians((270 - direction) % 360)
    
    # Calculate components
    u = speed * np.cos(rad_direction)
    v = speed * np.sin(rad_direction)
    
    return u, v

def create_time_features(df, time_col='Timestamp'):
    """
    Create time-based features from timestamp column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a timestamp column
    time_col : str
        Name of the timestamp column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added time features
    """
    df = df.copy()
    
    # Make sure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract features
    df['Hour'] = df[time_col].dt.hour
    df['Day'] = df[time_col].dt.dayofweek
    df['Month'] = df[time_col].dt.month
    df['Year'] = df[time_col].dt.year
    
    # Cyclical encoding of hour 
    # (helps models understand that hour 23 is close to hour 0)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
    
    # Cyclical encoding of day of week
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day']/7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day']/7)
    
    return df

def add_spatial_features(gdf):
    """
    Add spatial features to the GeoDataFrame.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometry column
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with added spatial features
    """
    gdf = gdf.copy()
    
    # Calculate center point
    center_x = gdf.geometry.x.mean()
    center_y = gdf.geometry.y.mean()
    
    # Calculate distance from center
    gdf['Distance_From_Center'] = np.sqrt(
        (gdf.geometry.x - center_x)**2 + 
        (gdf.geometry.y - center_y)**2
    )
    
    # Calculate bearing from center (in radians)
    gdf['Bearing_From_Center'] = np.arctan2(
        gdf.geometry.y - center_y,
        gdf.geometry.x - center_x
    )
    
    return gdf

def filter_data_by_time(gdf, start_time=None, end_time=None, time_col='Timestamp'):
    """
    Filter data by time range.
    
    Parameters:
    -----------
    gdf : pandas.DataFrame or geopandas.GeoDataFrame
        DataFrame with timestamp column
    start_time : str or datetime, optional
        Start time to filter from
    end_time : str or datetime, optional
        End time to filter to
    time_col : str
        Name of timestamp column
        
    Returns:
    --------
    pandas.DataFrame or geopandas.GeoDataFrame
        Filtered dataframe
    """
    filtered_gdf = gdf.copy()
    
    # Convert to datetime if string
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)
    
    # Filter by time range
    if start_time is not None:
        filtered_gdf = filtered_gdf[filtered_gdf[time_col] >= start_time]
    if end_time is not None:
        filtered_gdf = filtered_gdf[filtered_gdf[time_col] <= end_time]
    
    return filtered_gdf

def aggregate_data_by_time(gdf, freq='1H', agg_funcs=None, time_col='Timestamp'):
    """
    Aggregate data by time frequency.
    
    Parameters:
    -----------
    gdf : pandas.DataFrame
        DataFrame with timestamp column
    freq : str
        Frequency to aggregate by (e.g. '1H' for hourly)
    agg_funcs : dict, optional
        Dictionary of column:aggregation_function
    time_col : str
        Name of timestamp column
        
    Returns:
    --------
    pandas.DataFrame
        Aggregated dataframe
    """
    df = pd.DataFrame(gdf.drop(columns='geometry') if 'geometry' in gdf.columns else gdf)
    
    # Set timestamp as index
    df = df.set_index(time_col)
    
    # Default aggregation functions
    if agg_funcs is None:
        agg_funcs = {
            'Methane_Concentration (ppm)': ['mean', 'std', 'min', 'max'],
            'Wind_Speed (m/s)': 'mean',
            'Wind_Direction (°)': lambda x: np.mean(np.exp(1j*np.radians(x))).angle() * 180 / np.pi
        }
    
    # Resample and aggregate
    aggregated = df.resample(freq).agg(agg_funcs)
    
    # Reset index
    aggregated = aggregated.reset_index()
    
    return aggregated

def normalize_features(df, features=None, scaler=None):
    """
    Normalize selected features using StandardScaler.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features to normalize
    features : list, optional
        List of features to normalize, if None normalize all numeric
    scaler : sklearn.preprocessing.StandardScaler, optional
        Pre-fitted scaler to use, if None create a new one
        
    Returns:
    --------
    tuple
        (normalized_df, scaler)
    """
    df_norm = df.copy()
    
    # If no features specified, use all numeric
    if features is None:
        features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create new scaler if not provided
    if scaler is None:
        scaler = StandardScaler()
        df_norm[features] = scaler.fit_transform(df[features])
    else:
        df_norm[features] = scaler.transform(df[features])
    
    return df_norm, scaler