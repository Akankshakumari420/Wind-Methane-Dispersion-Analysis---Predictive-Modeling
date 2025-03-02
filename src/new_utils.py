import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from datetime import datetime

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return directory

def resolve_paths(base_path=None, filename=None, default_path=None):
    """Resolve file paths with fallbacks for convenience"""
    if base_path and filename:
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            return path
    
    # Fallback to common locations
    common_locations = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', filename),
        os.path.join(r"C:\Users\pradeep dubey\Downloads", filename) if os.name == 'nt' else None
    ]
    
    for location in common_locations:
        if location and os.path.exists(location):
            return location
    
    # Return default as last resort
    return default_path

def save_figure(fig, filename, output_dir=None, dpi=300, close=True):
    """Save matplotlib figure with proper directory handling"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    
    ensure_directory(output_dir)
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if close:
        plt.close(fig)
    
    return output_path

def create_timestamp_formatter():
    """Returns a formatter function for timestamps in visualizations"""
    def format_timestamp(ts):
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        if isinstance(ts, (pd.Timestamp, datetime)):
            return ts.strftime('%Y-%m-%d %H:%M')
        return str(ts)
    
    return format_timestamp

def create_colormap(values, cmap_name='viridis', alpha=0.7):
    """Create a standardized colormap for visualizations"""
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    vmin, vmax = np.min(values), np.max(values)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name, 256)
    
    def get_color(value):
        """Get rgba color for a value"""
        rgba = list(cmap(norm(value)))
        rgba[3] = alpha  # Set alpha
        return rgba
    
    return {
        'get_color': get_color,
        'cmap': cmap,
        'norm': norm,
        'vmin': vmin,
        'vmax': vmax,
        'to_hex': lambda value: colors.rgb2hex(cmap(norm(value)))
    }

def convert_wind_to_uv(wind_speed, wind_direction):
    """
    Convert wind speed and direction to U and V components
    
    Parameters:
    -----------
    wind_speed : float or array-like
        Wind speed magnitude
    wind_direction : float or array-like
        Wind direction in meteorological degrees (0=N, 90=E, 180=S, 270=W)
    
    Returns:
    --------
    tuple
        (u, v) components where u is eastward wind and v is northward wind
    """
    rad_dir = np.radians((270 - wind_direction) % 360)
    u = wind_speed * np.cos(rad_dir)
    v = wind_speed * np.sin(rad_dir)
    return u, v

def folium_map_with_markers(gdf, value_col, center_coords=None, radius=10, 
                          tooltip_template="{id}: {value:.2f}", zoom_start=13):
    """
    Create a folium map with circle markers colored by values
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometry column containing Point geometries
    value_col : str
        Column name for values to display and color
    center_coords : tuple, optional
        (lat, lon) for map center. If None, uses mean of points
    radius : int
        Radius for circle markers
    tooltip_template : str
        Template for tooltips with {id} and {value} placeholders
    zoom_start : int
        Initial zoom level
    
    Returns:
    --------
    folium.Map
        Map with markers added
    """
    # Determine center of map
    if center_coords is None:
        center_lat = gdf.geometry.y.mean()
        center_lon = gdf.geometry.x.mean()
    else:
        center_lat, center_lon = center_coords
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    
    # Create colormap for values
    values = gdf[value_col]
    vmin, vmax = values.min(), values.max()
    
    colormap = folium.LinearColormap(
        colors=['green', 'yellow', 'orange', 'red'],
        vmin=vmin, vmax=vmax,
        caption=value_col
    )
    m.add_child(colormap)
    
    # Add markers
    for idx, row in gdf.iterrows():
        color = colormap(row[value_col])
        tooltip = tooltip_template.format(id=row.get('Sensor_ID', idx), value=row[value_col])
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            color='black',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=tooltip
        ).add_to(m)
    
    return m

def add_time_features(df, time_col='Timestamp'):
    """
    Add time-based features to dataframe for modeling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with timestamp column
    time_col : str
        Name of timestamp column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added time features
    """
    result = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[time_col]):
        result[time_col] = pd.to_datetime(result[time_col])
    
    # Extract components
    result['hour'] = result[time_col].dt.hour
    result['day'] = result[time_col].dt.day
    result['month'] = result[time_col].dt.month
    result['weekday'] = result[time_col].dt.dayofweek
    result['is_weekend'] = (result['weekday'] >= 5).astype(int)
    
    # Cyclical features
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    result['weekday_sin'] = np.sin(2 * np.pi * result['weekday'] / 7)
    result['weekday_cos'] = np.cos(2 * np.pi * result['weekday'] / 7)
    
    return result
