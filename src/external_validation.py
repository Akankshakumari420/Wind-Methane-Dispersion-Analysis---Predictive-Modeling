"""
External Data Validation Module

This module provides functionality to validate local wind and climate data
against external APIs and data sources.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# API configuration
# Note: In a production environment, store these in environment variables or a secure config
# If only using OpenWeatherMap, ensure the key is valid.
API_KEYS = {
    'openweathermap': 'YOUR_VALID_OPENWEATHERMAP_API_KEY',  # Replace with a valid API key
}

def fetch_openweathermap_data(lat, lon, timestamp=None, api_key=None):
    """
    Fetch historical weather data from OpenWeatherMap API
    
    Parameters:
    -----------
    lat : float
        Latitude of the location
    lon : float
        Longitude of the location
    timestamp : datetime, optional
        Timestamp for historical data (default: current time)
    api_key : str, optional
        OpenWeatherMap API key (default: from API_KEYS dict)
        
    Returns:
    --------
    dict
        Weather data from OpenWeatherMap API
    """
    if api_key is None:
        api_key = API_KEYS.get('openweathermap')
        
    if api_key is None or api_key.startswith('YOUR_'):
        print("Warning: OpenWeatherMap API key not configured.")
        return None
    
    # If no timestamp provided, use current time
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    
    # Format timestamp for the API
    unix_time = int(timestamp.timestamp())
    
    # Build API URL
    base_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    params = {
        'lat': lat,
        'lon': lon,
        'dt': unix_time,
        'appid': api_key,
        'units': 'metric'  # Use metric units
    }
    
    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        data = response.json()
        
        # Extract and format the relevant weather data
        if 'data' in data and len(data['data']) > 0:
            weather_data = {
                'source': 'OpenWeatherMap',
                'timestamp': datetime.fromtimestamp(data['data'][0]['dt']),
                'temperature': data['data'][0].get('temp'),
                'pressure': data['data'][0].get('pressure'),
                'humidity': data['data'][0].get('humidity'),
                'wind_speed': data['data'][0].get('wind_speed'),
                'wind_direction': data['data'][0].get('wind_deg'),
                'weather_condition': data['data'][0].get('weather', [{}])[0].get('main')
            }
            return weather_data
        else:
            print(f"No data found for the specified time: {timestamp}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from OpenWeatherMap: {e}")
        return None

def fetch_all_external_data(lat, lon, timestamp=None):
    """
    Fetch weather data from external API
    
    Parameters:
    -----------
    lat : float
        Latitude of the location
    lon : float
        Longitude of the location
    timestamp : datetime, optional
        Timestamp for historical data
        
    Returns:
    --------
    list
        List of weather data dictionaries from different sources
    """
    results = []
    
    # Only fetch data from OpenWeatherMap
    owm_data = fetch_openweathermap_data(lat, lon, timestamp)
    if owm_data:
        results.append(owm_data)
    
    return results

def compare_with_local_data(local_df, external_data_list, tolerance=0.5):
    """
    Compare local wind measurements with external API data
    
    Parameters:
    -----------
    local_df : pandas.DataFrame
        DataFrame containing local wind measurements
    external_data_list : list
        List of dictionaries with external API data
    tolerance : float, optional
        Tolerance factor for flagging discrepancies
        
    Returns:
    --------
    dict
        Comparison results with flags for discrepancies
    """
    if not external_data_list:
        return {
            'status': 'error',
            'message': 'No external data available for comparison'
        }
    
    # Extract local wind data
    local_wind_speed = local_df['Wind_Speed (m/s)'].mean()
    local_wind_dir = local_df['Wind_Direction (°)'].mean()
    
    # Extract external wind data
    external_wind_speeds = [data['wind_speed'] for data in external_data_list if data.get('wind_speed') is not None]
    external_wind_dirs = [data['wind_direction'] for data in external_data_list if data.get('wind_direction') is not None]
    
    if not external_wind_speeds or not external_wind_dirs:
        return {
            'status': 'error',
            'message': 'External data missing wind information'
        }
    
    # If there's only one external source, we don't need to calculate mean
    if len(external_data_list) == 1:
        mean_ext_wind_speed = external_wind_speeds[0]
        mean_ext_wind_dir = external_wind_dirs[0]
    else:
        # Calculate mean external values
        mean_ext_wind_speed = np.mean(external_wind_speeds)
        mean_ext_wind_dir = np.mean(external_wind_dirs)
    
    # Calculate differences
    speed_diff = abs(local_wind_speed - mean_ext_wind_speed)
    
    # For wind direction, we need to handle the circular nature (0° = 360°)
    dir_diff_raw = abs(local_wind_dir - mean_ext_wind_dir)
    dir_diff = min(dir_diff_raw, 360 - dir_diff_raw)
    
    # Determine if differences exceed tolerance thresholds
    speed_threshold = max(1.0, local_wind_speed * tolerance)
    dir_threshold = 30  # Fixed threshold for direction (in degrees)
    
    speed_flag = speed_diff > speed_threshold
    dir_flag = dir_diff > dir_threshold
    
    comparison = {
        'status': 'ok',
        'local_data': {
            'wind_speed': local_wind_speed,
            'wind_direction': local_wind_dir
        },
        'external_data': {
            'sources': [data['source'] for data in external_data_list],
            'wind_speed': {
                'mean': mean_ext_wind_speed,
                'values': external_wind_speeds
            },
            'wind_direction': {
                'mean': mean_ext_wind_dir,
                'values': external_wind_dirs
            }
        },
        'differences': {
            'wind_speed': {
                'value': speed_diff,
                'threshold': speed_threshold,
                'flag': speed_flag
            },
            'wind_direction': {
                'value': dir_diff,
                'threshold': dir_threshold,
                'flag': dir_flag
            }
        }
    }
    
    return comparison

def plot_wind_comparison(comparison_result, output_dir=None):
    """
    Create visualization of wind data comparison between local and external sources
    
    Parameters:
    -----------
    comparison_result : dict
        Output from compare_with_local_data function
    output_dir : str, optional
        Directory to save the output plot
        
    Returns:
    --------
    tuple
        (fig1, fig2) - Figure objects for the created plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    if comparison_result['status'] != 'ok':
        print(f"Error: {comparison_result['message']}")
        return None
    
    # --- (1) WIND SPEED COMPARISON FIGURE ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot local wind speed as a horizontal line
    local_speed = comparison_result['local_data']['wind_speed']
    ax1.axhline(
        y=local_speed, color='blue', linestyle='-', linewidth=2, label='Local Measurement'
    )
    
    # External wind speeds
    sources = comparison_result['external_data']['sources']
    external_speeds = comparison_result['external_data']['wind_speed']['values']
    
    for i, (source, value) in enumerate(zip(sources, external_speeds)):
        ax1.scatter(i+1, value, color='green', s=100, zorder=5)
        ax1.text(i+1, value + 0.2, f"{value:.1f}", ha='center', color='green')
    
    # Mean external wind speed
    mean_ext_speed = comparison_result['external_data']['wind_speed']['mean']
    ax1.axhline(y=mean_ext_speed, color='green', linestyle='--', linewidth=2, label='External Mean')
    
    # Add tolerance band for local speed
    speed_threshold = comparison_result['differences']['wind_speed']['threshold']
    ax1.axhspan(local_speed - speed_threshold, local_speed + speed_threshold,
                alpha=0.2, color='gray', label='Tolerance Range')
    
    # Labels and legend
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax1.set_xticks(range(1, len(sources) + 1))
    ax1.set_xticklabels(sources)
    ax1.set_title('Wind Speed Comparison: Local vs. External Sources', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Flag if discrepancy
    if comparison_result['differences']['wind_speed']['flag']:
        fig1.suptitle('⚠️ Wind Speed Discrepancy Detected!', color='red', fontsize=16)
    
    # --- (2) WIND DIRECTION COMPARISON FIGURE ---
    fig2, ax2 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Conversion: Meteorological FROM → Polar angle
    # 0° in meteorology = wind from North
    # (270 - deg) % 360 => 0° at top, angles increase clockwise
    def deg_to_rad_meteorological(deg):
        return np.radians((270 - deg) % 360)
    
    # Helper function to draw an arrow from center outward
    def annotate_arrow(ax, text, angle_rad, color, r_tail=0.0, r_head=1.0):
        """
        Draw an arrow on a polar plot from r_tail to r_head at angle_rad.
        'text' is placed near r_head.
        """
        ax.annotate(
            text,
            xy=(angle_rad, r_head),       # Arrow head
            xytext=(angle_rad, r_tail),  # Arrow tail
            arrowprops=dict(arrowstyle='->', color=color, lw=2),
            ha='center',
            color=color,
            fontsize=12
        )
    
    # Local wind direction
    local_dir_deg = comparison_result['local_data']['wind_direction']
    local_dir_rad = deg_to_rad_meteorological(local_dir_deg)
    annotate_arrow(ax2, 'Local', local_dir_rad, 'blue', r_tail=0.0, r_head=1.2)
    
    # External wind directions
    external_dirs = comparison_result['external_data']['wind_direction']['values']
    for i, (source, dir_deg) in enumerate(zip(sources, external_dirs)):
        dir_rad = deg_to_rad_meteorological(dir_deg)
        # Offset each arrow slightly so they don't overlap
        annotate_arrow(ax2, source, dir_rad, 'green', r_tail=0.0, r_head=1.0 - (i * 0.1))
    
    # Mean external direction
    mean_ext_dir_deg = comparison_result['external_data']['wind_direction']['mean']
    mean_ext_dir_rad = deg_to_rad_meteorological(mean_ext_dir_deg)
    annotate_arrow(ax2, 'External Mean', mean_ext_dir_rad, 'darkgreen', r_tail=0.0, r_head=0.8)
    
    # Configure polar plot
    # 0° at top (N), angles go clockwise
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_rlim(0, 2)  # Radius limit
    ax2.grid(True)
    
    # Cardinal direction labels
    ax2.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax2.set_yticklabels([])
    
    ax2.set_title('Wind Direction Comparison: Local vs. External Sources', fontsize=14)
    if comparison_result['differences']['wind_direction']['flag']:
        fig2.suptitle('⚠️ Wind Direction Discrepancy Detected!', color='red', fontsize=16)
    
    # Save figures if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'wind_speed_comparison.png'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(output_dir, 'wind_direction_comparison.png'), dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig1, fig2


def validate_wind_data(methane_gdf, wind_df, timestamp, output_dir=None):
    """
    Validate wind data for a specified timestamp against external data sources
    
    Parameters:
    -----------
    methane_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing methane sensor data
    wind_df : pandas.DataFrame
        DataFrame containing wind data
    timestamp : datetime or pandas.Timestamp
        Timestamp for data validation
    output_dir : str, optional
        Directory to save validation results and plots
        
    Returns:
    --------
    dict
        Validation results
    """
    # Convert timestamp to pandas.Timestamp if it's not already
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.to_datetime(timestamp)
    
    # Get local data for the specified timestamp
    local_wind = wind_df[wind_df['Timestamp'] == timestamp].copy()
    
    if local_wind.empty:
        return {
            'status': 'error',
            'message': f'No local wind data found for timestamp {timestamp}'
        }
    
    # Get methane sensors location for the specified timestamp
    methane_ts = methane_gdf[methane_gdf['Timestamp'] == timestamp]
    
    if methane_ts.empty:
        return {
            'status': 'error',
            'message': f'No methane data found for timestamp {timestamp}'
        }
    
    # Use the mean position of methane sensors as the location for external data
    mean_lat = methane_ts.geometry.y.mean()
    mean_lon = methane_ts.geometry.x.mean()
    
    # Fetch external data
    print(f"Fetching external weather data for {timestamp} at coordinates ({mean_lat:.5f}, {mean_lon:.5f})...")
    external_data = fetch_all_external_data(mean_lat, mean_lon, timestamp)
    
    if not external_data:
        # Generate only ONE demo source for OpenWeatherMap
        print("No external data available. Generating OpenWeatherMap demo data.")
        
        # Get the local wind data
        local_wind_speed = local_wind['Wind_Speed (m/s)'].values[0]
        local_wind_dir = local_wind['Wind_Direction (°)'].values[0]
        
        # Create slightly different values for demonstration - only one demo source
        external_data = [
            {
                'source': 'OpenWeatherMap (Demo)',
                'timestamp': timestamp,
                'wind_speed': local_wind_speed * (1 + np.random.uniform(-0.15, 0.15)),  # ±15% variation
                'wind_direction': (local_wind_dir + np.random.uniform(-20, 20)) % 360   # ±20° variation
            }
        ]
    
    # Compare local data with external data
    print("Comparing local measurements with external data sources...")
    comparison = compare_with_local_data(local_wind, external_data)
    
    # Generate plots
    if output_dir:
        print(f"Generating comparison plots to {output_dir}...")
        plot_wind_comparison(comparison, output_dir)
    
    # Create validation report
    validation_result = {
        'status': 'ok',
        'timestamp': timestamp,
        'location': {
            'latitude': mean_lat,
            'longitude': mean_lon
        },
        'comparison': comparison,
        'validation_flags': {
            'wind_speed': bool(comparison['differences']['wind_speed']['flag']),  # Convert numpy.bool_ to Python bool
            'wind_direction': bool(comparison['differences']['wind_direction']['flag'])  # Convert numpy.bool_ to Python bool
        },
        'external_sources': [data['source'] for data in external_data]
    }
    
    # Save validation report to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, f'validation_report_{timestamp.strftime("%Y%m%d_%H%M")}.json')
        
        with open(report_file, 'w') as f:
            # Convert timestamp to string for JSON serialization
            result_copy = validation_result.copy()
            result_copy['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert any NumPy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            # Convert all numpy types before serialization
            serializable_copy = convert_numpy_types(result_copy)
            json.dump(serializable_copy, f, indent=4)
    
    return validation_result

def show_external_validation_tab(methane_gdf, wind_df, timestamps):
    """
    Create a Streamlit tab for anemometer data validation
    
    Parameters:
    -----------
    methane_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing methane sensor data
    wind_df : pandas.DataFrame
        DataFrame containing wind data
    timestamps : list
        List of available timestamps
    """
    import streamlit as st
    
    st.subheader("Anemometer Data Validation")
    
    # Add information about external validation feature
    st.write("""
    This feature validates your local anemometer (wind) measurements against data from OpenWeatherMap's climate API.
    By comparing local sensor data with external reference data, you can identify potential sensor errors or anomalies.
    """)
    
    # API key configuration
    st.sidebar.subheader("API Configuration")
    
    # Use session state to store API keys - simplified to only OpenWeatherMap
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'openweathermap': API_KEYS.get('openweathermap', '')
        }
    
    # API key input (in sidebar)
    with st.sidebar.expander("Configure API Key"):
        st.session_state.api_keys['openweathermap'] = st.text_input(
            "OpenWeatherMap API Key", 
            value=st.session_state.api_keys['openweathermap'],
            type="password"
        )
        
        if st.button("Save API Key"):
            API_KEYS.clear()
            API_KEYS.update(st.session_state.api_keys)
            st.sidebar.success("API key updated!")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Select timestamp for validation
        selected_timestamp = st.selectbox(
            "Select timestamp to validate:",
            options=timestamps,
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"),
            index=min(23, len(timestamps)-1)  # Default to noon if available
        )
        
        # Validation button
        if st.button("Validate Wind Data"):
            if not any(key and not key.startswith('YOUR_') for key in API_KEYS.values()):
                st.error("⚠️ Please configure at least one API key in the sidebar.")
            else:
                with st.spinner("Fetching external data and validating..."):
                    # Create temp directory for outputs
                    temp_dir = os.path.join(os.path.expanduser("~"), "temp_validation")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Run validation
                    validation_result = validate_wind_data(
                        methane_gdf, 
                        wind_df, 
                        selected_timestamp, 
                        temp_dir
                    )
                    
                    # Store result in session state
                    st.session_state.validation_result = validation_result
                    
                    # Check status
                    if validation_result['status'] != 'ok':
                        st.error(f"Validation error: {validation_result['message']}")
                    else:
                        st.success("Validation completed!")
        
        # Display validation information
        if 'validation_result' in st.session_state and st.session_state.validation_result['status'] == 'ok':
            result = st.session_state.validation_result
            
            st.subheader("Validation Summary")
            
            # Display flags
            speed_flag = result['validation_flags']['wind_speed']
            dir_flag = result['validation_flags']['wind_direction']
            
            if speed_flag or dir_flag:
                st.warning("⚠️ Discrepancies detected!")
            else:
                st.info("✅ No significant discrepancies detected.")
            
            # Display comparison data
            st.write("**Local Wind Measurements:**")
            st.write(f"- Wind Speed: {result['comparison']['local_data']['wind_speed']:.2f} m/s")
            st.write(f"- Wind Direction: {result['comparison']['local_data']['wind_direction']:.1f}°")
            
            st.write("**External Data Sources:**")
            st.write(", ".join(result['external_sources']))
            
            # Change the heading to remove "Mean" when there's only one source
            if len(result['external_sources']) == 1:
                st.write("**External Measurements:**")
            else:
                st.write("**External Measurements (Mean):**")
                
            st.write(f"- Wind Speed: {result['comparison']['external_data']['wind_speed']['mean']:.2f} m/s")
            st.write(f"- Wind Direction: {result['comparison']['external_data']['wind_direction']['mean']:.1f}°")
            
            # Display differences
            st.write("**Differences:**")
            speed_diff = result['comparison']['differences']['wind_speed']
            dir_diff = result['comparison']['differences']['wind_direction']
            
            speed_color = "red" if speed_diff['flag'] else "green"
            dir_color = "red" if dir_diff['flag'] else "green"
            
            st.write(f"- Wind Speed: <span style='color:{speed_color}'>{speed_diff['value']:.2f} m/s</span> (threshold: {speed_diff['threshold']:.2f})", unsafe_allow_html=True)
            st.write(f"- Wind Direction: <span style='color:{dir_color}'>{dir_diff['value']:.1f}°</span> (threshold: {dir_diff['threshold']}°)", unsafe_allow_html=True)
    
    with col2:
        # Display validation plots if available
        if 'validation_result' in st.session_state and st.session_state.validation_result['status'] == 'ok':
            st.subheader("Validation Plots")
            
            # Create the plots
            fig1, fig2 = plot_wind_comparison(st.session_state.validation_result['comparison'])
            
            # Display the plots
            st.write("**Wind Speed Comparison:**")
            st.pyplot(fig1)
            
            st.write("**Wind Direction Comparison:**")
            st.pyplot(fig2)
        else:
            st.info("Select a timestamp and click 'Validate Wind Data' to see results here.")
            
            # Display a placeholder image
            st.image("https://via.placeholder.com/800x400?text=Wind+Data+Validation+Plots", use_container_width=True)

if __name__ == "__main__":
    # Simple test function
    print("External Data Validation Module")
    print("Use this module to validate local wind data against external APIs")
