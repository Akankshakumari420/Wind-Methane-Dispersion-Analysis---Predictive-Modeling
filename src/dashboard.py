import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import TimestampedGeoJson, HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import branca.colormap as cm
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import from our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_data, preprocess_methane_data, preprocess_wind_data, merge_data
from src.interpolation import idw_interpolation, kriging_interpolation
from src.clustering import perform_dbscan_clustering, perform_kmeans_clustering
from src.visualization import plot_clusters, create_map, create_methane_heatmap
from src.analysis import calculate_model_performance, convert_wind_to_uv, create_time_features, show_feature_analysis_tab
from src.modeling import show_ml_models_tab, show_time_series_tab
from src.visualization import create_map_with_wind_vectors, create_simple_methane_map
from src.external_validation import show_external_validation_tab  
from src.wind_analysis import add_wind_rose_to_dashboard

def load_all_data():
    """
    Load and process all data.
    
    Returns:
    --------
    tuple
        (methane_gdf, wind_df, merged_gdf) : Processed data
    """
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    methane_path = os.path.join(project_dir, 'data', 'methane_sensors.csv')
    wind_path = os.path.join(project_dir, 'data', 'wind_data.csv')
    
    # For testing with direct paths (if needed)
    if not os.path.exists(methane_path):
        methane_path = r"C:\Users\Dell\Downloads\Wind-Methane-Dispersion-Analysis---Predictive-Modeling-master\Wind-Methane-Dispersion-Analysis---Predictive-Modeling-master\data\processed\methane_sensors.csv"
        wind_path = r"C:\Users\Dell\Downloads\Wind-Methane-Dispersion-Analysis---Predictive-Modeling-master\Wind-Methane-Dispersion-Analysis---Predictive-Modeling-master\data\processed\wind_processed.csv"
    
    # Load and preprocess data
    methane_df, wind_df = load_data(methane_path, wind_path)
    methane_gdf = preprocess_methane_data(methane_df)
    wind_df_processed = preprocess_wind_data(wind_df)
    merged_gdf = merge_data(methane_gdf, wind_df_processed)
    
    return methane_gdf, wind_df_processed, merged_gdf

def merge_methane_wind_data(methane_gdf, wind_df=None):
    """
    Merge methane and wind data on timestamp
    
    Parameters:
    -----------
    methane_gdf : GeoDataFrame
        GeoDataFrame containing methane sensor data
    wind_df : DataFrame, optional
        DataFrame containing wind data, will try to load if None
    
    Returns:
    --------
    GeoDataFrame
        Merged GeoDataFrame with wind data
    """
    import pandas as pd
    import os
    
    if wind_df is None:
        # Try to load wind data
        try:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            wind_path = os.path.join(project_dir, "data", "processed", "wind_processed.csv")
            
            if os.path.exists(wind_path):
                wind_df = pd.read_csv(wind_path)
                wind_df['Timestamp'] = pd.to_datetime(wind_df['Timestamp'])
            else:
                return methane_gdf  # Return original if wind data not found
        except Exception:
            return methane_gdf  # Return original on any error
    
    # Merge data
    result = methane_gdf.copy()
    
    # Create mapping of timestamps to wind data
    wind_dict = {}
    for _, row in wind_df.iterrows():
        wind_dict[row['Timestamp']] = {
            'Wind_Speed (m/s)': row['Wind_Speed (m/s)'],
            'Wind_Direction (°)': row['Wind_Direction (°)']
        }
    
    # Add wind data columns
    result['Wind_Speed (m/s)'] = result['Timestamp'].map(lambda x: wind_dict.get(x, {}).get('Wind_Speed (m/s)', None))
    result['Wind_Direction (°)'] = result['Timestamp'].map(lambda x: wind_dict.get(x, {}).get('Wind_Direction (°)', None))
    
    return result

def ensure_wind_data(df):
    """Add wind data columns if they don't exist"""
    if 'Wind_Speed (m/s)' not in df.columns:
        st.warning("Wind speed data not found - using default values for visualization")
        df['Wind_Speed (m/s)'] = 3.5  # Default wind speed
    
    if 'Wind_Direction (°)' not in df.columns:
        st.warning("Wind direction data not found - using default values for visualization")
        df['Wind_Direction (°)'] = 270.0  # Default direction (West)
    
    return df

def create_dashboard():
    """
    Create the Streamlit dashboard.
    """
    # Set page configuration
    st.set_page_config(page_title="Methane Monitoring Dashboard", layout="wide")
    
    # Load data
    methane_gdf, wind_df, merged_gdf = load_all_data()
    
    # Add the wind data to methane_gdf
    methane_gdf = merge_methane_wind_data(methane_gdf, wind_df)
    
    # Ensure wind data columns exist
    methane_gdf = ensure_wind_data(methane_gdf)
    
    # Get unique timestamps
    timestamps = sorted(methane_gdf['Timestamp'].unique())
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Overview", 
        "Geospatial Visualization", 
        "Interpolation", 
        "Clustering", 
        "Predictive Analytics",
        "External Validation",    # Add the new page option
        "Comparative Analysis"
    ])
    
    # Overview page
    if page == "Overview":
        st.title("Methane Monitoring Dashboard")
        st.markdown("""
        This dashboard provides an overview of methane concentration data collected from various sensors.
        You can explore the data through different visualizations and analyses.
        """)
        
        st.header("Data Overview")
        
        # Display data summary
        st.subheader("Methane Data")
        st.write(methane_gdf.head())
        
        st.subheader("Wind Data")
        st.write(wind_df.head())
        
        st.subheader("Merged Data")
        st.write(merged_gdf.head())
        
        # Display summary statistics
        st.header("Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Methane Concentration")
            st.write(methane_gdf['Methane_Concentration (ppm)'].describe())
        
        with col2:
            st.subheader("Wind Speed")
            st.write(wind_df['Wind_Speed (m/s)'].describe())
        
        # Display time series plots
        st.header("Time Series Plots")
        
        st.subheader("Methane Concentration Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        for sensor_id, group in methane_gdf.groupby('Sensor_ID'):
            ax.plot(group['Timestamp'], group['Methane_Concentration (ppm)'], label=f'Sensor {sensor_id}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Methane Concentration (ppm)')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Wind Speed and Direction Over Time")
        from src.visualization import plot_wind_data
        fig = plot_wind_data(wind_df)
        st.pyplot(fig)
        
        # Wind rose is already here
        st.header("Wind Analysis")
        add_wind_rose_to_dashboard(wind_df)
    
    # Geospatial Visualization page
    elif page == "Geospatial Visualization":
        st.header("Geospatial Visualization")
        
        # Timestamp selector
        selected_timestamp = st.select_slider(
            "Select timestamp:",
            options=timestamps,
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
        )
        
        # Map type selector
        map_type = st.radio(
            "Select map type:",
            ["Markers Map", "Heatmap"]
        )
        
        # Create map based on selection
        if map_type == "Markers Map":
            st.subheader("Methane Sensor Locations with Wind Data")
            st.write("The map shows methane sensor locations (red circles) and wind direction/speed (blue arrow).")
            st.write("Circle size indicates methane concentration. Hover over points for details.")
            
            with st.spinner("Creating map..."):
                try:
                    m = create_map_with_wind_vectors(methane_gdf, selected_timestamp)
                    folium_static(m)
                except Exception as e:
                    st.error(f"Error creating map: {str(e)}")
                    st.write("Please check that your data has the required columns.")
        else:
            st.subheader("Methane Concentration Heatmap")
            fig = create_methane_heatmap(methane_gdf, selected_timestamp)
            st.pyplot(fig)
        
        # Display data for selected timestamp
        st.subheader(f"Data for {selected_timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Methane Readings:")
            ts_methane = methane_gdf[methane_gdf['Timestamp'] == selected_timestamp].drop(columns='geometry')
            st.dataframe(ts_methane)
        
        with col2:
            st.write("Wind Conditions:")
            ts_wind = wind_df[wind_df['Timestamp'] == selected_timestamp]
            st.dataframe(ts_wind)
    
    # Interpolation page
    elif page == "Interpolation":
        st.header("Methane Concentration Interpolation")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Controls
            selected_timestamp = st.selectbox(
                "Select timestamp:",
                options=timestamps,
                format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"),
                index=23  # Noon timestamp by default
            )
            
            interp_method = st.radio(
                "Interpolation method:",
                ["IDW", "Kriging"]
            )
            
            st.info("""
            **Interpolation Methods:**
            - **IDW**: Inverse Distance Weighted interpolation
            - **Kriging**: Gaussian process regression
            """)
        
        with col2:
            with st.spinner(f"Performing {interp_method} interpolation..."):
                from src.visualization import plot_interpolation
                try:
                    fig = plot_interpolation(methane_gdf, selected_timestamp, interp_method)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating interpolation: {e}")
    
    # Clustering page
    elif page == "Clustering":
        st.header("Methane Concentration Clustering")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Controls
            selected_timestamp = st.selectbox(
                "Select timestamp:",
                options=timestamps,
                format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"),
                index=23  # Noon timestamp by default
            )
            
            cluster_method = st.radio(
                "Clustering method:",
                ["DBSCAN", "KMeans"]
            )
            
            st.info("""
            **Clustering Methods:**
            - **DBSCAN**: Density-based spatial clustering
            - **KMeans**: K-means clustering algorithm
            """)
        
        with col2:
            with st.spinner(f"Performing {cluster_method} clustering..."):
                from src.visualization import plot_clustering
                try:
                    fig = plot_clustering(methane_gdf, selected_timestamp, cluster_method)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating clustering visualization: {e}")
    
    # Predictive Analytics page
    elif page == "Predictive Analytics":
        st.header("Predictive Analytics")
        
        st.info("""
        This page shows advanced predictive modeling analysis of methane concentration data.
        Multiple models are trained and compared to provide the best predictive capabilities.
        """)
        
        # Create tabs for different predictive models
        tabs = st.tabs(["Machine Learning Models", "Time Series Forecasting", "Feature Analysis"])
        
        with tabs[0]:  # Machine Learning Models tab
            from src.modeling import show_ml_models_tab
            show_ml_models_tab(merged_gdf)
            
        with tabs[1]:  # Time Series Forecasting tab
            from src.forecasting import show_forecasting_tab
            show_forecasting_tab(methane_gdf)
            
        with tabs[2]:  # Feature Analysis tab
            from src.analysis import show_feature_analysis_tab
            show_feature_analysis_tab(merged_gdf)
    
    # Update the External Validation page
    elif page == "External Validation":
        st.header("Anemometer Data Validation")
        
        st.markdown("""
        This feature validates your anemometer (wind) measurements against the OpenWeatherMap climate API data.
        
        By comparing your local wind sensor data with external reference data, you can:
        - Validate the accuracy of your anemometer readings
        - Identify potential errors or calibration issues
        - Ensure reliable wind data for methane dispersion analysis
        
        > **Note:** This feature requires an OpenWeatherMap API key. Demo mode uses simulated data if no API key is provided.
        """)
        
        # Call the external validation tab with our loaded data
        show_external_validation_tab(methane_gdf, wind_df, timestamps)
    
    # Comparative Analysis page
    elif page == "Comparative Analysis":
        st.header("Comparative Analysis")
        
        st.info("""
        This section enables detailed comparison between different analysis methods, 
        time periods, and sensors to identify patterns and anomalies in methane data.
        """)
        
        tabs = st.tabs(["Method Comparison", "Temporal Comparison", "Sensor Comparison"])
        
        with tabs[0]:  # Method Comparison tab
            from src.comparative import show_method_comparison_tab
            show_method_comparison_tab(methane_gdf, timestamps)
            
        with tabs[1]:  # Temporal Comparison tab
            from src.comparative import show_temporal_comparison_tab
            show_temporal_comparison_tab(methane_gdf, timestamps)
            
        with tabs[2]:  # Sensor Comparison tab
            from src.comparative import show_sensor_comparison_tab
            show_sensor_comparison_tab(methane_gdf, timestamps)

# Run the dashboard
if __name__ == "__main__":
    create_dashboard()
