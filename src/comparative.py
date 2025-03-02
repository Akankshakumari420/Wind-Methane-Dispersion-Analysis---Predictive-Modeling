import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import datetime

def show_method_comparison_tab(methane_gdf, timestamps):
    """Display comparison between different analysis methods"""
    st.subheader("Comparison Between Analysis Methods")
    
    # Select timestamp for comparison
    selected_timestamp = st.select_slider(
        "Select timestamp for comparison:",
        options=timestamps,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
    )
    
    # Filter data for selected timestamp
    ts_data = methane_gdf[methane_gdf['Timestamp'] == selected_timestamp]
    
    # Choose analysis methods to compare
    method_options = ["IDW Interpolation", "Kriging Interpolation", "DBSCAN Clustering", "KMeans Clustering"]
    selected_methods = st.multiselect("Select methods to compare:", method_options, default=["IDW Interpolation", "Kriging Interpolation"])
    
    if not selected_methods:
        st.warning("Please select at least one analysis method.")
        return
    
    # Create figure for method comparison
    fig, axes = plt.subplots(1, len(selected_methods), figsize=(5*len(selected_methods), 5))
    if len(selected_methods) == 1: axes = [axes]  # Handle single axis case
    
    # Process each selected method
    for i, method in enumerate(selected_methods):
        try:
            if method == "IDW Interpolation":
                from src.interpolation import idw_interpolation
                result = idw_interpolation(ts_data, power=2)
                im = axes[i].imshow(result['grid_z'], extent=result['extent'], origin='lower', cmap='jet')
                axes[i].scatter(ts_data.geometry.x, ts_data.geometry.y, c=ts_data['Methane_Concentration (ppm)'], 
                              cmap='jet', edgecolor='k', s=50)
                plt.colorbar(im, ax=axes[i], label='Methane Concentration (ppm)')
                
            elif method == "Kriging Interpolation":
                from src.interpolation import kriging_interpolation
                result = kriging_interpolation(ts_data)
                im = axes[i].imshow(result['grid_z'], extent=result['extent'], origin='lower', cmap='jet')
                axes[i].scatter(ts_data.geometry.x, ts_data.geometry.y, c=ts_data['Methane_Concentration (ppm)'], 
                              cmap='jet', edgecolor='k', s=50)
                plt.colorbar(im, ax=axes[i], label='Methane Concentration (ppm)')
                
            elif method == "DBSCAN Clustering" or method == "KMeans Clustering":
                func = (lambda data: perform_dbscan_clustering(data)) if method == "DBSCAN Clustering" else (lambda data: perform_kmeans_clustering(data))
                from src.clustering import perform_dbscan_clustering, perform_kmeans_clustering
                clusters = func(ts_data)
                scatter = axes[i].scatter(ts_data.geometry.x, ts_data.geometry.y, c=clusters, cmap='tab10', 
                                        edgecolor='k', s=50)
                axes[i].add_artist(axes[i].legend(*scatter.legend_elements(), title="Clusters"))
            
            # Set common elements
            axes[i].set_title(method)
            axes[i].set_xlabel('X Coordinate')
            axes[i].set_ylabel('Y Coordinate') if i == 0 else axes[i].set_ylabel('')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"{method} - Error")
    
    plt.tight_layout()
    st.pyplot(fig)

def show_temporal_comparison_tab(methane_gdf, timestamps):
    """Display comparison between different time periods"""
    st.subheader("Temporal Comparison")
    
    # Select timestamps for comparison
    col1, col2 = st.columns(2)
    with col1:
        timestamp1 = st.selectbox("Select first timestamp:", options=timestamps,
                                format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"), index=0)
    with col2:
        timestamp2 = st.selectbox("Select second timestamp:", options=timestamps,
                                format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"), 
                                index=min(len(timestamps)-1, 12))
    
    # Filter data for selected timestamps
    data1 = methane_gdf[methane_gdf['Timestamp'] == timestamp1]
    data2 = methane_gdf[methane_gdf['Timestamp'] == timestamp2]
    
    # Create side-by-side scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (data, timestamp, ax) in enumerate(zip([data1, data2], [timestamp1, timestamp2], axes)):
        scatter = ax.scatter(data.geometry.x, data.geometry.y, c=data['Methane_Concentration (ppm)'], 
                           cmap='jet', edgecolor='k', s=50)
        ax.set_title(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        plt.colorbar(scatter, ax=ax, label='Methane Concentration (ppm)')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Difference analysis
    st.write("### Difference Analysis")
    
    # Merge data and calculate differences
    merged_data = pd.merge(
        data1[['Sensor_ID', 'Methane_Concentration (ppm)']].rename(columns={'Methane_Concentration (ppm)': 'Methane_Time1'}),
        data2[['Sensor_ID', 'Methane_Concentration (ppm)']].rename(columns={'Methane_Concentration (ppm)': 'Methane_Time2'}),
        on='Sensor_ID'
    )
    merged_data['Difference'] = merged_data['Methane_Time2'] - merged_data['Methane_Time1']
    
    # Create bar chart
    fig = px.bar(
        merged_data, x='Sensor_ID', y='Difference',
        title=f'Methane Concentration Difference ({timestamp2.strftime("%Y-%m-%d %H:%M")} - {timestamp1.strftime("%Y-%m-%d %H:%M")})',
        color='Difference', color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig)
    
    # Display statistical summary
    st.write("### Statistical Summary of Differences")
    stats_df = pd.DataFrame([{
        'Mean Difference': merged_data['Difference'].mean(),
        'Median Difference': merged_data['Difference'].median(),
        'Max Increase': merged_data['Difference'].max(),
        'Max Decrease': merged_data['Difference'].min(),
        'Standard Deviation': merged_data['Difference'].std()
    }]).T.reset_index()
    stats_df.columns = ['Statistic', 'Value']
    st.table(stats_df)

def show_sensor_comparison_tab(methane_gdf, timestamps):
    """Display comparison between different sensors"""
    st.subheader("Sensor Comparison")
    
    # Select sensors to compare
    sensors = sorted(methane_gdf['Sensor_ID'].unique())
    selected_sensors = st.multiselect("Select sensors to compare:", 
                                    options=sensors, 
                                    default=sensors[:3] if len(sensors) >= 3 else sensors)
    
    if not selected_sensors:
        st.warning("Please select at least one sensor.")
        return
    
    # Filter data for selected sensors
    sensor_data = methane_gdf[methane_gdf['Sensor_ID'].isin(selected_sensors)].copy()
    
    # Time series visualization
    st.write("### Time Series Comparison")
    fig = px.line(
        sensor_data, x='Timestamp', y='Methane_Concentration (ppm)', color='Sensor_ID',
        title='Methane Concentration Time Series by Sensor'
    )
    st.plotly_chart(fig)
    
    # Statistical comparison
    st.write("### Statistical Comparison")
    stats = sensor_data.groupby('Sensor_ID')['Methane_Concentration (ppm)'].agg(
        ['mean', 'std', 'min', 'median', 'max']
    ).reset_index()
    st.dataframe(stats)
    
    # Boxplot
    fig = px.box(sensor_data, x='Sensor_ID', y='Methane_Concentration (ppm)',
               title='Methane Concentration Distribution by Sensor', points='all')
    st.plotly_chart(fig)
    
    # Correlation analysis
    st.write("### Correlation Between Sensors")
    try:
        pivot_df = sensor_data.pivot_table(index='Timestamp', columns='Sensor_ID',
                                         values='Methane_Concentration (ppm)')
        corr_matrix = pivot_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, vmin=-1, vmax=1)
        plt.title("Sensor Correlation Matrix")
        st.pyplot(fig)
        
        st.info("""
        **Interpreting the Correlation Matrix:**
        - A value close to 1 indicates that two sensors have similar methane readings over time.
        - A value close to -1 indicates that when one sensor's readings increase, the other's decrease.
        - A value close to 0 indicates little or no relationship between the sensors' readings.
        """)
    except Exception as e:
        st.error(f"Error creating correlation matrix: {str(e)}")