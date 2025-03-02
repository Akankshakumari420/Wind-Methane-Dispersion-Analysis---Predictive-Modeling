import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
import tempfile
import warnings
from datetime import datetime, timedelta

# Import local utilities
from .new_utils import ensure_directory, save_figure, create_colormap, convert_wind_to_uv

warnings.filterwarnings('ignore')

def plot_methane_time_series(methane_gdf, figsize=(12, 6), highlight_sensors=None):
    """Plot methane concentration time series for all sensors"""
    fig, ax = plt.subplots(figsize=figsize)
    sensors = methane_gdf['Sensor_ID'].unique()
    
    # Plot each sensor
    for sensor in sensors:
        data = methane_gdf[methane_gdf['Sensor_ID'] == sensor]
        style = {'marker': 'o', 'linewidth': 2, 'label': f"Sensor {sensor}"} if highlight_sensors and sensor in highlight_sensors else \
                {'alpha': 0.5, 'linewidth': 1, 'label': f"Sensor {sensor}" if len(sensors) < 10 else None}
        ax.plot(data['Timestamp'], data['Methane_Concentration (ppm)'], **style)
    
    ax.set_xlabel('Time'), ax.set_ylabel('Methane Concentration (ppm)')
    ax.set_title('Methane Concentration Time Series')
    if len(sensors) < 10 or highlight_sensors: ax.legend(loc='best', ncol=2)
    plt.tight_layout()
    return fig

def plot_wind_data(wind_df, figsize=(12, 6)):
    """Plot wind speed and direction over time"""
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot wind speed on primary y-axis
    ax1.set_xlabel('Time'), ax1.set_ylabel('Wind Speed (m/s)', color='tab:blue')
    ax1.plot(wind_df['Timestamp'], wind_df['Wind_Speed (m/s)'], color='tab:blue', marker='.', linestyle='-')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot direction on secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Wind Direction (°)', color='tab:red')
    ax2.plot(wind_df['Timestamp'], wind_df['Wind_Direction (°)'], color='tab:red', marker='.', linestyle='-')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Wind Speed and Direction Over Time')
    fig.tight_layout()
    return fig

def create_methane_heatmap(methane_gdf, timestamp=None, resolution=100, method='linear'):
    """Create a static heatmap of methane concentrations"""
    from scipy.interpolate import griddata
    
    # Filter data for timestamp if specified
    data = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)] if timestamp else methane_gdf.copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract coordinates and values for interpolation
    points = np.array([[p.x, p.y] for p in data.geometry])
    values = data['Methane_Concentration (ppm)'].values
    
    # Create grid for interpolation
    bounds = data.total_bounds
    margin = 0.05  # 5% margin
    x_margin = (bounds[2] - bounds[0]) * margin
    y_margin = (bounds[3] - bounds[1]) * margin
    x = np.linspace(bounds[0] - x_margin, bounds[2] + x_margin, resolution)
    y = np.linspace(bounds[1] - y_margin, bounds[3] + y_margin, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Interpolate values on grid
    zz = griddata(points, values, (xx, yy), method=method)
    
    # Create contour plot
    contour = ax.contourf(xx, yy, zz, cmap='YlOrRd', levels=15, alpha=0.8)
    plt.colorbar(contour, label='Methane Concentration (ppm)')
    
    # Add sensor points and labels
    ax.scatter(points[:,0], points[:,1], c=values, cmap='YlOrRd', edgecolor='k', s=80, zorder=10)
    for i, row in data.iterrows():
        ax.text(row.geometry.x, row.geometry.y, row['Sensor_ID'],
               fontsize=9, ha='center', va='center', fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Set labels and title
    ax.set_xlabel('Longitude'), ax.set_ylabel('Latitude')
    title = f"Methane Concentration Heatmap - {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')}" if timestamp \
            else "Methane Concentration Heatmap"
    ax.set_title(title)
    
    return fig

def create_interactive_map(methane_gdf, wind_df=None, timestamp=None):
    """Create an interactive folium map with methane data and wind vectors"""
    # Filter data for timestamp
    if timestamp:
        methane_data = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)].copy()
        if wind_df is not None:
            wind_data = wind_df[wind_df['Timestamp'] == pd.to_datetime(timestamp)].copy()
    else:
        latest = methane_gdf['Timestamp'].max()
        methane_data = methane_gdf[methane_gdf['Timestamp'] == latest].copy()
        if wind_df is not None:
            wind_data = wind_df[wind_df['Timestamp'] == latest].copy()
    
    # Create map centered on data
    center = [methane_data.geometry.y.mean(), methane_data.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=15)
    folium.LayerControl().add_to(m)
    
    # Create colormap
    vmin, vmax = methane_gdf['Methane_Concentration (ppm)'].min(), methane_gdf['Methane_Concentration (ppm)'].max()
    colormap = folium.LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax,
                                  caption='Methane Concentration (ppm)')
    m.add_child(colormap)
    
    # Add methane markers
    markers = folium.FeatureGroup(name="Methane Sensors")
    for _, row in methane_data.iterrows():
        popup_html = f"""
        <div style="width: 200px">
            <h4>Sensor: {row['Sensor_ID']}</h4>
            <b>Methane:</b> {row['Methane_Concentration (ppm)']:.2f} ppm<br>
            <b>Time:</b> {pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')}<br>
            <b>Location:</b> ({row.geometry.y:.5f}, {row.geometry.x:.5f})
        </div>
        """
        folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=10, color='black', weight=1,
                          fill=True, fill_color=colormap(row['Methane_Concentration (ppm)']), fill_opacity=0.7,
                          popup=folium.Popup(popup_html, max_width=300),
                          tooltip=f"Sensor {row['Sensor_ID']}: {row['Methane_Concentration (ppm)']:.2f} ppm").add_to(markers)
    markers.add_to(m)
    
    # Add wind vectors if available
    if wind_df is not None and not wind_data.empty:
        wind_layer = folium.FeatureGroup(name="Wind Vectors")
        wind_speed, wind_direction = wind_data['Wind_Speed (m/s)'].values[0], wind_data['Wind_Direction (°)'].values[0]
        
        # Calculate U,V components
        if 'U' in wind_data.columns and 'V' in wind_data.columns:
            u, v = wind_data['U'].values[0], wind_data['V'].values[0]
        else:
            from math import radians, sin, cos
            rad_dir = radians((270 - wind_direction) % 360)
            u, v = wind_speed * cos(rad_dir), wind_speed * sin(rad_dir)
        
        # Add wind vectors and legend
        scale = 0.001
        for _, row in methane_data.iterrows():
            start = [row.geometry.y, row.geometry.x]
            end = [start[0] + v * scale, start[1] + u * scale]
            folium.PolyLine(locations=[start, end], color='blue', weight=2, opacity=0.6).add_to(wind_layer)
            # Fix: Use folium.RegularPolygonMarker with proper parameters
            folium.Marker(
                location=end,
                icon=folium.Icon(color='blue', icon='arrow-up', angle=np.degrees(np.arctan2(v, u)))
            ).add_to(wind_layer)
        
        # Add wind legend - use a string key to avoid the camelize error
        folium.Marker(
            location=[center[0] - 0.002, center[1] - 0.002],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12pt; color: blue;">Wind: {wind_speed:.1f} m/s, {wind_direction:.0f}°</div>'
            )
        ).add_to(wind_layer)
        wind_layer.add_to(m)
    
    # Add heatmap layer
    heat_data = [[row.geometry.y, row.geometry.x, row['Methane_Concentration (ppm)']] for _, row in methane_data.iterrows()]
    HeatMap(heat_data, radius=25, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
          min_opacity=0.3, blur=15, name="Heat Map", show=False).add_to(m)
    
    return m

def plot_clustering(methane_gdf, timestamp, method='DBSCAN'):
    """Create a static visualization of clustering results"""
    from src.clustering import perform_dbscan_clustering, perform_kmeans_clustering
    
    # Filter data for timestamp
    data = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)]
    
    # Perform clustering
    if method.upper() == 'DBSCAN':
        clustered = perform_dbscan_clustering(data)
    elif method.upper() == 'KMEANS':
        clustered = perform_kmeans_clustering(data)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot clusters with different colors
    for cluster_id in sorted(clustered['Cluster'].unique()):
        points = clustered[clustered['Cluster'] == cluster_id]
        if cluster_id == -1:  # Noise points in DBSCAN
            ax.scatter(points.geometry.x, points.geometry.y, c='black', marker='x', s=80, label='Noise')
        else:
            color = plt.cm.tab10(cluster_id % 10)
            ax.scatter(points.geometry.x, points.geometry.y, c=[color], marker='o', s=80, 
                     edgecolor='k', label=f'Cluster {cluster_id}')
    
    # Add methane values as text
    for _, row in clustered.iterrows():
        ax.text(row.geometry.x, row.geometry.y + 0.0001, f"{row['Methane_Concentration (ppm)']:.1f}", 
               ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, pad=0.1))
    
    # Add labels and title
    ax.set_xlabel('Longitude'), ax.set_ylabel('Latitude')
    ax.set_title(f"{method} Clustering - {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')}")
    ax.legend(loc='best')
    plt.tight_layout()
    
    return fig

def plot_interpolation(methane_gdf, timestamp, method='IDW'):
    """Create a visualization of interpolation results"""
    # Import appropriate functions based on method
    if method.upper() == 'IDW':
        from src.interpolation import idw_interpolation as interp_func
    elif method.upper() == 'KRIGING':
        from src.interpolation import kriging_interpolation as interp_func
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Perform interpolation
    try:
        xx, yy, zz = interp_func(methane_gdf, timestamp)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create contour plot
        contour = ax.contourf(xx, yy, zz, cmap='YlOrRd', levels=15)
        plt.colorbar(contour, label='Methane Concentration (ppm)')
        
        # Plot sensor points
        data = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)]
        ax.scatter(data.geometry.x, data.geometry.y, c=data['Methane_Concentration (ppm)'],
                 cmap='YlOrRd', edgecolor='k', s=80)
        
        # Add sensor labels
        for _, row in data.iterrows():
            ax.text(row.geometry.x, row.geometry.y, row['Sensor_ID'], fontsize=9, ha='center', va='center',
                  fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Set labels and title
        ax.set_xlabel('Longitude'), ax.set_ylabel('Latitude')
        ax.set_title(f"{method} Interpolation - {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')}")
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        # Handle interpolation errors
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error performing {method} interpolation:\n{str(e)}", 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title(f"Interpolation Error")
        return fig

def plot_clusters(clustered_gdf, method_name="DBSCAN", figsize=(10, 8)):
    """
    Plot clustering results on a map.
    
    Parameters:
    -----------
    clustered_gdf : geopandas.GeoDataFrame
        GeoDataFrame with clustering results, must include 'Cluster' column
    method_name : str
        Name of clustering method used (for title)
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique clusters
    clusters = sorted(clustered_gdf['Cluster'].unique())
    
    # Create a colormap with enough colors
    cmap = plt.cm.get_cmap('tab10', max(10, len(clusters)))
    
    # Plot each cluster with a different color
    for i, cluster_id in enumerate(clusters):
        # Get data for this cluster
        cluster_data = clustered_gdf[clustered_gdf['Cluster'] == cluster_id]
        
        # Determine color and marker
        if cluster_id == -1:  # Noise points in DBSCAN
            color = 'black'
            marker = 'x'
            label = 'Noise'
        else:
            color = cmap(i % 10)
            marker = 'o'
            label = f'Cluster {cluster_id}'
        
        # Plot points
        ax.scatter(
            cluster_data.geometry.x, 
            cluster_data.geometry.y, 
            c=[color], 
            marker=marker, 
            s=80, 
            edgecolor='k', 
            label=label
        )
        
        # Add methane concentration values as text
        for _, row in cluster_data.iterrows():
            ax.text(
                row.geometry.x, 
                row.geometry.y + 0.0001, 
                f"{row['Methane_Concentration (ppm)']:.1f}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.5, pad=0.1)
            )
    
    # Add timestamp information if available
    if 'Timestamp' in clustered_gdf.columns:
        timestamp = pd.to_datetime(clustered_gdf['Timestamp'].iloc[0])
        title = f"{method_name} Clustering - {timestamp.strftime('%Y-%m-%d %H:%M')}"
    else:
        title = f"{method_name} Clustering Results"
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_map(methane_gdf, wind_df=None, timestamp=None):
    """
    Create an interactive folium map (alias for create_interactive_map)
    """
    return create_interactive_map(methane_gdf, wind_df, timestamp)

def create_map_with_wind_vectors(methane_gdf, timestamp):
    """
    Create an interactive map showing methane sensor locations with wind vectors
    
    Parameters:
    -----------
    methane_gdf : GeoDataFrame
        GeoDataFrame containing methane sensor data
    timestamp : datetime or str
        Timestamp to display data for
    
    Returns:
    --------
    folium.Map
        Interactive map with methane sensors and wind vectors
    """
    import folium
    import numpy as np
    import pandas as pd
    from folium.vector_layers import PolyLine
    
    # Convert timestamp to datetime if it's a string
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    # Filter data for the selected timestamp
    filtered_gdf = methane_gdf[methane_gdf['Timestamp'] == timestamp].copy()
    
    if filtered_gdf.empty:
        print(f"No data found for timestamp {timestamp}")
        return folium.Map(location=[0, 0], zoom_start=2)  # Return empty map
    
    # Calculate map center
    center_lat = filtered_gdf['Latitude'].mean()
    center_lon = filtered_gdf['Longitude'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Check if wind data is available
    has_wind_data = ('Wind_Speed (m/s)' in filtered_gdf.columns and 
                     'Wind_Direction (°)' in filtered_gdf.columns)
    
    # Get wind data if available
    if has_wind_data:
        wind_speed = filtered_gdf['Wind_Speed (m/s)'].mean()
        wind_direction = filtered_gdf['Wind_Direction (°)'].mean()
        wind_info = f"Wind: {wind_speed:.1f} m/s, {wind_direction:.0f}°"
    else:
        # Create dummy wind data for visualization purposes
        wind_speed = 3.0  # Default wind speed in m/s
        wind_direction = 270.0  # Default wind direction (West)
        wind_info = f"Wind: {wind_speed:.1f} m/s, {wind_direction:.0f}° (Default)"
    
    # Add methane concentration circles
    for _, row in filtered_gdf.iterrows():
        # Create tooltip
        tooltip_text = f"Sensor ID: {row['Sensor_ID']}<br>Methane: {row['Methane_Concentration (ppm)']:.2f} ppm"
        if has_wind_data:
            tooltip_text += f"<br>{wind_info}"
        
        # Add circle marker
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5 + (row['Methane_Concentration (ppm)'] * 2),  # Size based on concentration
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            tooltip=tooltip_text,
            popup=f"Methane: {row['Methane_Concentration (ppm)']:.2f} ppm"
        ).add_to(m)
    
    # Add wind vector arrow - ENHANCED VERSION
    # Convert wind direction to radians (meteorological to mathematical)
    wind_direction_rad = np.radians((270 - wind_direction) % 360)
    
    # Calculate the arrow end point
    # INCREASED arrow length for better visibility
    arrow_length = 0.01 * (wind_speed + 1)  # Increased scale and added base length
    end_lat = center_lat + arrow_length * np.sin(wind_direction_rad)
    end_lon = center_lon + arrow_length * np.cos(wind_direction_rad)
    
    # Add wind arrow with increased width
    PolyLine(
        locations=[[center_lat, center_lon], [end_lat, end_lon]],
        color='blue',
        weight=5,  # Increased from 3 to 5
        opacity=0.9,  # Increased opacity
        tooltip=wind_info
    ).add_to(m)
    
    # Add a larger arrow head
    folium.CircleMarker(
        location=[end_lat, end_lon],
        radius=6,  # Increased from 3 to 6
        color='blue',
        fill=True,
        fill_opacity=1.0
    ).add_to(m)
    
    # Add a SECOND arrow head slightly before the end for better arrow appearance
    # Calculate a position 90% of the way to the end
    arrow_head_lat = center_lat + 0.9 * arrow_length * np.sin(wind_direction_rad)
    arrow_head_lon = center_lon + 0.9 * arrow_length * np.cos(wind_direction_rad)
    
    # Calculate perpendicular points for arrow fins
    perp_angle1 = wind_direction_rad + np.pi/2  # +90 degrees
    perp_angle2 = wind_direction_rad - np.pi/2  # -90 degrees
    fin_length = arrow_length * 0.2
    
    fin1_lat = arrow_head_lat + fin_length * 0.3 * np.sin(perp_angle1)
    fin1_lon = arrow_head_lon + fin_length * 0.3 * np.cos(perp_angle1)
    fin2_lat = arrow_head_lat + fin_length * 0.3 * np.sin(perp_angle2)
    fin2_lon = arrow_head_lon + fin_length * 0.3 * np.cos(perp_angle2)
    
    # Add the arrow fins
    PolyLine(
        locations=[[arrow_head_lat, arrow_head_lon], [fin1_lat, fin1_lon]],
        color='blue',
        weight=4,
        opacity=0.9
    ).add_to(m)
    
    PolyLine(
        locations=[[arrow_head_lat, arrow_head_lon], [fin2_lat, fin2_lon]],
        color='blue',
        weight=4,
        opacity=0.9
    ).add_to(m)
    
    # Add a wind direction label near the center
    folium.Marker(
        location=[center_lat + 0.002, center_lon + 0.002],
        icon=folium.DivIcon(
            icon_size=(150, 36),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12pt; color: blue; font-weight: bold; background-color: rgba(255, 255, 255, 0.7); padding: 3px; border-radius: 3px;">{wind_info}</div>'
        )
    ).add_to(m)
    
    # Add a visible wind rose in the corner of the map
    wind_rose_html = f'''
    <div style="position: fixed; bottom: 120px; right: 20px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid blue; border-radius: 50%; width: 80px; height: 80px; text-align: center;">
        <div style="position: absolute; top: 5px; left: 50%; transform: translateX(-50%);">N</div>
        <div style="position: absolute; bottom: 5px; left: 50%; transform: translateX(-50%);">S</div>
        <div style="position: absolute; left: 5px; top: 50%; transform: translateY(-50%);">W</div>
        <div style="position: absolute; right: 5px; top: 50%; transform: translateY(-50%);">E</div>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
            <div style="
                transform: rotate({wind_direction}deg); 
                width: 0; 
                height: 0; 
                border-left: 20px solid transparent; 
                border-right: 20px solid transparent; 
                border-bottom: 40px solid blue; 
                position: absolute; 
                top: -20px; 
                left: -20px;
            "></div>
            <div style="
                position: absolute;
                top: -5px;
                left: -5px;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: black;
            "></div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(wind_rose_html))
    
    # Add a legend with more details
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid grey; border-radius: 5px;">
      <div><b>Methane Sensor Map</b></div>
      <div style="margin-top: 5px;">• <span style="color: red; font-size: 16px;">●</span> Methane sensors (size = concentration)</div>
      <div style="margin-top: 5px;">• <span style="color: blue; font-size: 16px;">➔</span> Wind direction and speed</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Add a simple fallback function
def create_simple_methane_map(methane_gdf, timestamp):
    """Create a simple map of methane sensor locations"""
    import folium
    import pandas as pd
    
    # Filter data for the selected timestamp
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
        
    filtered_df = methane_gdf[methane_gdf['Timestamp'] == timestamp]
    
    if filtered_df.empty:
        # Return an empty map if no data found
        return folium.Map(location=[0, 0], zoom_start=2)
    
    # Create map centered on the data
    m = folium.Map(
        location=[filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()],
        zoom_start=13
    )
    
    # Add simple markers for each sensor
    for idx, row in filtered_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color='red',
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Sensor {row['Sensor_ID']}: {row['Methane_Concentration (ppm)']:.2f} ppm"
        ).add_to(m)
    
    return m