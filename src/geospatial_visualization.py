import os, sys, pandas as pd, numpy as np, folium, branca.colormap as cm
from folium.plugins import TimestampedGeoJson, HeatMap
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import tempfile, warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

def create_base_map(methane_gdf):
    center_lat, center_lon = methane_gdf.geometry.y.mean(), methane_gdf.geometry.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='OpenStreetMap')
    folium.LayerControl().add_to(m)
    return m

def add_methane_markers(m, methane_gdf, timestamp=None):
    data = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp if timestamp else methane_gdf['Timestamp'].max())].copy()
    
    # Create colormap
    vmin, vmax = methane_gdf['Methane_Concentration (ppm)'].min(), methane_gdf['Methane_Concentration (ppm)'].max()
    colormap = cm.LinearColormap(colors=['green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax, caption='Methane Concentration (ppm)')
    m.add_child(colormap)
    
    # Add markers
    methane_group = folium.FeatureGroup(name="Methane Sensors")
    for idx, row in data.iterrows():
        color = colormap(row['Methane_Concentration (ppm)'])
        popup_html = f"""
        <div style="width: 200px">
            <h4>Sensor: {row['Sensor_ID']}</h4>
            <b>Methane:</b> {row['Methane_Concentration (ppm)']:.2f} ppm<br>
            <b>Time:</b> {pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')}<br>
            <b>Location:</b> ({row.geometry.y:.5f}, {row.geometry.x:.5f})
        </div>
        """
        folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=10, color='black', weight=1,
                          fill=True, fill_color=color, fill_opacity=0.7,
                          popup=folium.Popup(popup_html, max_width=300),
                          tooltip=f"Sensor {row['Sensor_ID']}: {row['Methane_Concentration (ppm)']:.2f} ppm").add_to(methane_group)
    
    methane_group.add_to(m)
    
    # Add timestamp title
    if timestamp:
        title_html = f'''<h3 align="center" style="font-size:16px"><b>Methane Concentration Map - {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')}</b></h3>'''
        m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def add_wind_vectors(m, wind_df, methane_gdf, timestamp=None):
    # Filter data
    wind_data = wind_df[wind_df['Timestamp'] == (pd.to_datetime(timestamp) if timestamp else wind_df['Timestamp'].max())]
    if wind_data.empty: 
        print(f"No wind data available for timestamp {timestamp}")
        return m
    
    # Extract wind info
    wind_info = wind_data.iloc[0]
    wind_speed, wind_direction = wind_info['Wind_Speed (m/s)'], wind_info['Wind_Direction (°)']
    
    # Calculate U,V components
    if 'U' in wind_info and 'V' in wind_info:
        u, v = wind_info['U'], wind_info['V']
    else:
        wind_direction_rad = np.radians((wind_direction + 180) % 360)
        u, v = -wind_speed * np.sin(wind_direction_rad), -wind_speed * np.cos(wind_direction_rad)
    
    # Create feature group
    wind_group = folium.FeatureGroup(name="Wind Vectors")
    
    # Get methane data for this timestamp
    methane_data = methane_gdf[methane_gdf['Timestamp'] == (pd.to_datetime(timestamp) if timestamp else methane_gdf['Timestamp'].max())]
    
    # Add legend and vectors
    scale = 0.001
    legend_lat = methane_data.geometry.y.min() + (methane_data.geometry.y.max() - methane_data.geometry.y.min()) * 0.05
    legend_lon = methane_data.geometry.x.min() + (methane_data.geometry.x.max() - methane_data.geometry.x.min()) * 0.05
    legend_start_lat, legend_start_lon = legend_lat, legend_lon
    legend_end_lat, legend_end_lon = legend_start_lat + v * scale, legend_start_lon + u * scale
    dx, dy = legend_end_lon - legend_start_lon, legend_end_lat - legend_start_lat
    
    # Add marker and arrow
    folium.Marker(location=[legend_lat, legend_lon], icon=folium.DivIcon(icon_size=(200, 36), icon_anchor=(0, 0))).add_to(m)
    folium.RegularPolygonMarker(location=[legend_end_lat, legend_end_lon], number_of_sides=3, 
                              rotation=np.degrees(np.arctan2(dy, dx)), radius=5,
                              color='blue', fill=True, fill_color='blue').add_to(m)
    
    # Add wind vector label
    folium.Marker(location=[legend_start_lat, legend_start_lon], 
                icon=folium.DivIcon(icon_size=(150, 36), icon_anchor=(0, 0),
                                  html=f'<div style="font-size: 12pt; color: blue;">Wind: {wind_speed:.1f} m/s, {wind_direction:.0f}°</div>')).add_to(m)
    
    # Add wind vectors at each sensor location
    for idx, row in methane_data.iterrows():
        start_lat, start_lon = row.geometry.y, row.geometry.x
        end_lat, end_lon = start_lat + v * scale, start_lon + u * scale
        
        # Add line and arrow head
        folium.PolyLine(locations=[[start_lat, start_lon], [end_lat, end_lon]], 
                       color='blue', weight=2, opacity=0.6).add_to(m)
        folium.RegularPolygonMarker(location=[end_lat, end_lon], number_of_sides=3, 
                                  rotation=np.degrees(np.arctan2(v, u)), radius=3,
                                  color='blue', fill=True, fill_color='blue').add_to(m)
    
    return m

def create_time_series_map(methane_gdf, wind_df):
    # Create base map
    m = create_base_map(methane_gdf)
    
    # Create colormap
    vmin, vmax = methane_gdf['Methane_Concentration (ppm)'].min(), methane_gdf['Methane_Concentration (ppm)'].max()
    colormap = cm.LinearColormap(colors=['green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax, caption='Methane Concentration (ppm)')
    m.add_child(colormap)
    
    # Prepare features for TimestampedGeoJson
    features = []
    
    # Create features for each timestamp
    for timestamp in sorted(methane_gdf['Timestamp'].unique()):
        dt_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        ts_gdf = methane_gdf[methane_gdf['Timestamp'] == timestamp]
        ts_wind = wind_df[wind_df['Timestamp'] == timestamp]
        
        if ts_wind.empty:
            print(f"No wind data found for timestamp: {timestamp}")
            continue
            
        # Get wind data and components
        wind_speed, wind_direction = ts_wind['Wind_Speed (m/s)'].values[0], ts_wind['Wind_Direction (°)'].values[0]
        u, v = (ts_wind['U'].values[0], ts_wind['V'].values[0]) if 'U' in ts_wind.columns else (
            wind_speed * np.sin(np.radians((wind_direction + 180) % 360)), 
            wind_speed * np.cos(np.radians((wind_direction + 180) % 360))
        )
        
        # Add sensor features and wind vectors
        for _, row in ts_gdf.iterrows():
            color = colormap(row['Methane_Concentration (ppm)'])
            popup_content = f"""
            <div style="width: 200px">
                <h4>Sensor: {row['Sensor_ID']}</h4>
                <b>Methane:</b> {row['Methane_Concentration (ppm)']:.2f} ppm<br>
                <b>Time:</b> {pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')}<br>
                <b>Location:</b> ({row.geometry.y:.5f}, {row.geometry.x:.5f})
            </div>
            """
            
            # Add sensor point and wind vector
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [row.geometry.x, row.geometry.y]},
                'properties': {
                    'time': dt_str, 'popup': popup_content, 'icon': 'circle',
                    'iconstyle': {'fillColor': color, 'fillOpacity': 0.8, 'stroke': 'true', 'radius': 8, 'weight': 1}
                }
            })
            
            scale = 0.001
            end_lon, end_lat = row.geometry.x + u * scale, row.geometry.y + v * scale
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'LineString', 'coordinates': [[row.geometry.x, row.geometry.y], [end_lon, end_lat]]},
                'properties': {'time': dt_str, 'style': {'color': 'blue', 'weight': 2, 'opacity': 0.6}}
            })
    
    # Add TimestampedGeoJson layer
    TimestampedGeoJson({'type': 'FeatureCollection', 'features': features}, 
                      period='PT30M', duration='PT5M', transition_time=200, auto_play=False, loop=False).add_to(m)
    
    return m

def create_wind_rose_plot(wind_df, output_dir='../outputs/wind'):
    print("\nCreating wind rose plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Try to use windrose package
        from windrose import WindroseAxes
        import matplotlib.cm as cm
        
        fig = plt.figure(figsize=(10, 10))
        ax = WindroseAxes(fig, [0.1, 0.1, 0.8, 0.8])
        fig.add_axes(ax)
        
        ax.bar(wind_df['Wind_Direction (°)'], wind_df['Wind_Speed (m/s)'], normed=True, 
               opening=0.8, edgecolor='white', cmap=cm.viridis)
        ax.set_legend(title='Wind Speed (m/s)', loc='upper right')
        ax.set_title('Wind Rose - Direction and Speed Distribution', fontsize=14)
        
    except ImportError:
        # Fall back to regular plot if windrose not available
        print("Warning: windrose package not available. Creating standard plot instead.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wind_df['Timestamp'], wind_df['Wind_Direction (°)'], 'b-', label='Wind Direction')
        ax.set_ylabel('Wind Direction (degrees)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim([0, 360])
        
        ax2 = ax.twinx()
        ax2.plot(wind_df['Timestamp'], wind_df['Wind_Speed (m/s)'], 'r-', label='Wind Speed')
        ax2.set_ylabel('Wind Speed (m/s)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Wind Direction and Speed Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        plt.gcf().autofmt_xdate()
    
    # Save plot and return path
    output_path = os.path.join(output_dir, 'wind_rose.png' if 'WindroseAxes' in locals() else 'wind_direction_speed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Wind plot saved to: {output_path}")
    return output_path

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    methane_path = os.path.join(project_dir, 'data', 'methane_sensors.csv')
    wind_path = os.path.join(project_dir, 'data', 'wind_data.csv')
    output_dir = os.path.join(project_dir, 'outputs', 'visualization')
    
    # Use direct paths if needed
    if not os.path.exists(methane_path):
        methane_path, wind_path = r"C:\Users\Dell\Downloads\Wind-Methane-Dispersion-Analysis---Predictive-Modeling-master\Wind-Methane-Dispersion-Analysis---Predictive-Modeling-master\data\processed\methane_sensors.csv", r"C:\Users\Dell\Downloads\Wind-Methane-Dispersion-Analysis---Predictive-Modeling-master\Wind-Methane-Dispersion-Analysis---Predictive-Modeling-master\data\processed\wind_processed.csv"
    
    # Load data
    print("Loading and preprocessing data...")
    from src.data_processing import load_data, preprocess_methane_data, preprocess_wind_data
    methane_df, wind_df = load_data(methane_path, wind_path)
    methane_gdf = preprocess_methane_data(methane_df)
    wind_df_processed = preprocess_wind_data(wind_df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create maps for specific timestamps
    timestamps = [
        pd.Timestamp('2025-02-10 00:00:00'),  # Midnight
        pd.Timestamp('2025-02-10 06:00:00'),  # Morning
        pd.Timestamp('2025-02-10 12:00:00'),  # Noon (high methane)
        pd.Timestamp('2025-02-10 18:00:00')   # Evening
    ]
    
    # Process each timestamp
    for timestamp in timestamps:
        print(f"\nCreating map for timestamp: {timestamp}")
        m = create_base_map(methane_gdf)
        m = add_methane_markers(m, methane_gdf, timestamp)
        m = add_wind_vectors(m, wind_df_processed, methane_gdf, timestamp)
        
        map_path = os.path.join(output_dir, f"map_{timestamp.strftime('%Y%m%d_%H%M')}.html")
        m.save(map_path)
        print(f"Map saved to: {map_path}")
    
    # Create time series map and wind rose plot
    print("\nCreating time series map...")
    ts_map = create_time_series_map(methane_gdf, wind_df_processed)
    ts_map.save(os.path.join(output_dir, 'time_series_map.html'))
    
    wind_rose_path = create_wind_rose_plot(wind_df_processed, output_dir)
    print("\nGeospatial visualization complete!")

if __name__ == "__main__":
    print("Running geospatial visualization...")
    main()
    print("Done!")
``` 
