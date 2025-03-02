import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap
import sys
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon, Point
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import branca.colormap as cm
# Add parent directory to path to import from data_processing.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_data, preprocess_methane_data, preprocess_wind_data, merge_data

def perform_dbscan_clustering(methane_gdf, timestamp=None, eps=0.0005, min_samples=3, use_concentration=True):
    """
    Perform DBSCAN clustering on methane data to identify high-risk zones.
    
    Parameters:
    -----------
    methane_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing methane sensor data with geometries
    timestamp : datetime or str, optional
        If provided, filter data for this specific timestamp
    eps : float
        Maximum distance between samples for them to be in same cluster
    min_samples : int
        Minimum number of samples in neighborhood for a point to be a core point
    use_concentration : bool
        Whether to include methane concentration in clustering features
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with cluster labels added
    """
    # Filter data if timestamp is provided
    if timestamp:
        gdf = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)].copy()
    else:
        # If no timestamp provided, use all data
        gdf = methane_gdf.copy()
    
    if len(gdf) < min_samples:
        print(f"Warning: Only {len(gdf)} points available, which is less than min_samples={min_samples}")
        min_samples = max(1, len(gdf) - 1)  # Adjust min_samples
    
    # Extract features for clustering
    if use_concentration:
        # Use location and methane concentration
        X = np.column_stack([
            gdf.geometry.x,
            gdf.geometry.y,
            gdf['Methane_Concentration (ppm)']
        ])
    else:
        # Use only location
        X = np.column_stack([
            gdf.geometry.x,
            gdf.geometry.y
        ])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    gdf['Cluster'] = db.fit_predict(X_scaled)
    
    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(gdf['Cluster'])) - (1 if -1 in gdf['Cluster'] else 0)
    n_noise = list(gdf['Cluster']).count(-1)
    
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    return gdf

def perform_kmeans_clustering(methane_gdf, timestamp=None, n_clusters=3, use_concentration=True):
    """
    Perform KMeans clustering on methane data.
    
    Parameters:
    -----------
    methane_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing methane sensor data with geometries
    timestamp : datetime or str, optional
        If provided, filter data for this specific timestamp
    n_clusters : int
        Number of clusters to identify
    use_concentration : bool
        Whether to include methane concentration in clustering features
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with cluster labels added
    """
    # Filter data if timestamp is provided
    if timestamp:
        gdf = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)].copy()
    else:
        # If no timestamp provided, use all data (this may not be meaningful for multiple timestamps)
        gdf = methane_gdf.copy()
    
    # Adjust n_clusters if we have fewer data points
    if len(gdf) < n_clusters:
        print(f"Warning: Only {len(gdf)} points available, which is less than n_clusters={n_clusters}")
        n_clusters = max(1, len(gdf) - 1)  # Adjust n_clusters
    
    # Extract features for clustering
    if use_concentration:
        # Use location and methane concentration
        X = np.column_stack([
            gdf.geometry.x,
            gdf.geometry.y,
            gdf['Methane_Concentration (ppm)']
        ])
    else:
        # Use only location
        X = np.column_stack([
            gdf.geometry.x,
            gdf.geometry.y
        ])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    gdf['Cluster'] = kmeans.fit_predict(X_scaled)
    
    print(f"KMeans clustering completed with {n_clusters} clusters")
    
    return gdf

def plot_clusters(gdf, method_name, timestamp=None):
    """
    Create a static plot of cluster analysis results.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with cluster labels
    method_name : str
        Name of clustering method used
    timestamp : datetime or str, optional
        Timestamp for title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique clusters
    unique_clusters = sorted(gdf['Cluster'].unique())
    
    # Create a colormap
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot each cluster with a different color
    for i, cluster_id in enumerate(unique_clusters):
        cluster_points = gdf[gdf['Cluster'] == cluster_id]
        
        if cluster_id == -1:
            # Plot noise points as black X's
            ax.scatter(
                cluster_points.geometry.x,
                cluster_points.geometry.y,
                s=60,
                c='black',
                marker='x',
                label='Noise'
            )
        else:
            # Plot cluster points
            color = cluster_colors[i % len(cluster_colors)]
            ax.scatter(
                cluster_points.geometry.x,
                cluster_points.geometry.y,
                s=80,
                c=[color],
                marker='o',
                label=f'Cluster {cluster_id}'
            )
            
            # If there are enough points, draw a convex hull
            if len(cluster_points) > 2:
                try:
                    from scipy.spatial import ConvexHull
                    points = np.array(list(zip(cluster_points.geometry.x, cluster_points.geometry.y)))
                    hull = ConvexHull(points)
                    
                    # Get hull points
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                    
                    # Plot the hull
                    ax.plot(hull_points[:, 0], hull_points[:, 1], '--', c=color, linewidth=1.5)
                except Exception as e:
                    print(f"Could not create convex hull for cluster {cluster_id}: {e}")
    
    # Add labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    if timestamp:
        ts_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')
        ax.set_title(f'{method_name} Clustering Results - {ts_str}', fontsize=14)
    else:
        ax.set_title(f'{method_name} Clustering Results', fontsize=14)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add methane concentration info
    if 'Methane_Concentration (ppm)' in gdf.columns:
        for _, row in gdf.iterrows():
            ax.annotate(
                f"{row['Methane_Concentration (ppm)']:.1f}",
                (row.geometry.x, row.geometry.y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                fontsize=8
            )
    
    plt.tight_layout()
    return fig

def create_interactive_cluster_map(gdf, timestamp=None):
    """
    Create an interactive folium map with clusters.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with cluster labels
    timestamp : datetime or str, optional
        If provided, filter data for this specific timestamp
    
    Returns:
    --------
    folium.Map
        Interactive map with clustered data
    """
    # Filter data if timestamp is provided
    if timestamp:
        filtered_gdf = gdf[gdf['Timestamp'] == pd.to_datetime(timestamp)].copy()
    else:
        # Use all data (assuming single timestamp already filtered)
        filtered_gdf = gdf.copy()
    
    # Create map centered on data
    m = folium.Map(
        location=[filtered_gdf.geometry.y.mean(), filtered_gdf.geometry.x.mean()],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Create colormap for methane concentration
    vmin = filtered_gdf['Methane_Concentration (ppm)'].min()
    vmax = filtered_gdf['Methane_Concentration (ppm)'].max()
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'orange', 'red'],
        vmin=vmin,
        vmax=vmax,
        caption='Methane Concentration (ppm)'
    )
    m.add_child(colormap)
    
    # Add heatmap layer
    heat_data = [[row.geometry.y, row.geometry.x, row['Methane_Concentration (ppm)']] 
                for idx, row in filtered_gdf.iterrows()]
    
    HeatMap(heat_data, 
            radius=20, 
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
            name='Methane Heatmap',
            show=True).add_to(m)
    
    # Create a feature group for each cluster
    unique_clusters = sorted(filtered_gdf['Cluster'].unique())
    
    for cluster_id in unique_clusters:
        cluster_points = filtered_gdf[filtered_gdf['Cluster'] == cluster_id]
        
        # Choose color based on cluster_id
        if cluster_id == -1:
            color = 'gray'  # Noise points
            group_name = 'Noise Points'
        else:
            # Get a color from viridis colormap
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
            idx = unique_clusters.index(cluster_id) if cluster_id != -1 else 0
            cluster_color = colors[idx]
            color = to_hex(cluster_color)
            group_name = f'Cluster {cluster_id}'
        
        # Create feature group for this cluster
        cluster_group = folium.FeatureGroup(name=group_name, show=True)
        
        # Add points to the group
        for idx, row in cluster_points.iterrows():
            popup_text = f"""
            <div style="width:200px">
                <h4>Sensor {row['Sensor_ID']}</h4>
                <b>Methane:</b> {row['Methane_Concentration (ppm)']:.2f} ppm<br>
                <b>Cluster:</b> {row['Cluster']}<br>
                <b>Time:</b> {pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')}
            </div>
            """
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=10,
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Sensor {row['Sensor_ID']}: {row['Methane_Concentration (ppm)']:.2f} ppm"
            ).add_to(cluster_group)
        
        # Add convex hull for clusters (except noise)
        if cluster_id != -1 and len(cluster_points) > 2:
            try:
                points = np.column_stack([
                    cluster_points.geometry.x,
                    cluster_points.geometry.y
                ])
                hull = ConvexHull(points)
                hull_points = [[points[v, 1], points[v, 0]] for v in hull.vertices]
                hull_points.append(hull_points[0])  # Close the polygon
                
                folium.PolyLine(
                    hull_points,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    tooltip=f"Cluster {cluster_id} boundary"
                ).add_to(cluster_group)
            except Exception as e:
                print(f"Could not create convex hull for cluster {cluster_id}: {e}")
        
        # Add the feature group to the map
        cluster_group.add_to(m)
    
    return m

def to_hex(rgb_color):
    """Convert RGB tuple to hex color string."""
    if len(rgb_color) == 4:  # If RGBA
        r, g, b, _ = rgb_color  # Ignore alpha
    else:
        r, g, b = rgb_color
    
    # Convert to 0-255 range and then to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(r * 255), 
        int(g * 255), 
        int(b * 255)
    )
    return hex_color

def save_cluster_analysis(gdf, output_dir, method_name, timestamp=None):
    """
    Save cluster analysis results to files.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with cluster labels
    output_dir : str
        Directory to save the results
    method_name : str
        Name of clustering method used
    timestamp : datetime or str, optional
        Timestamp for filenames
    
    Returns:
    --------
    tuple
        (static_map_path, interactive_map_path) : Paths to saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format timestamp string for filename
    if timestamp is not None:
        timestamp_str = pd.to_datetime(timestamp).strftime('%Y%m%d_%H%M')
    else:
        timestamp_str = "all_data"
    
    # Create and save static plot
    fig = plot_clusters(gdf, method_name, timestamp)
    static_map_path = os.path.join(output_dir, f"{method_name.lower()}_clustering_{timestamp_str}.png")
    fig.savefig(static_map_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Skip interactive map to avoid the error
    # Create a simple folium map instead
    try:
        # Create a simpler map without problematic elements
        m = folium.Map(
            location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()],
            zoom_start=15,
            tiles='OpenStreetMap'
        )
        
        # Add markers without complex clustering visualization
        for idx, row in gdf.iterrows():
            # Choose color based on cluster
            if row['Cluster'] == -1:
                color = 'gray'  # Noise
            else:
                color = ['blue', 'green', 'red', 'purple', 'orange'][int(row['Cluster']) % 5]
                
            # Create popup
            popup_text = f"""
            <div style="width:200px">
                <h4>Sensor {row['Sensor_ID']}</h4>
                <b>Methane:</b> {row['Methane_Concentration (ppm)']:.2f} ppm<br>
                <b>Cluster:</b> {row['Cluster']}<br>
            </div>
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=10,
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
        
        interactive_map_path = os.path.join(output_dir, f"{method_name.lower()}_clustering_interactive_{timestamp_str}.html")
        m.save(interactive_map_path)
        print(f"Saved interactive map to: {interactive_map_path}")
    except Exception as e:
        print(f"Error creating interactive map: {e}")
        interactive_map_path = None
    
    print(f"Saved static map to: {static_map_path}")
    
    return static_map_path, interactive_map_path

def analyze_high_methane_period(methane_gdf, output_dir='../outputs/clustering'):
    """
    Analyze the high-methane period using both DBSCAN and K-Means clustering.
    
    Parameters:
    -----------
    methane_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing methane sensor data
    output_dir : str
        Directory to save the results
    
    Returns:
    --------
    dict
        Dictionary with analysis results and file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find noon timestamp (peak methane time)
    noon_timestamp = pd.Timestamp('2025-02-10 12:00:00')
    
    # Perform DBSCAN clustering
    print("\nPerforming DBSCAN clustering for noon data...")
    dbscan_gdf = perform_dbscan_clustering(
        methane_gdf, 
        timestamp=noon_timestamp,
        eps=0.0005,  # Adjust based on your coordinate units
        min_samples=2,
        use_concentration=True
    )
    
    # Save DBSCAN results
    dbscan_static_path, dbscan_interactive_path = save_cluster_analysis(
        dbscan_gdf, output_dir, 'DBSCAN', noon_timestamp
    )
    
    # Perform KMeans clustering 
    print("\nPerforming KMeans clustering for noon data...")
    kmeans_gdf = perform_kmeans_clustering(
        methane_gdf, 
        timestamp=noon_timestamp,
        n_clusters=3,  # Adjust based on your data
        use_concentration=True
    )
    
    # Save KMeans results
    kmeans_static_path, kmeans_interactive_path = save_cluster_analysis(
        kmeans_gdf, output_dir, 'KMeans', noon_timestamp
    )
    
    # Compare clustering results
    results = {
        'timestamp': noon_timestamp,
        'dbscan': {
            'gdf': dbscan_gdf,
            'static_map': dbscan_static_path,
            'interactive_map': dbscan_interactive_path,
            'n_clusters': len(set(dbscan_gdf['Cluster'])) - (1 if -1 in dbscan_gdf['Cluster'] else 0),
            'n_noise': list(dbscan_gdf['Cluster']).count(-1)
        },
        'kmeans': {
            'gdf': kmeans_gdf,
            'static_map': kmeans_static_path,
            'interactive_map': kmeans_interactive_path,
            'n_clusters': len(set(kmeans_gdf['Cluster']))
        }
    }
    
    return results

def main():
    """
    Main function to run the clustering analysis.
    """
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    methane_path = os.path.join(project_dir, 'data', 'methane_sensors.csv')
    wind_path = os.path.join(project_dir, 'data', 'wind_data.csv')
    output_dir = os.path.join(project_dir, 'outputs', 'clustering')
    
    # For testing with direct paths (if needed)
    if not os.path.exists(methane_path):
        methane_path = r"C:\Users\pradeep dubey\Downloads\methane_sensors.csv"
        wind_path = r"C:\Users\pradeep dubey\Downloads\wind_data.csv"
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    methane_df, wind_df = load_data(methane_path, wind_path)
    methane_gdf = preprocess_methane_data(methane_df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze high methane period
    print("\nAnalyzing high methane period...")
    analysis_results = analyze_high_methane_period(methane_gdf, output_dir)
    
    # Print summary of findings
    print("\nClustering Analysis Summary:")
    print(f"Timestamp: {analysis_results['timestamp']}")
    print(f"DBSCAN found {analysis_results['dbscan']['n_clusters']} clusters and {analysis_results['dbscan']['n_noise']} noise points")
    print(f"KMeans identified {analysis_results['kmeans']['n_clusters']} clusters")
    
    print("\nClustering analysis complete!")

if __name__ == "__main__":
    print("Running clustering analysis...")
    main()
    print("Done!")