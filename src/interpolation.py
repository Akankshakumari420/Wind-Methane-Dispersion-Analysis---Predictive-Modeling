import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, geopandas as gpd
from scipy.interpolate import griddata, Rbf
import folium
from folium.plugins import HeatMap
import tempfile, contextily as ctx, matplotlib.animation as animation
from IPython.display import HTML
from pykrige.ok import OrdinaryKriging
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_data, preprocess_methane_data

warnings.filterwarnings('ignore')

def create_grid(gdf, resolution=100, buffer=0.001):
    bounds = gdf.geometry.total_bounds
    min_x, min_y = bounds[0] - buffer, bounds[1] - buffer
    max_x, max_y = bounds[2] + buffer, bounds[3] + buffer
    x_grid = np.linspace(min_x, max_x, resolution)
    y_grid = np.linspace(min_y, max_y, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.flatten(), yy.flatten()])
    return x_grid, y_grid, grid_points

def interpolate(methane_gdf, timestamp, method='idw', resolution=100, **kwargs):
    """Perform interpolation using specified method"""
    time_gdf = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)].copy()
    x_grid, y_grid, _ = create_grid(time_gdf, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.array([(point.x, point.y) for point in time_gdf.geometry])
    values = time_gdf['Methane_Concentration (ppm)'].values
    
    if method.lower() == 'idw':
        power = kwargs.get('power', 2)
        zz = np.zeros(xx.shape)
        for i in range(len(xx)):
            for j in range(len(xx[0])):
                distances = np.sqrt((xx[i,j] - points[:,0])**2 + (yy[i,j] - points[:,1])**2)
                distances[distances < 1e-10] = 1e-10
                weights = 1.0 / (distances ** power)
                zz[i,j] = np.sum(weights * values) / np.sum(weights)
    
    elif method.lower() == 'rbf':
        function = kwargs.get('function', 'multiquadric')
        epsilon = kwargs.get('epsilon', 2)
        x, y = np.array([point.x for point in time_gdf.geometry]), np.array([point.y for point in time_gdf.geometry])
        rbf = Rbf(x, y, values, function=function, epsilon=epsilon)
        zz = rbf(xx, yy)
    
    elif method.lower() == 'kriging':
        try:
            x, y = np.array([point.x for point in time_gdf.geometry]), np.array([point.y for point in time_gdf.geometry])
            variogram_model = kwargs.get('variogram_model', 'gaussian')
            OK = OrdinaryKriging(x, y, values, variogram_model=variogram_model, verbose=False, enable_plotting=False)
            z, _ = OK.execute('grid', x_grid, y_grid)
            zz = z.reshape(xx.shape)
        except Exception as e:
            print(f"Error in kriging interpolation: {e}, falling back to IDW")
            return interpolate(methane_gdf, timestamp, method='idw', resolution=resolution)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return xx, yy, zz

# Aliases for backward compatibility
idw_interpolation = lambda methane_gdf, timestamp, resolution=100, power=2: interpolate(
    methane_gdf, timestamp, method='idw', resolution=resolution, power=power)
rbf_interpolation = lambda methane_gdf, timestamp, resolution=100, function='multiquadric', epsilon=2: interpolate(
    methane_gdf, timestamp, method='rbf', resolution=resolution, function=function, epsilon=epsilon)
kriging_interpolation = lambda methane_gdf, timestamp, resolution=100, variogram_model='gaussian': interpolate(
    methane_gdf, timestamp, method='kriging', resolution=resolution, variogram_model=variogram_model)

def plot_interpolation(methane_gdf, xx, yy, zz, timestamp, method_name='IDW', add_basemap=False):
    time_gdf = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)]
    fig, ax = plt.subplots(figsize=(12, 10))
    contour = ax.contourf(xx, yy, zz, cmap='YlOrRd', levels=15)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Methane Concentration (ppm)')
    scatter = ax.scatter(time_gdf.geometry.x, time_gdf.geometry.y, c=time_gdf['Methane_Concentration (ppm)'],
                       cmap='YlOrRd', edgecolor='k', s=80)
    
    for idx, row in time_gdf.iterrows():
        ax.annotate(row['Sensor_ID'], (row.geometry.x, row.geometry.y), xytext=(5, 5),
                  textcoords="offset points", fontsize=10, fontweight='bold')
    
    ax.set_title(f'{method_name} Interpolation of Methane Concentration\n{pd.to_datetime(timestamp)}', fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=time_gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    plt.tight_layout()
    return fig

def create_folium_interpolation_map(methane_gdf, timestamp, method='IDW'):
    time_gdf = methane_gdf[methane_gdf['Timestamp'] == pd.to_datetime(timestamp)].copy()
    center_lat, center_lon = time_gdf.geometry.y.mean(), time_gdf.geometry.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='OpenStreetMap')
    
    if method == 'IDW': xx, yy, zz = idw_interpolation(methane_gdf, timestamp, resolution=50)
    elif method == 'RBF': xx, yy, zz = rbf_interpolation(methane_gdf, timestamp, resolution=50)
    elif method == 'Kriging': xx, yy, zz = kriging_interpolation(methane_gdf, timestamp, resolution=50)
    else: raise ValueError(f"Unknown interpolation method: {method}")
    
    vmin, vmax = methane_gdf['Methane_Concentration (ppm)'].min(), methane_gdf['Methane_Concentration (ppm)'].max()
    colormap = folium.LinearColormap(colors=['green', 'yellow', 'orange', 'red'], vmin=vmin, vmax=vmax,
                                  caption=f'{method} Interpolation - Methane Concentration (ppm)')
    m.add_child(colormap)
    
    heat_data = [[yy[i, j], xx[i, j], zz[i, j]] for i in range(len(xx)) for j in range(len(yy))]
    HeatMap(heat_data, radius=10, gradient={0.4: 'green', 0.65: 'yellow', 0.8: 'orange', 1: 'red'},
          min_opacity=0.5, blur=15, max_zoom=1).add_to(m)
    
    for idx, row in time_gdf.iterrows():
        color = colormap(row['Methane_Concentration (ppm)'])
        popup_html = f"""
        <div style="width: 200px">
            <h4>Sensor: {row['Sensor_ID']}</h4>
            <b>Methane:</b> {row['Methane_Concentration (ppm)']:.2f} ppm<br>
            <b>Time:</b> {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')}<br>
            <b>Location:</b> ({row.geometry.y:.5f}, {row.geometry.x:.5f})
        </div>
        """
        folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=8, color='black', weight=1,
                          fill=True, fill_color='white', fill_opacity=0.9,
                          popup=folium.Popup(popup_html, max_width=300),
                          tooltip=f"Sensor {row['Sensor_ID']}: {row['Methane_Concentration (ppm)']:.2f} ppm").add_to(m)
    return m

def compare_interpolation_methods(methane_gdf, timestamp, output_dir='../outputs/interpolation'):
    print(f"\nComparing interpolation methods for timestamp: {timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = pd.to_datetime(timestamp).strftime('%Y%m%d_%H%M')
    output_files = {}
    
    # Generate all interpolation types in a loop to reduce repeated code
    for method_name, interp_func in [('idw', idw_interpolation), ('rbf', rbf_interpolation), ('kriging', kriging_interpolation)]:
        try:
            xx, yy, zz = interp_func(methane_gdf, timestamp, resolution=100)
            fig = plot_interpolation(methane_gdf, xx, yy, zz, timestamp, method_name.upper(), add_basemap=True)
            output_path = os.path.join(output_dir, f'{method_name}_{timestamp_str}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            output_files[method_name] = output_path
            print(f"{method_name.upper()} interpolation saved to: {output_path}")
            
            # Create interactive maps for IDW and kriging
            if method_name in ['idw', 'kriging']:
                interactive_map = create_folium_interpolation_map(methane_gdf, timestamp, method=method_name.upper())
                interactive_path = os.path.join(output_dir, f'{method_name}_interactive_{timestamp_str}.html')
                interactive_map.save(interactive_path)
                output_files[f'{method_name}_interactive'] = interactive_path
                print(f"Interactive {method_name.upper()} map saved to: {interactive_path}")
        except Exception as e:
            print(f"Error in {method_name} interpolation: {e}")
    
    # Create comparison figure
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        methods = [('IDW', idw_interpolation), ('RBF', rbf_interpolation), ('Kriging', kriging_interpolation)]
        
        for i, (name, func) in enumerate(methods):
            xx, yy, zz = func(methane_gdf, timestamp)
            contour = axes[i].contourf(xx, yy, zz, cmap='YlOrRd', levels=15)
            axes[i].scatter(methane_gdf[methane_gdf['Timestamp'] == timestamp].geometry.x,
                          methane_gdf[methane_gdf['Timestamp'] == timestamp].geometry.y,
                          c='black', s=30, edgecolor='white')
            axes[i].set_title(f'{name} Interpolation')
            axes[i].set_xlabel('Longitude')
            if i == 0: axes[i].set_ylabel('Latitude')
        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(contour, cax=cbar_ax, label='Methane Concentration (ppm)')
        fig.suptitle(f'Comparison of Interpolation Methods - {timestamp}', fontsize=16, y=0.98)
        
        comparison_path = os.path.join(output_dir, f'comparison_{timestamp_str}.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_files['comparison'] = comparison_path
        print(f"Comparison plot saved to: {comparison_path}")
    except Exception as e:
        print(f"Error in creating comparison plot: {e}")
    
    return output_files

def create_interpolation_animation(methane_gdf, method_name='IDW', start_hour=0, end_hour=23, interval_minutes=30, output_dir='../outputs/interpolation'):
    print(f"\nCreating animation of {method_name} interpolation from hour {start_hour} to {end_hour}")
    os.makedirs(output_dir, exist_ok=True)
    base_date = methane_gdf['Timestamp'].min().date()
    
    # Generate timestamps
    timestamps = []
    curr_hour, curr_minute = start_hour, 0
    while curr_hour <= end_hour:
        timestamp = pd.Timestamp(f"{base_date} {curr_hour:02d}:{curr_minute:02d}:00")
        if timestamp in methane_gdf['Timestamp'].values: timestamps.append(timestamp)
        curr_minute += interval_minutes
        if curr_minute >= 60: curr_hour, curr_minute = curr_hour + 1, 0
    
    # Select interpolation function
    interp_func = {'IDW': idw_interpolation, 'RBF': rbf_interpolation, 'Kriging': kriging_interpolation}.get(
        method_name, idw_interpolation)
    
    frame_paths = []
    for i, timestamp in enumerate(timestamps):
        print(f"  Processing frame {i+1}/{len(timestamps)}: {timestamp}")
        try:
            xx, yy, zz = interp_func(methane_gdf, timestamp, resolution=80)
            fig = plot_interpolation(methane_gdf, xx, yy, zz, timestamp, method_name, add_basemap=False)
            frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
            fig.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(frame_path)
        except Exception as e:
            print(f"  Error processing frame for timestamp {timestamp}: {e}")
    
    output_files = {}
    if frame_paths:
        try:
            # Create GIF
            import imageio
            output_gif = os.path.join(output_dir, f'{method_name.lower()}_animation.gif')
            frames = [imageio.imread(path) for path in frame_paths]
            imageio.mimsave(output_gif, frames, fps=2, loop=0)
            output_files['gif'] = output_gif
            print(f"Animation saved to: {output_gif}")
            
            # Create MP4
            output_mp4 = os.path.join(output_dir, f'{method_name.lower()}_animation.mp4')
            fig, ax = plt.subplots(figsize=(8, 6))
            def update(frame): ax.clear(); ax.imshow(plt.imread(frame_paths[frame])); ax.axis('off'); return ax,
            ani = animation.FuncAnimation(fig, update, frames=len(frame_paths), blit=True)
            ani.save(output_mp4, writer='ffmpeg', fps=2)
            output_files['mp4'] = output_mp4
            print(f"MP4 animation saved to: {output_mp4}")
            
            # Clean up frame files
            for path in frame_paths:
                try: os.remove(path)
                except: pass
                
        except Exception as e:
            print(f"Error creating animation: {e}")
    
    return output_files

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    methane_path = os.path.join(project_dir, 'data', 'methane_sensors.csv')
    wind_path = os.path.join(project_dir, 'data', 'wind_data.csv')
    output_dir = os.path.join(project_dir, 'outputs', 'interpolation')
    
    if not os.path.exists(methane_path):
        methane_path, wind_path = r"C:\Users\pradeep dubey\Downloads\methane_sensors.csv", r"C:\Users\pradeep dubey\Downloads\wind_data.csv"
    
    print("Loading and preprocessing data...")
    methane_df, wind_df = load_data(methane_path, wind_path)
    methane_gdf = preprocess_methane_data(methane_df)
    os.makedirs(output_dir, exist_ok=True)
    
    noon_timestamp = pd.Timestamp('2025-02-10 12:00:00')
    print(f"\nAnalyzing interpolation methods for {noon_timestamp}...")
    output_files = compare_interpolation_methods(methane_gdf, noon_timestamp, output_dir)
    
    print("\nCreating animation for high-methane period...")
    animation_files = create_interpolation_animation(methane_gdf, method_name='Kriging', 
                                                  start_hour=10, end_hour=14, interval_minutes=30, output_dir=output_dir)
    
    print("\nInterpolation analysis complete!")

if __name__ == "__main__":
    print("Running interpolation analysis...")
    main()
    print("Done!")
