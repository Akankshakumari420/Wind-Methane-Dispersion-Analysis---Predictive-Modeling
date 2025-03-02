
"""
Main entry point for the Wind-Methane Dispersion Analysis.
This script runs the complete analysis pipeline.
"""

import os
import sys
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import load_data, preprocess_methane_data, preprocess_wind_data, merge_data, save_processed_data
from src.geospatial_visualization import create_base_map, add_methane_markers, add_wind_vectors, create_time_series_map
from src.interpolation import compare_interpolation_methods, create_interpolation_animation
from src.clustering import perform_dbscan_clustering, perform_kmeans_clustering, save_cluster_analysis
from src.predictive_model import prepare_regression_data, train_random_forest_model, evaluate_feature_importance

def run_full_pipeline(data_dir="data", output_dir="outputs"):
    """
    Run the complete analysis pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing input data
    output_dir : str
        Directory to store outputs
    """
    print("=" * 80)
    print("WIND-METHANE DISPERSION ANALYSIS")
    print("=" * 80)
    
    start_time = time.time()
    
    # Set up paths
    methane_path = os.path.join(data_dir, "methane_sensors.csv")
    wind_path = os.path.join(data_dir, "wind_data.csv")
    
    # Handle case where user specified direct paths
    if not os.path.exists(methane_path):
        methane_path = r"C:\Users\pradeep dubey\Downloads\methane_sensors.csv"
        wind_path = r"C:\Users\pradeep dubey\Downloads\wind_data.csv"
        
        if not os.path.exists(methane_path):
            print("Error: Could not find data files.")
            print("Please place methane_sensors.csv and wind_data.csv in the data directory or provide correct paths.")
            return
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "interpolation"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clustering"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "modeling"), exist_ok=True)
    
    # Step 1: Data Processing
    print("\n\n1. DATA PROCESSING")
    print("-" * 80)
    print(f"Loading data from: {methane_path} and {wind_path}")
    
    methane_df, wind_df = load_data(methane_path, wind_path)
    methane_gdf = preprocess_methane_data(methane_df)
    wind_df_processed = preprocess_wind_data(wind_df)
    merged_gdf = merge_data(methane_gdf, wind_df_processed)
    
    # Save processed data
    processed_dir = os.path.join(data_dir, "processed")
    save_processed_data(methane_gdf, wind_df_processed, merged_gdf, processed_dir)
    
    # Step 2: Geospatial Visualization
    print("\n\n2. GEOSPATIAL VISUALIZATION")
    print("-" * 80)
    
    # Create maps for key timestamps
    timestamps = [
        pd.Timestamp('2025-02-10 00:00:00'),  # Midnight
        pd.Timestamp('2025-02-10 06:00:00'),  # Morning
        pd.Timestamp('2025-02-10 12:00:00'),  # Noon (high methane)
        pd.Timestamp('2025-02-10 18:00:00')   # Evening
    ]
    
    vis_dir = os.path.join(output_dir, "visualization")
    
    print("Creating maps for key timestamps...")
    for timestamp in timestamps:
        print(f"  - Processing {timestamp}")
        m = create_base_map(methane_gdf)
        m = add_methane_markers(m, methane_gdf, timestamp)
        m = add_wind_vectors(m, wind_df_processed, methane_gdf, timestamp)
        
        map_path = os.path.join(vis_dir, f"map_{timestamp.strftime('%Y%m%d_%H%M')}.html")
        m.save(map_path)
        print(f"    Saved to {map_path}")
    
    # Create time series map
    print("\nCreating time series map...")
    ts_map = create_time_series_map(methane_gdf, wind_df_processed)
    ts_map_path = os.path.join(vis_dir, 'time_series_map.html')
    ts_map.save(ts_map_path)
    print(f"Saved to {ts_map_path}")
    
    # Step 3: Spatial Interpolation
    print("\n\n3. SPATIAL INTERPOLATION")
    print("-" * 80)
    
    interp_dir = os.path.join(output_dir, "interpolation")
    
    # Compare interpolation methods (IDW, RBF, Kriging) at noon
    noon_timestamp = pd.Timestamp('2025-02-10 12:00:00')
    print(f"Comparing interpolation methods at {noon_timestamp}...")
    compare_files = compare_interpolation_methods(methane_gdf, noon_timestamp, interp_dir)
    
    # Create animation for the high-methane period
    print("\nCreating interpolation animation for high-methane period...")
    animation_files = create_interpolation_animation(
        methane_gdf, 
        method_name='Kriging',
        start_hour=10, 
        end_hour=16, 
        interval_minutes=30,
        output_dir=interp_dir
    )
    
    # Step 4: Spatial Clustering
    print("\n\n4. SPATIAL CLUSTERING")
    print("-" * 80)
    
    clust_dir = os.path.join(output_dir, "clustering")
    
    # Perform DBSCAN clustering
    print("Performing DBSCAN clustering at noon...")
    dbscan_gdf = perform_dbscan_clustering(
        methane_gdf, 
        timestamp=noon_timestamp,
        eps=0.0005,
        min_samples=2,
        use_concentration=True
    )
    
    # Save DBSCAN results
    dbscan_paths = save_cluster_analysis(dbscan_gdf, clust_dir, 'DBSCAN', noon_timestamp)
    
    # Perform KMeans clustering
    print("\nPerforming KMeans clustering at noon...")
    kmeans_gdf = perform_kmeans_clustering(
        methane_gdf, 
        timestamp=noon_timestamp,
        n_clusters=3,
        use_concentration=True
    )
    
    # Save KMeans results
    kmeans_paths = save_cluster_analysis(kmeans_gdf, clust_dir, 'KMeans', noon_timestamp)
    
    # Step 5: Predictive Modeling
    print("\n\n5. PREDICTIVE MODELING")
    print("-" * 80)
    
    model_dir = os.path.join(output_dir, "modeling")
    
    # Prepare data for regression modeling
    print("Preparing data for regression modeling...")
    X, y, feature_names = prepare_regression_data(merged_gdf)
    
    # Train Random Forest model
    print("\nTraining Random Forest regression model...")
    rf_model, _, _, _, y_test_rf, y_pred_rf, _ = train_random_forest_model(X, y)
    
    # Evaluate feature importance
    print("\nEvaluating feature importance...")
    rf_feat_importance = evaluate_feature_importance(rf_model, feature_names, "RandomForest", model_dir)
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE! (Time elapsed: {elapsed_time:.2f} seconds)")
    print("=" * 80)
    print("\nOutput files are available in:")
    print(f"- {os.path.abspath(output_dir)}")
    print("\nTo run the interactive dashboard:")
    print("$ streamlit run src/dashboard.py")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Wind-Methane Dispersion Analysis pipeline.")
    parser.add_argument('--data-dir', default="data", help="Directory containing input data files")
    parser.add_argument('--output-dir', default="outputs", help="Directory to store output files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_full_pipeline(args.data_dir, args.output_dir)
