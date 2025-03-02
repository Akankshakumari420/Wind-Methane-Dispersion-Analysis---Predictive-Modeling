"""
Wind Analysis Module

This module provides functions for analyzing wind data, including wind roses,
directional histograms, and other wind-related visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.projections import polar
from datetime import datetime

def create_wind_rose(wind_df, output_path=None, dpi=300, title="Wind Rose Diagram"):
    """
    Create a wind rose plot showing the distribution of wind directions and speeds.
    
    Parameters:
    -----------
    wind_df : pandas.DataFrame
        DataFrame with wind data including 'Wind_Speed (m/s)' and 'Wind_Direction (°)'
    output_path : str, optional
        Path to save the output image
    dpi : int, optional
        Resolution of the output image
    title : str, optional
        Title for the wind rose plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The wind rose figure
    """
    try:
        # Try using the windrose package if available
        from windrose import WindroseAxes
        
        # Create figure and axes
        fig = plt.figure(figsize=(10, 10))
        rect = [0.1, 0.1, 0.8, 0.8]
        ax = WindroseAxes(fig, rect)
        fig.add_axes(ax)
        
        # Plot wind rose
        ax.bar(
            wind_df['Wind_Direction (°)'].values, 
            wind_df['Wind_Speed (m/s)'].values,
            normed=True, 
            opening=0.8, 
            edgecolor='white',
            nsector=36,  # Number of bins (10° each)
            cmap=cm.viridis
        )
        
        # Add legend
        ax.set_legend(
            title='Wind Speed (m/s)', 
            loc='upper right', 
            bbox_to_anchor=(1.1, 1.1)
        )
        
    except ImportError:
        # Fallback to manual implementation with matplotlib
        print("WindroseAxes not available. Creating a custom wind rose...")
        
        # Create figure with polar projection
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Define wind speed bins
        bins = np.arange(0, 361, 10)  # Direction bins (10° each)
        speed_bins = [0, 2, 4, 6, 8, 10, np.inf]  # Speed bins
        speed_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10+']
        
        # Convert to radians and handle directions
        # In wind roses, 0° is North, and angles increase clockwise
        # In matplotlib polar, 0° is East, and angles increase counter-clockwise
        theta = np.radians(90 - wind_df['Wind_Direction (°)'].values)
        
        # Create a histogram for each speed bin
        hist, bin_edges = np.histogram(np.degrees(theta) % 360, bins=bins)
        
        # Assign colors for different speed ranges
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(speed_bins)-1))
        
        # Plot each speed bin as a stacked bar
        width = np.radians(10)  # Width of each bar (10°)
        bottom = 0
        
        # Calculate the histogram for each speed bin
        for i, (lower, upper) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
            mask = (wind_df['Wind_Speed (m/s)'] >= lower) & (wind_df['Wind_Speed (m/s)'] < upper)
            speeds = wind_df.loc[mask, 'Wind_Direction (°)'].values
            if len(speeds) > 0:
                hist, _ = np.histogram(speeds, bins=bins)
                hist = hist / len(wind_df)  # Normalize
                bars = ax.bar(
                    np.radians(bin_edges[:-1] - 90),  # Convert to correct orientation
                    hist,
                    width=width,
                    bottom=bottom,
                    color=colors[i],
                    label=speed_labels[i]
                )
                bottom += hist
        
        # Set the direction labels correctly
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        
        # Add legend
        ax.legend(title='Wind Speed (m/s)', loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    # Add title
    plt.title(title, y=1.08, fontsize=16)
    
    # If output path is provided, save the figure
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Wind rose saved to: {output_path}")
    
    return fig

def add_wind_rose_to_dashboard(wind_df):
    """Create a wind rose plot for the Streamlit dashboard"""
    import streamlit as st
    
    st.subheader("Wind Direction and Speed Distribution")
    
    # Add filtering options
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("**Filter Options:**")
        # Add date range selector
        all_dates = pd.to_datetime(wind_df['Timestamp']).dt.date.unique()
        min_date, max_date = min(all_dates), max(all_dates)
        start_date = st.date_input("Start date", min_date)
        end_date = st.date_input("End date", max_date)
        
        # Time of day filter
        time_filter = st.multiselect(
            "Time of day",
            ["Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)", "Night (0-6)"],
            default=["Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)", "Night (0-6)"]
        )
        
    # Filter data based on selections
    filtered_df = wind_df.copy()
    if start_date and end_date:
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['Timestamp']).dt.date >= start_date) & 
            (pd.to_datetime(filtered_df['Timestamp']).dt.date <= end_date)
        ]
    
    # Apply time of day filter
    hour_filters = []
    if "Morning (6-12)" in time_filter:
        hour_filters.extend(range(6, 12))
    if "Afternoon (12-18)" in time_filter:
        hour_filters.extend(range(12, 18))
    if "Evening (18-24)" in time_filter:
        hour_filters.extend(range(18, 24))
    if "Night (0-6)" in time_filter:
        hour_filters.extend(range(0, 6))
    
    if hour_filters:
        filtered_df = filtered_df[pd.to_datetime(filtered_df['Timestamp']).dt.hour.isin(hour_filters)]
    
    with col2:
        # Create the wind rose figure with filtered data
        with st.spinner("Generating wind rose plot..."):
            if len(filtered_df) > 0:
                fig = create_wind_rose(filtered_df, title="Wind Direction and Speed Distribution")
                st.pyplot(fig)
            else:
                st.warning("No data available for the selected filters")
    
    # Add analysis section with key statistics
    st.write("### Wind Statistics")
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Speed Statistics:**")
            st.write(f"- Mean: {filtered_df['Wind_Speed (m/s)'].mean():.2f} m/s")
            st.write(f"- Median: {filtered_df['Wind_Speed (m/s)'].median():.2f} m/s")
            st.write(f"- Max: {filtered_df['Wind_Speed (m/s)'].max():.2f} m/s")
        
        with col2:
            st.write("**Direction Analysis:**")
            # Calculate dominant direction
            filtered_df['Direction_Bin'] = (filtered_df['Wind_Direction (°)'] // 10 * 10).astype(int)
            dominant_dir = filtered_df['Direction_Bin'].value_counts().idxmax()
            # Convert to compass direction
            dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            compass_dir = dirs[int(((dominant_dir + 11.25) % 360) // 22.5)]
            st.write(f"- Dominant direction: {dominant_dir}° ({compass_dir})")
            
            # Calculate stability
            dir_std = filtered_df['Wind_Direction (°)'].std()
            st.write(f"- Direction variability: {dir_std:.1f}°")
            if dir_std < 20:
                st.write("- Wind direction is relatively stable")
            else:
                st.write("- Wind direction is variable")
    else:
        st.warning("No data available to calculate statistics")
    
    return fig
