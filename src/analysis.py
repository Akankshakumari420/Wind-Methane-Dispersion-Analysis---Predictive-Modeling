import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols

def calculate_model_performance(y_true, y_pred, model_name=None):
    """
    Calculate performance metrics for model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str, optional
        Name of the model being evaluated
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    results = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Calculate additional metrics
    # Mean absolute percentage error (MAPE)
    mask = y_true != 0  # Avoid division by zero
    if np.any(mask):
        results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        results['mape'] = np.nan
        
    # Store model name if provided
    if model_name:
        results['model'] = model_name
        
    return results

def convert_wind_to_uv(wind_speed, wind_direction):
    """Convert wind speed and direction to U and V components"""
    # Convert to radians and compute components
    rads = np.radians(90 - wind_direction)
    u = -wind_speed * np.cos(rads)  # u is positive eastward
    v = -wind_speed * np.sin(rads)  # v is positive northward
    return u, v

def create_time_features(df, timestamp_col='Timestamp'):
    """
    Extract time-based features from timestamp column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the timestamp column
    timestamp_col : str, optional
        Name of the timestamp column, default is 'Timestamp'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with new time-based features added
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
    
    # Extract basic time components
    result_df['hour'] = result_df[timestamp_col].dt.hour
    result_df['day'] = result_df[timestamp_col].dt.day
    result_df['month'] = result_df[timestamp_col].dt.month
    result_df['year'] = result_df[timestamp_col].dt.year
    result_df['day_of_week'] = result_df[timestamp_col].dt.dayofweek
    result_df['day_of_year'] = result_df[timestamp_col].dt.dayofyear
    
    # Is weekend or not (0 = Monday, 6 = Sunday)
    result_df['is_weekend'] = result_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Time of day features (morning, afternoon, evening, night)
    result_df['time_of_day'] = result_df['hour'].apply(
        lambda x: 'morning' if 5 <= x < 12 else
                  'afternoon' if 12 <= x < 17 else
                  'evening' if 17 <= x < 21 else 'night'
    )
    
    # Cyclic encoding for hour and month (since they're cyclical features)
    result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
    result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
    
    return result_df

def show_feature_analysis_tab(merged_gdf=None):
    """Display feature analysis tab in the dashboard"""
    st.write("## Feature Analysis")
    
    # Load data if not provided
    if merged_gdf is None:
        st.error("No data provided for feature analysis")
        return
    
    # Make a copy to avoid modifying the original
    df = merged_gdf.copy().reset_index(drop=True)
    
    # Prepare regression_df early (before any conditional blocks) 
    # to avoid UnboundLocalError
    regression_df = df.copy()
    regression_df = regression_df.rename(columns={
        'Methane_Concentration (ppm)': 'Methane',
        'Wind_Speed (m/s)': 'WindSpeed',
        'Wind_Direction (°)': 'WindDirection'
    })
    
    # Extract hour for time analysis
    if 'Timestamp' in regression_df.columns:
        regression_df['Hour'] = regression_df['Timestamp'].dt.hour
    
    # Use streamlit to create analysis tools
    st.subheader("Relationship Between Variables")
    
    # Select analysis type
    analysis_type = st.radio(
        "Select analysis type:",
        ["Wind Analysis", "Location Analysis", "Time Analysis", "Advanced Regression"]
    )
    
    if analysis_type == "Wind Analysis":
        st.write("### Wind Speed and Methane Concentration")
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            df["Wind_Speed (m/s)"], 
            df["Methane_Concentration (ppm)"],
            alpha=0.6
        )
        
        # Add trend line
        m, b = np.polyfit(df["Wind_Speed (m/s)"], df["Methane_Concentration (ppm)"], 1)
        ax.plot(
            df["Wind_Speed (m/s)"].sort_values(), 
            m * df["Wind_Speed (m/s)"].sort_values() + b, 
            'r--'
        )
        
        # Add correlation coefficient
        corr = df["Wind_Speed (m/s)"].corr(df["Methane_Concentration (ppm)"])
        ax.text(
            0.05, 0.95, 
            f"Correlation: {corr:.2f}", 
            transform=ax.transAxes, 
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5)
        )
        
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Methane Concentration (ppm)")
        ax.set_title("Methane Concentration vs Wind Speed")
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        # Wind direction analysis
        st.write("### Wind Direction and Methane Concentration")
        
        # Create wind direction bins
        bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        # Add wind direction category
        df['Wind_Direction_Cat'] = pd.cut(
            df['Wind_Direction (°)'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Wind_Direction_Cat', y='Methane_Concentration (ppm)', data=df, ax=ax)
        ax.set_xlabel("Wind Direction")
        ax.set_ylabel("Methane Concentration (ppm)")
        ax.set_title("Methane Concentration by Wind Direction")
        
        st.pyplot(fig)
        
    elif analysis_type == "Location Analysis":
        st.write("### Location and Methane Concentration")
        
        # Create scatter plot on map coordinates
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            df['Longitude'], 
            df['Latitude'],
            c=df['Methane_Concentration (ppm)'],
            cmap='YlOrRd',
            alpha=0.7,
            s=100,
            edgecolor='black'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Methane Concentration (ppm)')
        
        # Add sensor IDs
        for i, row in df.iterrows():
            if i % 30 == 0:  # Plot every 30th point to avoid crowding
                ax.annotate(
                    f"{row['Sensor_ID']}",
                    (row['Longitude'], row['Latitude']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Methane Concentration by Location")
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        # Linear relationship between coordinates and concentration
        st.write("#### Linear Relationship between Location and Methane Concentration")
        
        # Create linear model
        from sklearn.linear_model import LinearRegression
        X = df[['Longitude', 'Latitude']]
        y = df['Methane_Concentration (ppm)']
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate and display metrics
        y_pred = model.predict(X)
        metrics = calculate_model_performance(y, y_pred)
        
        st.write(f"Longitude coefficient: {model.coef_[0]:.4f}")
        st.write(f"Latitude coefficient: {model.coef_[1]:.4f}")
        st.write(f"R² score: {metrics['r2']:.4f}")
        
    elif analysis_type == "Time Analysis":
        st.write("### Time of Day and Methane Concentration")
        
        # Extract hour
        df['Hour'] = df['Timestamp'].dt.hour
        
        # Create hour-based analysis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by hour and calculate statistics
        hour_stats = df.groupby('Hour')['Methane_Concentration (ppm)'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # Plot mean with error bars
        ax.errorbar(
            hour_stats['Hour'], 
            hour_stats['mean'], 
            yerr=hour_stats['std'],
            capsize=5,
            fmt='o-',
            label='Mean ± Std Dev'
        )
        
        # Add range
        ax.fill_between(
            hour_stats['Hour'],
            hour_stats['min'],
            hour_stats['max'],
            alpha=0.2,
            label='Min-Max Range'
        )
        
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Methane Concentration (ppm)")
        ax.set_title("Methane Concentration by Hour of Day")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        # Time and wind combined effect
        st.write("### Combined Effect of Time and Wind")
        
        # FIX: Create Wind_Direction_Cat for the Time Analysis section too
        if 'Wind_Direction_Cat' not in df.columns:
            bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
            labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            df['Wind_Direction_Cat'] = pd.cut(
                df['Wind_Direction (°)'], 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        try:
            pivot_data = df.pivot_table(
                index='Hour',
                columns='Wind_Direction_Cat',
                values='Methane_Concentration (ppm)',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.2f', ax=ax)
            ax.set_title("Mean Methane Concentration by Hour and Wind Direction")
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating pivot table: {str(e)}")
            st.write("Not enough data to create the combined analysis. Try using a dataset with more measurements.")
        
    elif analysis_type == "Advanced Regression":
        st.write("### Advanced Regression Analysis")
        
        # Show regression options
        regression_type = st.selectbox(
            "Select regression analysis:",
            ["Methane vs Wind Speed", 
             "Methane vs Wind Direction", 
             "Methane vs Location",
             "Methane vs All Factors"]
        )
        
        if regression_type == "Methane vs Wind Speed":
            # Fit model
            formula = "Methane ~ WindSpeed"
            model = ols(formula, data=regression_df).fit()
            
            # Display results
            st.write("#### OLS Regression Results")
            st.write(f"R-squared: {model.rsquared:.4f}")
            st.write(f"F-statistic: {model.fvalue:.2f}, p-value: {model.f_pvalue:.4f}")
            
            # Show coefficient table
            coef_df = pd.DataFrame({
                'Variable': ['Intercept', 'Wind Speed'],
                'Coefficient': model.params,
                'Std Error': model.bse,
                'P-value': model.pvalues
            })
            st.write(coef_df)
            
            # Plot regression
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(
                x='WindSpeed', 
                y='Methane', 
                data=regression_df,
                scatter_kws={'alpha': 0.5},
                ax=ax
            )
            
            ax.set_xlabel("Wind Speed (m/s)")
            ax.set_ylabel("Methane Concentration (ppm)")
            ax.set_title("Linear Regression: Methane vs Wind Speed")
            
            st.pyplot(fig)
            
        elif regression_type == "Methane vs Wind Direction":
            # Create cyclical features for direction
            regression_df['WindDir_Sin'] = np.sin(np.radians(regression_df['WindDirection']))
            regression_df['WindDir_Cos'] = np.cos(np.radians(regression_df['WindDirection']))
            
            # Fit model
            formula = "Methane ~ WindDir_Sin + WindDir_Cos"
            model = ols(formula, data=regression_df).fit()
            
            # Display results
            st.write("#### OLS Regression Results")
            st.write(f"R-squared: {model.rsquared:.4f}")
            st.write(f"F-statistic: {model.fvalue:.2f}, p-value: {model.f_pvalue:.4f}")
            
            # Show coefficient table
            coef_df = pd.DataFrame({
                'Variable': ['Intercept', 'WindDir_Sin', 'WindDir_Cos'],
                'Coefficient': model.params,
                'Std Error': model.bse,
                'P-value': model.pvalues
            })
            st.write(coef_df)
            
            # Plot direction effect
            fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='polar'))
            
            # Generate predictions for all angles
            theta = np.linspace(0, 2*np.pi, 360)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            # Predict using model coefficients
            r = model.params[0] + model.params[1] * sin_theta + model.params[2] * cos_theta
            
            # Plot
            ax.plot(theta, r, color='red')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)  # clockwise
            ax.set_title("Predicted Methane Concentration by Wind Direction")
            
            st.pyplot(fig)
            
        elif regression_type == "Methane vs Location":
            # Fit model
            formula = "Methane ~ Latitude + Longitude"
            model = ols(formula, data=regression_df).fit()
            
            # Display results
            st.write("#### OLS Regression Results")
            st.write(f"R-squared: {model.rsquared:.4f}")
            st.write(f"F-statistic: {model.fvalue:.2f}, p-value: {model.f_pvalue:.4f}")
            
            # Show coefficient table
            coef_df = pd.DataFrame({
                'Variable': ['Intercept', 'Latitude', 'Longitude'],
                'Coefficient': model.params,
                'Std Error': model.bse,
                'P-value': model.pvalues
            })
            st.write(coef_df)
            
            # Create 3D plot
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(
                regression_df['Longitude'], 
                regression_df['Latitude'],
                regression_df['Methane'],
                c=regression_df['Methane'],
                cmap='YlOrRd',
                alpha=0.7
            )
            
            # Create mesh grid for prediction surface
            x_range = np.linspace(
                regression_df['Longitude'].min(),
                regression_df['Longitude'].max(),
                20
            )
            y_range = np.linspace(
                regression_df['Latitude'].min(),
                regression_df['Latitude'].max(),
                20
            )
            xx, yy = np.meshgrid(x_range, y_range)
            
            # Predict on grid
            zz = model.params[0] + model.params[1] * yy + model.params[2] * xx
            
            # Plot surface
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Methane Concentration (ppm)')
            ax.set_title('Methane Concentration vs Location')
            
            st.pyplot(fig)
            
        elif regression_type == "Methane vs All Factors":
            # Create time features
            regression_df['Hour'] = regression_df['Timestamp'].dt.hour
            regression_df['Hour_Sin'] = np.sin(2 * np.pi * regression_df['Hour'] / 24)
            regression_df['Hour_Cos'] = np.cos(2 * np.pi * regression_df['Hour'] / 24)
            
            # Create direction features
            regression_df['WindDir_Sin'] = np.sin(np.radians(regression_df['WindDirection']))
            regression_df['WindDir_Cos'] = np.cos(np.radians(regression_df['WindDirection']))
            
            # Fit model
            formula = "Methane ~ WindSpeed + WindDir_Sin + WindDir_Cos + Hour_Sin + Hour_Cos + Latitude + Longitude"
            model = ols(formula, data=regression_df).fit()
            
            # Display results
            st.write("#### OLS Regression Results")
            st.write(f"R-squared: {model.rsquared:.4f}")
            st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
            st.write(f"F-statistic: {model.fvalue:.2f}, p-value: {model.f_pvalue:.4f}")
            
            # Show coefficient table
            coef_df = pd.DataFrame({
                'Variable': model.params.index,
                'Coefficient': model.params.values,
                'Std Error': model.bse.values,
                'P-value': model.pvalues.values,
                'Significant': model.pvalues < 0.05
            })
            st.write(coef_df)
            
            # Feature importance plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get standardized coefficients for comparison
            from sklearn.preprocessing import StandardScaler
            X = regression_df[['WindSpeed', 'WindDir_Sin', 'WindDir_Cos', 'Hour_Sin', 'Hour_Cos', 'Latitude', 'Longitude']]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            std_model = LinearRegression().fit(X_scaled, regression_df['Methane'])
            
            # Plot
            importance = pd.DataFrame({
                'Feature': ['Wind Speed', 'Wind Dir (Sin)', 'Wind Dir (Cos)', 'Hour (Sin)', 'Hour (Cos)', 'Latitude', 'Longitude'],
                'Importance': np.abs(std_model.coef_)
            })
            importance = importance.sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
            ax.set_title('Feature Importance (Standardized Coefficients)')
            
            st.pyplot(fig)
    
    # Display correlation heatmap
    st.subheader("Correlation Matrix")
    
    # Select relevant columns for correlation
    corr_cols = ['Methane', 'WindSpeed', 'WindDirection', 'Latitude', 'Longitude']
    if 'Hour' in regression_df.columns:
        corr_cols.append('Hour')
    
    # Make sure all columns exist before creating correlation matrix
    valid_cols = [col for col in corr_cols if col in regression_df.columns]
    
    if valid_cols:
        # Calculate correlation matrix
        corr_matrix = regression_df[valid_cols].corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        
        st.pyplot(fig)
    else:
        st.warning("Not enough valid columns for correlation matrix")
    
    st.write("### Key Findings")
    st.write("""
    - Wind speed shows a relationship with methane concentration
    - Wind direction affects methane dispersion patterns
    - Time of day has a cyclical effect on methane levels
    - Location is a significant factor in predicting methane concentration
    """)
