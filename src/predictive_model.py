import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
import sys

warnings.filterwarnings("ignore")

# Add parent directory to path to import from data_processing.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_data, preprocess_methane_data, preprocess_wind_data, merge_data

def prepare_regression_data(merged_gdf):
    """
    Prepare merged data for regression modeling.
    
    Parameters:
    -----------
    merged_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing merged methane and wind data
    
    Returns:
    --------
    tuple
        (X, y, feature_names) : Features, target, and feature names
    """
    print("Preparing data for regression modeling...")
    
    # Convert timestamp to hour for temporal features
    merged_gdf = merged_gdf.copy()
    merged_gdf['Hour'] = merged_gdf['Timestamp'].dt.hour
    merged_gdf['Minute'] = merged_gdf['Timestamp'].dt.minute
    
    # Calculate hour of day as continuous feature (for cyclical patterns)
    merged_gdf['Hour_Continuous'] = merged_gdf['Hour'] + merged_gdf['Minute']/60
    
    # Create cyclical features for time of day
    merged_gdf['Hour_Sin'] = np.sin(2 * np.pi * merged_gdf['Hour_Continuous']/24)
    merged_gdf['Hour_Cos'] = np.cos(2 * np.pi * merged_gdf['Hour_Continuous']/24)
    
    # Select features for model training
    feature_columns = [
        'Wind_Speed (m/s)', 
        'Wind_Direction (°)', 
        'U', 
        'V', 
        'Hour_Sin',
        'Hour_Cos',
        'Latitude',
        'Longitude'
    ]
    
    target_column = 'Methane_Concentration (ppm)'
    
    # Extract features and target
    X = merged_gdf[feature_columns].values
    y = merged_gdf[target_column].values
    
    print(f"Created feature matrix with shape {X.shape}")
    print(f"Target vector has shape {y.shape}")
    
    return X, y, feature_columns


def train_random_forest_model(X, y):
    """
    Train a Random Forest regression model to predict methane concentration.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test, y_pred, scaler)
    """
    print("Training Random Forest regression model...")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest model metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return rf_model, X_train, X_test, y_train, y_test, y_pred, scaler


def train_xgboost_model(X, y):
    """
    Train an XGBoost regression model to predict methane concentration.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test, y_pred, scaler)
    """
    print("Training XGBoost regression model...")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    # Fit the model
    xgb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test_scaled)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"XGBoost model metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return xgb_model, X_train, X_test, y_train, y_test, y_pred, scaler


def evaluate_feature_importance(model, feature_names, model_name, output_dir):
    """
    Evaluate and visualize feature importance from a trained model.
    
    Parameters:
    -----------
    model : sklearn or xgboost model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model for plot titles and filenames
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    dict
        Dictionary with feature importance values
    """
    print(f"Evaluating feature importance for {model_name} model...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract feature importance
    importance = model.feature_importances_
    
    # Create a dictionary of feature importance
    feature_importance = dict(zip(feature_names, importance))
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Feature Importance - {model_name}', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{model_name.lower()}_feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to: {output_path}")
    
    # Save feature importance to CSV
    imp_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importance[indices]
    })
    csv_path = os.path.join(output_dir, f'{model_name.lower()}_feature_importance.csv')
    imp_df.to_csv(csv_path, index=False)
    print(f"Feature importance data saved to: {csv_path}")
    
    return feature_importance


def plot_prediction_results(y_test, y_pred, model_name, output_dir):
    """
    Plot actual vs. predicted values and residuals.
    
    Parameters:
    -----------
    y_test : numpy.ndarray
        True target values
    y_pred : numpy.ndarray
        Predicted target values
    model_name : str
        Name of the model for plot titles and filenames
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    tuple
        (actual_vs_predicted_path, residuals_path)
    """
    print(f"Creating prediction results plots for {model_name}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 1. Actual vs. Predicted plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'{model_name}: Actual vs. Predicted (RMSE={rmse:.3f}, R²={r2:.3f})', fontsize=14)
    plt.xlabel('Actual Methane Concentration (ppm)', fontsize=12)
    plt.ylabel('Predicted Methane Concentration (ppm)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    actual_vs_predicted_path = os.path.join(output_dir, f'{model_name.lower()}_actual_vs_predicted.png')
    plt.savefig(actual_vs_predicted_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name}: Residual Plot', fontsize=14)
    plt.xlabel('Predicted Methane Concentration (ppm)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    residuals_path = os.path.join(output_dir, f'{model_name.lower()}_residuals.png')
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction plots saved to: {actual_vs_predicted_path} and {residuals_path}")
    
    return actual_vs_predicted_path, residuals_path


def prepare_time_series_data(merged_gdf, sensor_id='S1'):
    """
    Prepare time series data for a specific sensor.
    
    Parameters:
    -----------
    merged_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing merged methane and wind data
    sensor_id : str
        ID of the sensor to prepare data for
    
    Returns:
    --------
    pandas.DataFrame
        Time series data for the specified sensor
    """
    print(f"Preparing time series data for sensor {sensor_id}...")
    
    # Filter data for the specific sensor
    sensor_data = merged_gdf[merged_gdf['Sensor_ID'] == sensor_id].copy()
    
    # Sort by timestamp
    sensor_data = sensor_data.sort_values('Timestamp')
    
    # Set timestamp as index
    ts_data = sensor_data.set_index('Timestamp')
    
    # Select relevant columns
    ts_cols = ['Methane_Concentration (ppm)', 'Wind_Speed (m/s)', 'Wind_Direction (°)', 'U', 'V']
    ts_data = ts_data[ts_cols]
    
    print(f"Created time series data with {len(ts_data)} rows and {len(ts_cols)} columns")
    
    return ts_data


def fit_arima_model(time_series_data, column='Methane_Concentration (ppm)'):
    """
    Fit an ARIMA model to time series data.
    
    Parameters:
    -----------
    time_series_data : pandas.DataFrame
        Time series data with datetime index
    column : str
        Column name to use for modeling
    
    Returns:
    --------
    tuple
        (arima_model, arima_order)
    """
    print(f"Fitting ARIMA model to {column} time series...")
    
    # Extract the series to model
    series = time_series_data[column]
    
    try:
        # Use auto_arima to find the best parameters
        print("Finding optimal ARIMA parameters...")
        auto_model = auto_arima(
            series,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None,  # Let auto_arima determine 'd'
            seasonal=False,
            trace=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        # Get the best order
        best_order = auto_model.order
        print(f"Best ARIMA order: {best_order}")
        
        # Fit the ARIMA model with the best order
        arima_model = ARIMA(series, order=best_order)
        fitted_model = arima_model.fit()
        
        print("ARIMA model fitting completed")
        
        return fitted_model, best_order
        
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        # Fall back to a simple order if auto_arima fails
        print("Falling back to ARIMA(1,1,1)")
        
        arima_model = ARIMA(series, order=(1,1,1))
        fitted_model = arima_model.fit()
        
        return fitted_model, (1,1,1)


def forecast_methane(arima_model, steps=12, output_dir='../outputs/forecasting'):
    """
    Generate and plot forecasts from a fitted ARIMA model.
    
    Parameters:
    -----------
    arima_model : ARIMA model
        Fitted ARIMA model
    steps : int
        Number of steps to forecast
    output_dir : str
        Directory to save output plots
    
    Returns:
    --------
    tuple
        (forecast_series, forecast_path)
    """
    print(f"Generating {steps}-step forecast from ARIMA model...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate forecast
        forecast = arima_model.forecast(steps=steps)
        
        # Create forecast dates (assuming 30-minute interval)
        last_date = arima_model.data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=30), periods=steps, freq='30T')
        
        # Create a series with the forecasts
        forecast_series = pd.Series(forecast, index=forecast_dates)
        
        # Plot historical data and forecasts
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        arima_model.data.plot(label='Historical')
        
        # Plot forecast
        forecast_series.plot(label=f'{steps}-Step Forecast', color='red')
        
        # Add confidence intervals if available
        try:
            pred = arima_model.get_forecast(steps=steps)
            conf_int = pred.conf_int()
            plt.fill_between(
                forecast_dates,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='red',
                alpha=0.1,
                label='95% Confidence Interval'
            )
        except Exception as e:
            print(f"Could not add confidence intervals: {e}")
        
        plt.title(f'Methane Concentration Forecast ({steps} steps)', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Methane Concentration (ppm)', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis date ticks
        plt.gcf().autofmt_xdate()
        
        # Save the plot
        forecast_path = os.path.join(output_dir, f'methane_forecast_{steps}_steps.png')
        plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Forecast plot saved to: {forecast_path}")
        
        return forecast_series, forecast_path
        
    except Exception as e:
        print(f"Error in forecasting: {e}")
        return None, None


def analyze_wind_methane_relationship(merged_gdf, output_dir='../outputs/analysis'):
    """
    Analyze and visualize the relationship between wind patterns and methane concentration.
    
    Parameters:
    -----------
    merged_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing merged methane and wind data
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    dict
        Dictionary with paths to saved visualizations
    """
    print("Analyzing relationship between wind patterns and methane concentration...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    
    try:
        # 1. Wind direction vs methane concentration
        plt.figure(figsize=(12, 8))
        
        # Create wind direction bins (every 15 degrees)
        merged_gdf['Wind_Direction_Bin'] = np.round(merged_gdf['Wind_Direction (°)'] / 15) * 15
        
        # Create boxplot
        sns.boxplot(x='Wind_Direction_Bin', y='Methane_Concentration (ppm)', data=merged_gdf)
        plt.title('Methane Concentration by Wind Direction', fontsize=16)
        plt.xlabel('Wind Direction (degrees)', fontsize=14)
        plt.ylabel('Methane Concentration (ppm)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        wind_dir_path = os.path.join(output_dir, 'methane_by_wind_direction.png')
        plt.savefig(wind_dir_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        output_paths['wind_direction'] = wind_dir_path
        
        # 2. Wind speed vs methane concentration
        plt.figure(figsize=(12, 8))
        
        # Create wind speed bins
        merged_gdf['Wind_Speed_Bin'] = np.round(merged_gdf['Wind_Speed (m/s)'])
        
        # Create boxplot
        sns.boxplot(x='Wind_Speed_Bin', y='Methane_Concentration (ppm)', data=merged_gdf)
        plt.title('Methane Concentration by Wind Speed', fontsize=16)
        plt.xlabel('Wind Speed (m/s)', fontsize=14)
        plt.ylabel('Methane Concentration (ppm)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        wind_speed_path = os.path.join(output_dir, 'methane_by_wind_speed.png')
        plt.savefig(wind_speed_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        output_paths['wind_speed'] = wind_speed_path
        
        # 3. Hour of day vs methane concentration, grouped by wind direction
        plt.figure(figsize=(14, 10))
        
        # Create broader wind direction categories (N, NE, E, SE, S, SW, W, NW)
        bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        merged_gdf['Wind_Direction_Category'] = pd.cut(
            merged_gdf['Wind_Direction (°)'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True, 
            right=False
        )
        
        # Create the plot
        sns.boxplot(x='Hour', y='Methane_Concentration (ppm)', hue='Wind_Direction_Category', data=merged_gdf)
        plt.title('Methane Concentration by Hour and Wind Direction', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Methane Concentration (ppm)', fontsize=14)
        plt.legend(title='Wind Direction', loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        hour_wind_path = os.path.join(output_dir, 'methane_by_hour_and_wind.png')
        plt.savefig(hour_wind_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        output_paths['hour_wind'] = hour_wind_path
        
        # 4. Create a 2D heatmap of wind direction vs. wind speed, colored by methane concentration
        plt.figure(figsize=(12, 10))
        
        # Create a pivot table
        pivot = merged_gdf.pivot_table(
            index='Wind_Speed_Bin',
            columns='Wind_Direction_Bin',
            values='Methane_Concentration (ppm)',
            aggfunc='mean'
        )
        
        # Plot heatmap
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.5)
        plt.title('Mean Methane Concentration by Wind Speed and Direction', fontsize=16)
        plt.xlabel('Wind Direction (degrees)', fontsize=14)
        plt.ylabel('Wind Speed (m/s)', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        heatmap_path = os.path.join(output_dir, 'methane_wind_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        output_paths['heatmap'] = heatmap_path
        
        print(f"Wind-methane relationship analysis completed. Saved {len(output_paths)} visualizations.")
        
        return output_paths
        
    except Exception as e:
        print(f"Error in analyzing wind-methane relationship: {e}")
        return output_paths


def main():
    """
    Main function to run the predictive modeling pipeline.
    """
    print("Running predictive modeling pipeline...")
    
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    methane_path = os.path.join(project_dir, 'data', 'methane_sensors.csv')
    wind_path = os.path.join(project_dir, 'data', 'wind_data.csv')
    output_dir = os.path.join(project_dir, 'outputs', 'modeling')
    
    # For testing with direct paths (if needed)
    if not os.path.exists(methane_path):
        methane_path = r"C:\Users\pradeep dubey\Downloads\methane_sensors.csv"
        wind_path = r"C:\Users\pradeep dubey\Downloads\wind_data.csv"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    methane_df, wind_df = load_data(methane_path, wind_path)
    methane_gdf = preprocess_methane_data(methane_df)
    wind_df_processed = preprocess_wind_data(wind_df)
    merged_gdf = merge_data(methane_gdf, wind_df_processed)
    
    # 1. Prepare data for regression modeling
    print("\n1. REGRESSION MODELING")
    print("-" * 80)
    X, y, feature_names = prepare_regression_data(merged_gdf)
    
    # 2. Train Random Forest model
    rf_model, X_train_rf, X_test_rf, y_train_rf, y_test_rf, y_pred_rf, rf_scaler = train_random_forest_model(X, y)
    
    # Evaluate Random Forest model
    rf_importance = evaluate_feature_importance(rf_model, feature_names, "RandomForest", output_dir)
    plot_prediction_results(y_test_rf, y_pred_rf, "RandomForest", output_dir)
    
    # 3. Train XGBoost model
    xgb_model, X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, y_pred_xgb, xgb_scaler = train_xgboost_model(X, y)
    
    # Evaluate XGBoost model
    xgb_importance = evaluate_feature_importance(xgb_model, feature_names, "XGBoost", output_dir)
    plot_prediction_results(y_test_xgb, y_pred_xgb, "XGBoost", output_dir)
    
    # 4. Time Series analysis and forecasting
    print("\n2. TIME SERIES ANALYSIS")
    print("-" * 80)
    
    # Choose a sensor for time series analysis
    sensor_id = 'S1'
    ts_data = prepare_time_series_data(merged_gdf, sensor_id)
    
    # Fit ARIMA model
    arima_model, arima_order = fit_arima_model(ts_data)
    
    # Generate forecasts
    forecast_steps = 12  # 6 hours ahead (with 30-minute intervals)
    forecast_series, forecast_path = forecast_methane(arima_model, steps=forecast_steps, output_dir=output_dir)
    
    # 5. Wind-Methane relationship analysis
    print("\n3. WIND-METHANE RELATIONSHIP ANALYSIS")
    print("-" * 80)
    wind_methane_visuals = analyze_wind_methane_relationship(merged_gdf, output_dir)
    
    print("\nPredictive modeling completed!")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()