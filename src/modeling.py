import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def show_ml_models_tab(merged_gdf):
    """
    Display machine learning models for methane prediction
    
    Parameters:
    -----------
    merged_gdf : GeoDataFrame
        The merged geodataframe containing methane and wind data
    """
    st.subheader("Machine Learning Models for Methane Prediction")
    
    # Prepare data for modeling
    modeling_df = merged_gdf.copy()
    
    # Drop geometry and unnecessary columns
    if 'geometry' in modeling_df.columns:
        modeling_df = modeling_df.drop(columns=['geometry'])
    
    # Handle timestamps
    if 'Timestamp' in modeling_df.columns:
        modeling_df['Hour'] = modeling_df['Timestamp'].dt.hour
        modeling_df['Day'] = modeling_df['Timestamp'].dt.day
        modeling_df['Month'] = modeling_df['Timestamp'].dt.month
        modeling_df = modeling_df.drop(columns=['Timestamp'])
    
    # Ensure all data is numeric
    modeling_df = pd.get_dummies(modeling_df, drop_first=True)
    
    # Define features and target
    X = modeling_df.drop(columns=['Methane_Concentration (ppm)'])
    y = modeling_df['Methane_Concentration (ppm)']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }
    
    # Display model comparison
    st.write("### Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results).T
    
    # Display metrics table
    st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE']).highlight_max(axis=0, subset=['R²']))
    
    # Plot comparison
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    comparison_df['RMSE'].plot(kind='bar', ax=ax[0], color='skyblue')
    ax[0].set_title('RMSE Comparison')
    ax[0].set_ylabel('RMSE')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # R² comparison
    comparison_df['R²'].plot(kind='bar', ax=ax[1], color='lightgreen')
    ax[1].set_title('R² Comparison')
    ax[1].set_ylabel('R²')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature importance for best model
    st.write("### Feature Importance")
    
    if 'Random Forest' in models:
        # Use Random Forest for feature importance
        model = models['Random Forest']
        
        # Extract feature importance
        importances = model.feature_importances_
        feature_names = X.columns
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot top 10 features
        top_n = min(10, len(feature_names))
        
        plt.figure(figsize=(10, 6))
        plt.title('Top Feature Importance')
        plt.bar(range(top_n), importances[indices][:top_n], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

def show_time_series_tab():
    """Legacy function for compatibility"""
    st.warning("This function has been moved to forecasting.py")

def show_feature_analysis_tab():
    """Legacy function for compatibility"""
    st.warning("This function has been moved to analysis.py")

# Export all tabs for use in dashboard
__all__ = ['show_ml_models_tab', 'show_time_series_tab', 'show_feature_analysis_tab']