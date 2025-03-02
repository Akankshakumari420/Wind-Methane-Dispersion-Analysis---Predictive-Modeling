import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def calculate_model_performance(y_true, y_pred):
    """Calculate regression model performance metrics"""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }

def train_and_compare_models(X, y, test_size=0.2, random_state=42):
    """Train and compare multiple regression models"""
    # Split data and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define and train models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            **calculate_model_performance(y_test, y_pred),
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'scaler': scaler
        }
    
    return results

def forecast_time_series(time_series, method='ARIMA', forecast_horizon=24, params=None):
    """Forecast a time series using different methods"""
    params = params or {}
    if not isinstance(time_series.index, pd.DatetimeIndex):
        raise ValueError("Time series must have a datetime index")
    
    # Determine frequency and create forecast index
    freq = pd.infer_freq(time_series.index) or 'H'
    forecast_index = pd.date_range(start=time_series.index[-1] + pd.Timedelta(hours=1),
                                 periods=forecast_horizon, freq=freq)
    
    # Choose forecasting method and generate forecast
    if method == 'ARIMA':
        order = params.get('order', (1, 1, 1))
        model = ARIMA(time_series, order=order).fit()
        forecast_result = model.forecast(steps=forecast_horizon)
        confidence = model.get_forecast(steps=forecast_horizon).conf_int()
        lower_ci, upper_ci = confidence.iloc[:, 0], confidence.iloc[:, 1]
        
    elif method == 'ExponentialSmoothing':
        model = ExponentialSmoothing(time_series, trend=params.get('trend', 'add'),
                                   seasonal=params.get('seasonal', None),
                                   seasonal_periods=params.get('seasonal_periods', None)).fit()
        forecast_result = model.forecast(forecast_horizon)
        residual_std = model.resid.std()
        lower_ci = forecast_result - 1.96 * residual_std
        upper_ci = forecast_result + 1.96 * residual_std
        
    elif method == 'Prophet':
        try:
            from prophet import Prophet
            prophet_data = pd.DataFrame({'ds': time_series.index, 'y': time_series.values})
            model = Prophet(interval_width=0.95, **params).fit(prophet_data)
            future = model.make_future_dataframe(periods=forecast_horizon, freq=freq)
            forecast = model.predict(future)
            
            # Extract results
            f_slice = slice(-forecast_horizon, None)
            forecast_result = pd.Series(forecast.iloc[f_slice]['yhat'].values, index=forecast.iloc[f_slice]['ds'])
            lower_ci = pd.Series(forecast.iloc[f_slice]['yhat_lower'].values, index=forecast.iloc[f_slice]['ds'])
            upper_ci = pd.Series(forecast.iloc[f_slice]['yhat_upper'].values, index=forecast.iloc[f_slice]['ds'])
        except ImportError:
            raise ImportError("Prophet package not installed. Install with: pip install prophet")
    else:
        raise ValueError(f"Unknown forecasting method: {method}")
    
    return {
        'forecast': forecast_result, 'lower_ci': lower_ci, 'upper_ci': upper_ci,
        'forecast_index': forecast_index, 'method': method, 'model': model
    }

def detect_anomalies(methane_gdf, method='zscore', threshold=3.0):
    """Detect anomalies in methane concentration data"""
    result_gdf = methane_gdf.copy()
    conc = methane_gdf['Methane_Concentration (ppm)']
    
    if method == 'zscore':
        # Z-score based anomaly detection
        z_scores = (conc - conc.mean()) / conc.std()
        result_gdf['is_anomaly'] = np.abs(z_scores) > threshold
        result_gdf['anomaly_score'] = np.abs(z_scores)
        
    elif method == 'iqr':
        # IQR based anomaly detection
        q1, q3 = conc.quantile(0.25), conc.quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - threshold * iqr, q3 + threshold * iqr
        
        # Identify anomalies and calculate scores
        result_gdf['is_anomaly'] = (conc < lower_bound) | (conc > upper_bound)
        result_gdf['anomaly_score'] = 0.0
        below_mask, above_mask = conc < lower_bound, conc > upper_bound
        
        if below_mask.any():
            result_gdf.loc[below_mask, 'anomaly_score'] = (lower_bound - conc[below_mask]) / iqr
        if above_mask.any():
            result_gdf.loc[above_mask, 'anomaly_score'] = (conc[above_mask] - upper_bound) / iqr
        
    elif method == 'isolation_forest':
        # Isolation Forest based anomaly detection
        try:
            from sklearn.ensemble import IsolationForest
            X = conc.values.reshape(-1, 1)
            model = IsolationForest(contamination=threshold/100.0, random_state=42)
            result_gdf['is_anomaly'] = model.fit_predict(X) == -1
            result_gdf['anomaly_score'] = -model.score_samples(X)
        except ImportError:
            raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
    else:
        raise ValueError(f"Unknown anomaly detection method: {method}")
    
    return result_gdf
