import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import plotly.express as px
import plotly.graph_objects as go

def show_forecasting_tab(methane_gdf):
    """
    Display time series forecasting for methane concentration data
    
    Parameters:
    -----------
    methane_gdf : GeoDataFrame
        GeoDataFrame containing methane sensor data
    """
    st.subheader("Time Series Forecasting")
    
    # Sensor selection
    sensors = sorted(methane_gdf['Sensor_ID'].unique())
    selected_sensor = st.selectbox("Select sensor for forecasting:", sensors)
    
    # Filter data for selected sensor
    sensor_data = methane_gdf[methane_gdf['Sensor_ID'] == selected_sensor].copy()
    
    # Set timestamp as index for time series analysis
    ts_data = sensor_data.set_index('Timestamp')['Methane_Concentration (ppm)']
    
    # Plot the time series data
    st.write("### Methane Concentration Time Series")
    fig = px.line(ts_data, labels={"value": "Methane Concentration (ppm)", "Timestamp": "Date"})
    st.plotly_chart(fig)
    
    # Time series decomposition
    st.write("### Time Series Decomposition")
    
    try:
        # Resample to regular intervals if needed
        ts_regular = ts_data.resample('1H').mean().interpolate(method='linear')
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_regular, model='additive', period=24)
        
        # Create figure for decomposition plots
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        # Original series
        decomposition.observed.plot(ax=axes[0])
        axes[0].set_ylabel('Observed')
        axes[0].set_title('Time Series Decomposition')
        
        # Trend component
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_ylabel('Trend')
        
        # Seasonal component
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_ylabel('Seasonal')
        
        # Residual component
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_ylabel('Residual')
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in decomposition: {str(e)}")
        st.warning("Time series decomposition requires regular time intervals and sufficient data. Try selecting a sensor with more data points.")
    
    # ARIMA forecasting
    st.write("### ARIMA Forecasting")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("p (AR order)", 0, 5, 1)
    with col2:
        d = st.number_input("d (Differencing)", 0, 2, 1)
    with col3:
        q = st.number_input("q (MA order)", 0, 5, 1)
    
    forecast_periods = st.slider("Forecast periods ahead", 1, 24, 6)
    
    try:
        # Prepare data for ARIMA
        train_data = ts_regular.dropna()
        
        # Fit ARIMA model
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast
        forecast_result = model_fit.forecast(steps=forecast_periods)
        
        # Create forecast index
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=forecast_periods+1, freq='1H')[1:]
        
        # Create plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data.values,
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_result.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add confidence intervals
        if hasattr(forecast_result, 'conf_int'):
            conf_int = forecast_result.conf_int()
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=conf_int.iloc[:, 0],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=conf_int.iloc[:, 1],
                mode='lines',
                fill='tonexty',
                name='95% Confidence Interval',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)'
            ))
        
        fig.update_layout(
            title=f"ARIMA({p},{d},{q}) Forecast for Sensor {selected_sensor}",
            xaxis_title="Date",
            yaxis_title="Methane Concentration (ppm)",
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig)
        
        # Show model summary
        with st.expander("ARIMA Model Summary"):
            st.text(str(model_fit.summary()))
    
    except Exception as e:
        st.error(f"Error in ARIMA forecasting: {str(e)}")
        st.warning("Try different p, d, q parameters or select a sensor with more regular data.")
