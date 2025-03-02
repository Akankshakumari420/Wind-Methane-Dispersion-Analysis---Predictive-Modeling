# Wind-Methane Dispersion Analysis & Predictive Modeling

This project is developed as part of a GIS & Data Science Interview Task to analyze methane emissions dispersion using local sensor data and wind measurements. The solution integrates geospatial analyses, predictive modeling, and interactive visualizations into a unified dashboard.

---

## Table of Contents

- [Wind-Methane Dispersion Analysis \& Predictive Modeling](#wind-methane-dispersion-analysis--predictive-modeling)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Project Structure](#project-structure)
  - [Features](#features)
  - [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Running the Dashboard](#running-the-dashboard)
- [Methane Analysis Project](#methane-analysis-project)
  - [Overview](#overview)
  - [External Data Validation Feature](#external-data-validation-feature)
    - [Supported API Providers](#supported-api-providers)
    - [How to Use](#how-to-use)
    - [Configuration](#configuration)
  - [Running the Application](#running-the-application)

---

## Project Overview

The project aims to analyze methane emissions at an industrial site in the Permian Basin, Texas. Sensors record methane concentration every 30 minutes, and an anemometer records wind speed and direction simultaneously. The objectives include:

- Geospatial processing of methane sensor and wind data.
- Visualizing sensor locations, wind vectors, and wind rose plots using GIS tools.
- Performing spatial interpolation and clustering to identify high-risk zones.
- Building predictive models using machine learning (Random Forest, ARIMA, etc.) to estimate methane dispersion.
- Integrating external climate API data to validate local wind measurements.
- Creating an interactive dashboard for visual exploration.

---

## Project Structure

## Features

- **Data Processing**: Load, clean, and preprocess methane and wind data
- **Geospatial Visualization**: Create interactive maps and heatmaps
- **Interpolation**: Apply IDW, RBF, and Kriging interpolation methods
- **Clustering**: Identify hotspots using DBSCAN and KMeans clustering
- **Time Series Analysis**: Analyze temporal patterns and forecast future values
- **Dashboard**: Interactive Streamlit dashboard for data exploration

## Quick Start

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/methane-analysis.git
   cd methane-analysis
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

Start the interactive dashboard:

# Methane Analysis Project

## Overview
This project provides tools for analyzing methane sensor data and correlating it with wind measurements to understand methane dispersion patterns.

## External Data Validation Feature

The new external data validation feature allows users to compare local wind measurements with data from multiple external weather APIs to:

- Validate the accuracy of local sensors
- Identify potential measurement errors or anomalies
- Provide additional confidence in analysis results

### Supported API Providers
- OpenWeatherMap
- Visual Crossing
- Weatherbit

### How to Use

1. Navigate to the "External Validation" section in the dashboard
2. Enter your API keys in the sidebar configuration panel
3. Select a timestamp for validation
4. Click "Validate Wind Data" to fetch external data and compare with local measurements
5. Review the comparison charts and discrepancy flags

### Configuration

You'll need to obtain API keys from one or more of the supported weather data providers:
- [OpenWeatherMap](https://openweathermap.org/api)
- [Visual Crossing](https://www.visualcrossing.com/weather-api)
- [Weatherbit](https://www.weatherbit.io/api)

## Running the Application
