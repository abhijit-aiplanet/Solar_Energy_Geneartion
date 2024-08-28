import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from prophet import Prophet
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

# Define expected columns
EXPECTED_COLUMNS = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']

# Preprocessing function
def preprocess_data(data):
    df = data.copy()
    df.drop(['PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER', 'TOTAL_YIELD'], axis=1, inplace=True)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df.set_index('DATE_TIME', inplace=True)
    return df

# Enhanced feature engineering for LightGBM
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    for lag in [1, 2, 3, 24, 48, 72, 96]:
        df[f'yield_lag_{lag}'] = df['DAILY_YIELD'].shift(lag)
    
    for window in [6, 12, 24, 48]:
        df[f'yield_rolling_mean_{window}'] = df['DAILY_YIELD'].rolling(window=window).mean()
        df[f'yield_rolling_std_{window}'] = df['DAILY_YIELD'].rolling(window=window).std()
    
    return df

# Train ARIMA model
def train_arima(df):
    model = SARIMAX(df['DAILY_YIELD'], 
                    order=(0, 1, 1), 
                    seasonal_order=(1, 1, 0, 48))
    model_fit = model.fit(disp=False)
    return model_fit

# Load LightGBM, Prophet models (AutoGluon will do zero-shot inference)
def load_all_models():
    models = {}
    model_names = ["LightGBM", "Prophet"]
    for model_name in model_names:
        with open(f"{model_name}.pkl", "rb") as file:
            models[model_name] = pickle.load(file)
    return models

# Function to generate forecasts using the selected model
def generate_forecast(df, model_name, models, duration):
    if model_name == "ARIMA":
        # ARIMA Model
        from statsmodels.tsa.arima.model import ARIMA
        df = preprocess_data(df)
        model = ARIMA(df['DAILY_YIELD'], order=(0,1,1), seasonal_order=(1,1,0,48))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=duration)
        future_dates = pd.date_range(start=df.index[-1], periods=duration + 1, freq='D')[1:]
        return future_dates, forecast, None, None

    elif model_name == "LightGBM":
        # LightGBM Model
        df = create_features(df)
        df = df.dropna()
        y = df['DAILY_YIELD']
        X = df.drop('DAILY_YIELD', axis=1)

        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        model = models["LightGBM"]
        future_dates = pd.date_range(start=df.index[-1], periods=duration + 1, freq='D')[1:]
        X_future = X_scaled.iloc[-1:].copy()
        future_preds = []
        for i in range(duration):
            future_pred = model.predict(X_future)
            future_preds.append(future_pred[-1])
            X_future = X_future.shift(-1)
            X_future.iloc[-1, X.columns.get_loc('yield_lag_1')] = future_pred[-1]
        return future_dates, future_preds, None, None

    elif model_name == "Prophet":
        # Prophet Model
        df = df.reset_index().rename(columns={"DATE_TIME": "ds", "DAILY_YIELD": "y"})
        model = models["Prophet"]
        future = model.make_future_dataframe(periods=duration, freq='D')
        forecast = model.predict(future)
        return forecast['ds'].tail(duration), forecast['yhat'].tail(duration), None, None

    elif model_name == "AutoGluon":
        # AutoGluon Model
        data = TimeSeriesDataFrame(df)
        train_data, test_data = data.train_test_split(prediction_length=duration)

        predictor = TimeSeriesPredictor(prediction_length=duration).fit(
            train_data, presets="chronos_large",
            skip_model_selection=True,
            verbosity=0,
        )
        predictions = predictor.predict(train_data)
        return predictions.index, predictions.values, None, None

# Streamlit app layout
st.title("Forecasting App")

# Load all models at the start
models = load_all_models()

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if list(df.columns) != EXPECTED_COLUMNS:
        st.error(f"The CSV must contain the following columns: {EXPECTED_COLUMNS}")
    else:
        st.success("CSV file validated successfully!")
        df = preprocess_data(df)

        # Model selection
        model_option = st.selectbox("Select a model", ("ARIMA", "LightGBM", "Prophet", "AutoGluon"))

        # Duration selection
        duration = st.slider("Select duration for forecasting (in days)", 1, 365, 30)

        if st.button("Generate Forecast"):
            future_dates, forecast, mae, r2 = generate_forecast(df, model_option, models, duration)

            # Plotting
            st.subheader("Forecast Results")
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['DAILY_YIELD'], label="Historical Data")
            plt.plot(future_dates, forecast, label="Forecast")
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Daily Yield")
            st.pyplot(plt)

            # Displaying metrics (if available)
            if mae is not None and r2 is not None:
                st.subheader("Metrics")
                st.write(f"MAE: {mae}")
                st.write(f"R²: {r2}")
