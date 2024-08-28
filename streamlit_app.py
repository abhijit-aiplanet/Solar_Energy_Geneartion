import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define expected columns for LightGBM
LIGHTGBM_COLUMNS = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER']

# Preprocessing function for LightGBM
def preprocess_data(data):
    df = data.copy()

    # Ensure all expected columns are present
    for col in LIGHTGBM_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Select only the required columns, reordering them as needed
    df = df[LIGHTGBM_COLUMNS]

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df.set_index('DATE_TIME', inplace=True)

    return df

# Feature engineering for LightGBM
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

    return df

# Load LightGBM model
def load_lightgbm_model():
    with open("LightGBM_2.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Function to generate forecasts using LightGBM
def generate_forecast(df, models, duration):
    df = create_features(df)
    df = df.dropna()
    X = df.drop(columns=['PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER'])

    # Load selected features
    with open('selected_features_2.pkl', 'rb') as file:
        selected_features = pickle.load(file)

    # Use only the selected features
    X = X[selected_features]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    model = models["LightGBM_1"]
    future_dates = pd.date_range(start=df.index[-1], periods=duration + 1, freq='15min')[1:]
    X_future = X_scaled.iloc[-1:].copy()
    future_preds = []
    for i in range(duration):
        future_pred = model.predict(X_future)
        future_preds.append(future_pred[-1])
        X_future = X_future.shift(-1)
        X_future.iloc[-1, X.columns.get_loc('hour')] = future_pred[-1]
    return future_dates, future_preds, None, None

# Streamlit app layout
st.title("Forecasting App")

# Load LightGBM model
models = {"LightGBM_1": load_lightgbm_model()}

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    try:
        df = preprocess_data(df)
        st.success("CSV file validated and processed successfully!")

        # Duration selection
        duration = st.slider("Select duration for forecasting (in days)", 1, 100, 30)

        if st.button("Generate Forecast"):
            future_dates, forecast, mae, r2 = generate_forecast(df, models, duration)

            # Plotting
            st.subheader("Forecast Results")
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['AC_POWER'], label="Historical Data")
            plt.plot(future_dates, forecast, label="Forecast")
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("AC Power")
            st.pyplot(plt)

            # Displaying metrics (if available)
            if mae is not None and r2 is not None:
                st.subheader("Metrics")
                st.write(f"MAE: {mae}")
                st.write(f"R²: {r2}")
    except ValueError as e:
        st.error(f"Error processing CSV: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
