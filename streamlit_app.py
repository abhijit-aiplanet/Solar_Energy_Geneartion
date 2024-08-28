import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

# Define expected columns for each model
LIGHTGBM_COLUMNS = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER']
AUTOGLUON_COLUMNS = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']

# Preprocessing function
def preprocess_data(data, model_name):
    df = data.copy()

    if model_name == "LightGBM":
        expected_columns = LIGHTGBM_COLUMNS
    elif model_name == "AutoGluon":
        expected_columns = AUTOGLUON_COLUMNS
    else:
        raise ValueError("Invalid model name selected.")

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Select only the required columns, reordering them as needed
    df = df[expected_columns]

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

    return df

# Load LightGBM model
def load_lightgbm_model():
    with open("LightGBM_2.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Function to generate forecasts using the selected model
def generate_forecast(df, model_name, models, duration):
    if model_name == "LightGBM":
        df = create_features(df)
        df = df.dropna()
        X = df.drop(columns=['PLANT_ID', 'SOURCE_KEY'])

        # Load selected features
        with open('selected_features_2.pkl', 'rb') as file:
            selected_features = pickle.load(file)

        # Use only the selected features
        X = X[selected_features]

        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        model = models["LightGBM_1"]
        future_dates = pd.date_range(start=df.index[-1], periods=duration + 1, freq='D')[1:]
        X_future = X_scaled.iloc[-1:].copy()
        future_preds = []
        for i in range(duration):
            future_pred = model.predict(X_future)
            future_preds.append(future_pred[-1])
            X_future = X_future.shift(-1)
            X_future.iloc[-1, X.columns.get_loc('hour')] = future_pred[-1]
        return future_dates, future_preds, None, None

    elif model_name == "AutoGluon":
        data = TimeSeriesDataFrame(df)
        data["item_id"]="1"
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

# Model selection
model_option = st.selectbox("Select a model", ("LightGBM", "AutoGluon"))

# Load model based on selection
models = {}
if model_option == "LightGBM":
    models["LightGBM_1"] = load_lightgbm_model()

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    try:
        df = preprocess_data(df, model_option)
        st.success("CSV file validated and processed successfully!")

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
    except ValueError as e:
        st.error(f"Error processing CSV: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
