import streamlit as st
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

# Expected columns for AutoGluon
AUTOGLUON_COLUMNS = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']

# Preprocessing function for AutoGluon
def preprocess_data(data):
    df = data.copy()

    # Ensure all expected columns are present
    for col in AUTOGLUON_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Select only the required columns, reordering them as needed
    df = df[AUTOGLUON_COLUMNS]
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df.set_index('DATE_TIME', inplace=True)

    return df

# Function to generate forecasts using AutoGluon
def generate_forecast(df, duration):
    df = df.reset_index()
    auto_df = pd.DataFrame()
    auto_df["timestamp"] = df["DATE_TIME"].copy()
    auto_df["target"] = df["DAILY_YIELD"].copy()
    auto_df["item_id"] = "1"
    
    data = TimeSeriesDataFrame(auto_df)
    train_data, test_data = data.train_test_split(prediction_length=duration)

    predictor = TimeSeriesPredictor(prediction_length=duration, freq="15min").fit(
        train_data, presets="chronos_large",
        hyperparameters={
            "Chronos": {
                "model_path": "tiny",
                "device": "cpu",
            }
        },
        skip_model_selection=True,
        verbosity=0,
    )
    
    predictions = predictor.predict(train_data)
    return predictions.index, predictions.values

# Streamlit app layout
st.title("Forecasting App")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    try:
        df = preprocess_data(df)
        st.success("CSV file validated and processed successfully!")

        # Duration selection
        duration = st.slider("Select duration for forecasting (in days)", 1, 365, 30)

        if st.button("Generate Forecast"):
            future_dates, forecast = generate_forecast(df, duration)

            # Plotting
            st.subheader("Forecast Results")
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['DAILY_YIELD'], label="Historical Data")
            plt.plot(future_dates, forecast, label="Forecast")
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Daily Yield")
            st.pyplot(plt)

    except ValueError as e:
        st.error(f"Error processing CSV: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
