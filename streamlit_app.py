import streamlit as st
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

# Load the model and features
def load_model_and_features(model_path, features_path):
    """Load the saved LightGBM model and selected features from pickle files."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    with open(features_path, 'rb') as file:
        selected_features = pickle.load(file)
    
    return model, selected_features

# Enhanced feature engineering
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
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def preprocess_data(df, selected_features, scaler=None):
    """Preprocess the data by creating features, selecting relevant ones, and scaling them."""
    df = create_features(df)
    df = df.dropna()  # Remove rows with NaN values
    
    # Prepare features
    X = df[selected_features]
    
    # Scale the features
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    
    return X_scaled, scaler

def predict(model, X_scaled):
    """Use the trained model to make predictions."""
    return model.predict(X_scaled, num_iteration=model.best_iteration)

# Load the model and features
model_path = 'LightGBM_2.pkl'
features_path = 'selected_features_2.pkl'
model, selected_features = load_model_and_features(model_path, features_path)

# Streamlit app
st.title("LightGBM Inference App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file, index_col='DATE_TIME', parse_dates=True)
    
    # Preprocess the new data
    X_scaled, scaler = preprocess_data(new_data, selected_features)
    
    # Make predictions
    y_pred = predict(model, X_scaled)
    
    # Add predictions to the dataframe
    new_data["DAILY_YIELD"] = y_pred
    
    # Display the dataframe with predictions
    st.write(new_data)

    # Optionally, you can provide a download button for the result
    csv = new_data.to_csv(index=True)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )
