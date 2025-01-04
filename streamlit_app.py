# Streamlit App Implementation
import streamlit as st
import pandas as pd
import time
import requests
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Load default dataset for EDA
#DEFAULT_DATASET_PATH = '/Users/sandeepkumar/NCI/AI/Moodle/Programming_For_AI/CA/CA-2-Final/FurnitureSalesForecast/notebook/cleaned_super_store_data.csv'
DEFAULT_DATASET_PATH = 'cleaned_super_store_data.csv'
def load_default_dataset():
    return pd.read_csv(DEFAULT_DATASET_PATH)

# Cache EDA report to avoid reloading
@st.cache_resource
def generate_eda_report(dataset):
    return ProfileReport(dataset, title="EDA Report", explorative=True)

# Cache initial forecast data to avoid reloading
@st.cache_data
def fetch_default_forecast():
    url = "http://127.0.0.1:5000/forecast"
    response = requests.post(url, json={"periods": 12})
    return pd.DataFrame(response.json())

# Cache generated forecast data
@st.cache_data
def fetch_forecast_data(periods):
    url = "http://127.0.0.1:5000/forecast"
    response = requests.post(url, json={"periods": periods})
    return pd.DataFrame(response.json())

# Tabs for Forecasting (Home) and EDA
tab1, tab2 = st.tabs(["Home", "EDA"])

# Forecasting Tab (Home)
with tab1:
    st.header("Furniture Sales Forecasting Dashboard")

    # Sidebar inputs
    st.sidebar.header("Forecast Settings")
    forecast_periods = st.sidebar.number_input("Forecast Periods (months)", min_value=1, max_value=36, value=12)

    # Default forecast data
    if "forecast_df" not in st.session_state:
        st.session_state["forecast_df"] = fetch_default_forecast()

    # Generate forecast button
    if st.sidebar.button("Generate Forecast"):
        st.session_state["forecast_df"] = fetch_forecast_data(forecast_periods)

    # Use the cached forecast data
    forecast_df = st.session_state["forecast_df"]

    # Format date and round values
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.date
    forecast_df[['yhat', 'yhat_lower', 'yhat_upper']] = forecast_df[['yhat', 'yhat_lower', 'yhat_upper']].round(2)

    # Plot the forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecasted Sales', color='orange', linestyle='--')
    ax.fill_between(
        forecast_df['ds'],
        forecast_df['yhat_lower'],
        forecast_df['yhat_upper'],
        color='orange',
        alpha=0.2,
        label='Confidence Interval'
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Monthly Furniture Sales Forecast")
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.legend()
    st.pyplot(fig)

    # Display forecasted dates and sales
    st.write("Forecasted Sales")
    st.table(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'Date', 
        'yhat': 'Forecasted Sales',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    }))

# EDA Tab
with tab2:
    st.header("Exploratory Data Analysis (EDA)")

    # Load default dataset initially
    dataset = load_default_dataset()

    time.sleep(2)
    # Generate and cache the profiling report
    profile = generate_eda_report(dataset)

    # Display the profiling report
    st_profile_report(profile)
