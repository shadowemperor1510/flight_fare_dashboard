import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ----------------------------
# 📂 Define Paths
# ----------------------------

# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_flights_data_final.csv")
DATA_DIR = os.path.join(BASE_DIR, "predictions")  # forecast CSVs

# ----------------------------
# 📊 Load Main Dataset
# ----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['route'] = df['from'] + " → " + df['to']
    return df

df = load_data()

# ----------------------------
# 🧠 Streamlit App Starts
# ----------------------------

st.set_page_config(page_title="Flight Fare Forecast", layout="wide")
st.title("✈️ Flight Fare Forecast Dashboard")

# ----------------------------
# 🔘 User Selections
# ----------------------------

col1, col2, col3 = st.columns(3)

from_city = col1.selectbox("From City", sorted(df['from'].unique()))
to_city = col2.selectbox("To City", sorted(df['to'].unique()))
flight_class = col3.selectbox("Class", sorted(df['Class'].unique()))

filtered_df = df[(df['from'] == from_city) & (df['to'] == to_city) & (df['Class'] == flight_class)]

if filtered_df.empty:
    st.warning("No data available for this route and class.")
    st.stop()

# ----------------------------
# 📈 Summary Stats
# ----------------------------

st.subheader("🔎 Summary Metrics")
avg_price = int(filtered_df['price'].mean())
min_price = int(filtered_df['price'].min())
max_price = int(filtered_df['price'].max())
popular_airline = filtered_df['airline'].value_counts().idxmax()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Fare", f"₹{avg_price}")
col2.metric("Lowest Fare", f"₹{min_price}")
col3.metric("Highest Fare", f"₹{max_price}")
col4.metric("Top Airline", popular_airline)

# ----------------------------
# 📊 Historical Price Trend
# ----------------------------

st.subheader("📅 Historical Daily Average Fare")
daily_avg = filtered_df.groupby('date')['price'].mean().reset_index()

fig_hist = px.line(
    daily_avg, x='date', y='price',
    title=f"Historical Fare Trend: {from_city} → {to_city} ({flight_class})",
    labels={'price': 'Average Price', 'date': 'Date'}
)
st.plotly_chart(fig_hist, use_container_width=True)

# ----------------------------
# 📈 Forecast Plot (Prophet)
# ----------------------------

st.subheader("🔮 Forecast for Next 30 Days")

# Create safe filename
safe_name = f"{from_city}_{to_city}".replace(" ", "_").lower()
forecast_file = os.path.join(DATA_DIR, f"forecast_{safe_name}_{flight_class.lower()}.csv")

if os.path.exists(forecast_file):
    forecast_df = pd.read_csv(forecast_file)
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    fig_forecast = px.line(
        forecast_df, x='ds', y='yhat',
        title="Prophet Forecast - Next 30 Days",
        labels={'ds': 'Date', 'yhat': 'Predicted Price'}
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.error("⚠️ Forecast not found for this route and class. Run Prophet model in Colab to generate it.")

# ----------------------------
# 📚 Optional: Airline-wise bar chart
# ----------------------------

st.subheader("🛫 Airline-wise Average Fare")
airline_avg = filtered_df.groupby('airline')['price'].mean().sort_values(ascending=False).reset_index()

fig_airline = px.bar(
    airline_avg, x='airline', y='price',
    title="Average Fare by Airline",
    labels={'airline': 'Airline', 'price': 'Average Price'}
)
st.plotly_chart(fig_airline, use_container_width=True)
