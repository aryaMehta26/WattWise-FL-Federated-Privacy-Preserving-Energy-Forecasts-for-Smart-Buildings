import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from utils.config import load_config, get_paths

st.set_page_config(page_title="WattWise-FL Dashboard", layout="wide")

st.title("⚡ WattWise-FL: Smart Building Energy Forecasting")
st.markdown("""
**Federated, Privacy-Preserving Energy Forecasts**
This dashboard visualizes the energy consumption data and model predictions for the WattWise-FL project.
""")

# Sidebar
st.sidebar.header("Configuration")
site_id = st.sidebar.selectbox("Select Site", range(0, 19), format_func=lambda x: f"Site {x}")
meter_type = st.sidebar.selectbox("Meter Type", ["Electricity", "Chilled Water", "Steam", "Hot Water"])

# Load Data (Mock or Real)
@st.cache_data
def load_sample_data():
    # Try to load real processed data
    try:
        # This path might need adjustment based on where the app runs
        path = Path("data/processed/processed_data.parquet")
        if path.exists():
            return pd.read_parquet(path)
    except:
        pass
    
    # Fallback to synthetic data for demo
    dates = pd.date_range('2016-01-01', '2016-12-31', freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'site_id': np.random.randint(0, 5, len(dates)),
        'meter_reading': np.random.normal(100, 20, len(dates)) + np.sin(np.arange(len(dates))/24)*10,
        'prediction': np.random.normal(100, 20, len(dates)) + np.sin(np.arange(len(dates))/24)*10 + np.random.normal(0, 5, len(dates))
    })
    return df

df = load_sample_data()

# Filter Data
filtered_df = df[df['site_id'] == int(site_id)] if 'site_id' in df.columns else df

# Main Dashboard
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg Consumption", f"{filtered_df['meter_reading'].mean():.2f} kWh")
with col2:
    st.metric("Peak Consumption", f"{filtered_df['meter_reading'].max():.2f} kWh")
with col3:
    if 'prediction' in filtered_df.columns:
        mae = np.mean(np.abs(filtered_df['meter_reading'] - filtered_df['prediction']))
        st.metric("Model MAE", f"{mae:.2f}")

# Plots
st.subheader("Energy Consumption Over Time")
fig = px.line(filtered_df.iloc[:500], x='timestamp', y=['meter_reading', 'prediction'] if 'prediction' in filtered_df.columns else ['meter_reading'], 
              title=f"Site {site_id} - First 500 Hours")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Weekly Profile")
filtered_df['hour'] = filtered_df['timestamp'].dt.hour
filtered_df['dayofweek'] = filtered_df['timestamp'].dt.dayofweek
weekly_profile = filtered_df.groupby(['dayofweek', 'hour'])['meter_reading'].mean().reset_index()
fig2 = px.density_heatmap(weekly_profile, x='hour', y='dayofweek', z='meter_reading', 
                          title="Average Consumption Heatmap (Day vs Hour)")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("ℹ️ About Federated Learning")
st.info("""
**Federated Learning (FL)** is a machine learning approach where the model is trained across multiple decentralized edge devices or servers holding local data samples, without exchanging them.

In **WattWise-FL**:
1. Each building/site trains a local model on its own energy data.
2. Only the model updates (weights/gradients) are sent to a central server.
3. The server aggregates these updates to create a global model.
4. **Benefit**: Raw energy data (which reveals occupancy patterns) never leaves the building!
""")
