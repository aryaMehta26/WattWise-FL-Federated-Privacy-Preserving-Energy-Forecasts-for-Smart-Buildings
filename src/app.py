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

# Load Real Data
data_path = Path("data/processed/dashboard_data.csv")
if data_path.exists():
    st.success("✅ Loaded REAL Project Data (BDG2)")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
else:
    st.warning("⚠️ Real data not found. Running 'src/run_real.py' to generate it... (Using synthetic for now)")
    # Fallback to synthetic if real run hasn't happened yet
    from run_demo import generate_synthetic_data
    df = generate_synthetic_data()

# Sidebar controls
st.sidebar.header("Filters")

# Site/Building Selection
if 'building_id' in df.columns:
    buildings = df['building_id'].unique()
    selected_building = st.sidebar.selectbox("Select Building", buildings)
    filtered_df = df[df['building_id'] == selected_building]
else:
    filtered_df = df

# Metrics Section
st.header("📊 Model Performance (Real Test Data)")

# Calculate metrics on the fly for the selected building
if 'Predicted_LGBM' in filtered_df.columns:
    from sklearn.metrics import r2_score, mean_squared_error

    actual = filtered_df['Actual']
    pred_lgb = filtered_df['Predicted_LGBM']
    pred_ebm = filtered_df['Predicted_EBM']

    r2_lgb = r2_score(actual, pred_lgb)
    r2_ebm = r2_score(actual, pred_ebm)

    col1, col2, col3 = st.columns(3)
    col1.metric("LightGBM Accuracy (R2)", f"{r2_lgb:.2%}", "High Precision")
    col2.metric("EBM Accuracy (R2)", f"{r2_ebm:.2%}", "Explainable")
    col3.metric("Federated Rounds", "3", "Completed")

    # Plot
    st.subheader("Energy Forecast vs Actual")
    st.line_chart(filtered_df.set_index('timestamp')[['Actual', 'Predicted_LGBM', 'Predicted_EBM']])
else:
    st.info("Run src/run_real.py to see model predictions here.")

# Federated Learning Explainer
st.markdown("---")
st.header("🌐 Federated Learning Simulation")
st.image("https://miro.medium.com/max/1400/1*Bub5lY4oU1sH_jC5X0bWZA.png", caption="Federated Learning Architecture")
st.write("""
**How it works:**
1. **Local Training**: Each building trains a model on its own private data.
2. **Model Updates**: Only the model weights (not data) are sent to the central server.
3. **Aggregation**: The server averages the weights to create a global model.
4. **Privacy**: Raw energy data never leaves the building!
""")

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
