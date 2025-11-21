# 📚 WattWise-FL Learning Guide

This guide explains **what each step does**, **why it matters**, and **what you should learn from it**.

---

## 🎯 Project Overview

**Goal**: Predict hourly building energy consumption using Machine Learning while preserving privacy through Federated Learning.

**Real-World Impact**: 
- Buildings consume 40% of global energy
- Accurate forecasts enable peak shaving (reduce costs)
- Federated Learning keeps building data private (no data sharing)

---

## 📖 Step-by-Step Learning Path

### **Step 1: Data Cleaning** (`src/step1_data_cleaning.py`)

#### What It Does:
1. Loads raw electricity data from BDG2 dataset
2. Converts "wide format" (buildings as columns) → "long format" (one row per reading)
3. Merges with building metadata (age, size, use)
4. Merges with weather data (temperature, humidity)

#### Why It Matters:
- **Real data is messy**: The BDG2 dataset comes in wide format (1636 buildings = 1636 columns!)
- **Melting**: Pandas `.melt()` transforms it into a clean time-series format
- **Merging**: We need context (building type, weather) to predict energy use

#### Key Concepts to Learn:
- **Wide vs Long Format**: Long format is better for time-series ML
- **Pandas Merge**: Combining datasets on common keys (`site_id`, `timestamp`)
- **Data Provenance**: Always know where your data came from

#### Code Snippet to Study:
```python
# Wide format: Each building is a column
# timestamp | building_1 | building_2 | ...
# 2016-01-01 | 100 | 200 | ...

# Long format (what we want):
# timestamp | building_id | meter_reading
# 2016-01-01 | building_1 | 100
# 2016-01-01 | building_2 | 200

df = df.melt(id_vars=['timestamp'], var_name='building_id', value_name='meter_reading')
```

#### Output:
- `data/interim/cleaned_data.pkl` (clean, merged dataset)

---

### **Step 2: Feature Engineering** (`src/step2_feature_engineering.py`)

#### What It Does:
1. **Calendar Features**: Hour, day, month, weekend flag, cyclical encoding
2. **Weather Features**: Rolling temperature averages, humidity
3. **Building Features**: Age, size, use type
4. **Lag Features**: Past energy consumption (1h ago, 24h ago)

#### Why It Matters:
- **ML models need patterns**: Raw data doesn't show "Monday mornings use more energy"
- **Cyclical Encoding**: Hour 23 and Hour 0 are close, but numerically far (23 vs 0). We use sine/cosine to fix this.
- **Lag Features**: "What was the energy use 24 hours ago?" is a strong predictor

#### Key Concepts to Learn:
- **Feature Engineering**: The most important part of ML (better features > better models)
- **Cyclical Encoding**: `sin(2π * hour / 24)` and `cos(2π * hour / 24)`
- **Lag Features**: Use past values to predict future (time-series 101)

#### Code Snippet to Study:
```python
# Cyclical encoding for hour (0-23)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Lag feature: Energy use 24 hours ago
df['meter_reading_lag_24h'] = df.groupby('building_id')['meter_reading'].shift(24)
```

#### Output:
- `data/processed/final_features.pkl` (feature-rich dataset ready for ML)

---

### **Step 3: Model Training** (`src/step3_train_models.py`)

#### What It Does:
1. Splits data into Train (80%) and Test (20%) by time
2. Trains **LightGBM** (gradient boosting, high accuracy)
3. Trains **EBM** (explainable boosting, interpretable)
4. Runs **Federated Learning** simulation (privacy-preserving)
5. Saves models and results for the dashboard

#### Why It Matters:
- **Time-based split**: We test on future data (realistic scenario)
- **LightGBM**: Industry-standard for tabular data (fast, accurate)
- **EBM**: You can see *why* the model made a prediction (trust + debugging)
- **Federated Learning**: Trains models without sharing raw data (privacy)

#### Key Concepts to Learn:
- **Train/Test Split**: Never test on data the model has seen
- **Gradient Boosting**: Builds many weak models, combines them
- **R2 Score**: Measures "how much variance does the model explain?" (0-100%)
- **RMSLE**: Log-scale error (treats 10→20 kWh same as 100→200 kWh)

#### Code Snippet to Study:
```python
# Time-based split (no shuffling!)
split_date = dates[int(len(dates) * 0.8)]
train_df = df[df['timestamp'] < split_date]
test_df = df[df['timestamp'] >= split_date]

# Train LightGBM
lgb_model = LightGBMModel({'n_estimators': 100})
lgb_model.fit(X_train, y_train, X_test, y_test)

# Evaluate
metrics = lgb_model.evaluate(X_test, y_test)
print(f"R2 Score: {metrics['R2']:.2%}")  # 98%!
```

#### Output:
- `models/lgb_model.pkl` (trained LightGBM)
- `models/ebm_model.pkl` (trained EBM)
- `data/processed/dashboard_data.csv` (results for Streamlit)

---

### **Step 4: Dashboard** (`src/app.py`)

#### What It Does:
1. Loads the saved results from Step 3
2. Displays model accuracy (R2 Score)
3. Shows actual vs predicted energy consumption graphs
4. Explains Federated Learning with visuals

#### Why It Matters:
- **Visualization**: Stakeholders don't read code, they read dashboards
- **Streamlit**: Easy way to build interactive web apps with Python
- **Storytelling**: Good ML projects communicate results clearly

#### Key Concepts to Learn:
- **Streamlit**: `st.metric()`, `st.line_chart()`, `st.selectbox()`
- **Data Visualization**: Line charts for time-series, metrics for KPIs
- **User Experience**: Make it easy for non-technical users

#### Code Snippet to Study:
```python
# Display metrics
col1, col2 = st.columns(2)
col1.metric("LightGBM Accuracy (R2)", f"{r2_lgb:.2%}", "High Precision")
col2.metric("EBM Accuracy (R2)", f"{r2_ebm:.2%}", "Explainable")

# Plot predictions
st.line_chart(df.set_index('timestamp')[['Actual', 'Predicted_LGBM']])
```

---

## 🧠 Key Machine Learning Concepts

### 1. **Supervised Learning**
- We have labels (actual energy consumption)
- Model learns to predict labels from features
- Example: Given [hour=9, temp=20°C, building_age=10], predict energy=150 kWh

### 2. **Time-Series Forecasting**
- Data has a time dimension (hourly readings)
- Past values help predict future (lag features)
- Must split by time (not randomly!)

### 3. **Gradient Boosting (LightGBM)**
- Builds many decision trees sequentially
- Each tree fixes errors of previous trees
- Fast, accurate, industry-standard

### 4. **Explainable AI (EBM)**
- Shows feature importance (which features matter most?)
- Shows feature shapes (how does temperature affect energy?)
- Builds trust with stakeholders

### 5. **Federated Learning**
- Traditional ML: Send all data to central server → train model
- Federated ML: Train model locally → send only model updates
- Privacy: Raw data never leaves the building

---

## 📊 Evaluation Metrics Explained

### **R2 Score (Accuracy)**
- **Range**: 0% to 100%
- **Meaning**: "How much variance does the model explain?"
- **Your Result**: 98% (excellent!)
- **Interpretation**: The model captures 98% of the patterns in energy use

### **RMSLE (Error)**
- **Range**: 0 to ∞ (lower is better)
- **Meaning**: Average prediction error in log-space
- **Your Result**: 0.27 (very good!)
- **Why Log?**: Treats 10→20 kWh error same as 100→200 kWh (relative error)

---

## 🔥 What to Learn Next

### For This Project:
1. **Hyperparameter Tuning**: Try different `n_estimators`, `learning_rate` in LightGBM
2. **Feature Selection**: Which features matter most? (Use EBM global importance)
3. **Cross-Validation**: Test on multiple time windows (not just one split)
4. **More Models**: Try XGBoost, CatBoost, Neural Networks

### For Your Career:
1. **Pandas**: Master data manipulation (groupby, merge, pivot)
2. **Scikit-Learn**: Standard ML library (preprocessing, metrics, pipelines)
3. **LightGBM/XGBoost**: Industry-standard for tabular data
4. **Time-Series**: ARIMA, Prophet, LSTM (for sequential data)
5. **MLOps**: Deploy models to production (Docker, FastAPI, AWS)

---

## 🎤 Presentation Tips

### Opening (30 seconds):
*"Buildings consume 40% of global energy. My project predicts hourly energy use with 98% accuracy using Federated Learning to keep data private."*

### Demo Flow (2 minutes):
1. **Step 1**: "First, I clean the raw data..." (show terminal)
2. **Step 2**: "Then, I engineer features like lag values..." (show terminal)
3. **Step 3**: "Finally, I train LightGBM and EBM models..." (show terminal)
4. **Dashboard**: "Here are the results—98% accuracy!" (show Streamlit)

### Q&A Prep:
- **"Why Federated Learning?"** → Privacy (hospitals, banks can't share data)
- **"Why LightGBM?"** → Fast, accurate, industry-standard for tabular data
- **"What's the business impact?"** → Peak shaving saves 10-20% on energy bills

---

## 📁 File Structure Summary

```
src/
├── step1_data_cleaning.py       ← Load, melt, merge data
├── step2_feature_engineering.py ← Create calendar, weather, lag features
├── step3_train_models.py        ← Train LightGBM, EBM, Federated Learning
└── app.py                       ← Streamlit dashboard

data/
├── raw/                         ← Original BDG2 files (git-ignored)
├── interim/                     ← Cleaned data (after Step 1)
└── processed/                   ← Final features (after Step 2)

models/                          ← Saved models (git-ignored)
```

---

## 🚀 Running the Project

```bash
# Step 1: Clean data
python src/step1_data_cleaning.py

# Step 2: Engineer features
python src/step2_feature_engineering.py

# Step 3: Train models
python src/step3_train_models.py

# Step 4: Launch dashboard
streamlit run src/app.py
```

---

## 🎓 Resources to Learn More

- **Pandas**: [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- **LightGBM**: [Official Docs](https://lightgbm.readthedocs.io/)
- **EBM**: [InterpretML Tutorial](https://interpret.ml/docs/ebm.html)
- **Federated Learning**: [Google AI Blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- **Time-Series**: [Kaggle Time-Series Course](https://www.kaggle.com/learn/time-series)

---

**Good luck with your presentation! You've built something impressive.** 🌟
