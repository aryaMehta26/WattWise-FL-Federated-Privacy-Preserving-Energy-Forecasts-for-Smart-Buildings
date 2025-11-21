# 🔐 Federated Learning Explained (For Your Project)

## ❓ Your Question: "We're Using Raw Data from GitHub, So How Is This Federated?"

**Great observation!** You're absolutely right to be confused. Let me clarify:

---

## 🎯 What We're Actually Doing

### **Reality Check:**
1. **We ARE downloading raw data** from the BDG2 GitHub repository
2. **We ARE training on centralized data** (all data in one place)
3. **We ARE simulating Federated Learning** (not doing real FL)

### **Why the Confusion?**
This is a **research/academic project**, not a real-world deployment. We're **demonstrating** how Federated Learning *would* work if we had real buildings with private data.

---

## 🏢 Real-World Federated Learning (How It Actually Works)

### **Scenario: 100 Office Buildings Want Better Energy Forecasts**

#### **Traditional Centralized ML (What We DON'T Want):**
```
Building A → Sends all energy data → Central Server
Building B → Sends all energy data → Central Server
Building C → Sends all energy data → Central Server
                                      ↓
                            Central Server trains model
                                      ↓
                            Sends model back to buildings
```

**Problem:** 
- Buildings must share sensitive data (occupancy patterns, business hours)
- Privacy risk (competitors could learn your operations)
- Legal issues (GDPR, data sovereignty)

#### **Federated Learning (What We WANT):**
```
Building A → Trains model locally → Sends only MODEL WEIGHTS → Central Server
Building B → Trains model locally → Sends only MODEL WEIGHTS → Central Server
Building C → Trains model locally → Sends only MODEL WEIGHTS → Central Server
                                                                      ↓
                                                    Central Server averages weights
                                                                      ↓
                                                    Sends improved model back
```

**Benefit:**
- Raw data NEVER leaves the building
- Only model parameters (numbers) are shared
- Privacy preserved!

---

## 🧪 What Our Project Does (Simulation)

Since we don't have access to 100 real buildings with private servers, we **simulate** this process:

### **Our Simulation:**
```python
# In src/models/federated.py

# Step 1: Pretend each "site_id" is a separate building with its own data
sites = df['site_id'].unique()  # 18 different sites

# Step 2: For each site, train a model on ONLY that site's data
for site in sites:
    site_data = df[df['site_id'] == site]
    local_model = train_model(site_data)  # Trained locally
    model_weights.append(local_model.get_weights())  # Extract weights

# Step 3: Average the weights (FedAvg algorithm)
global_weights = average_weights(model_weights)

# Step 4: Send global model back to all sites
global_model.set_weights(global_weights)
```

### **Key Point:**
- We're **simulating** 18 separate buildings
- In reality, they're just different subsets of the same dataset
- But the **algorithm** is the same as real Federated Learning

---

## 📊 The Dataset: Building Data Genome 2 (BDG2)

### **What Is BDG2?**
- **Source:** [GitHub Repository](https://github.com/buds-lab/building-data-genome-project-2)
- **Buildings:** 1,636 non-residential buildings
- **Sites:** 19 locations (universities, offices, hospitals)
- **Time Period:** 2016-2017 (2 years of hourly data)
- **Meters:** Electricity, chilled water, hot water, steam

### **Why This Dataset?**
1. **Real-world data** (not synthetic)
2. **Publicly available** (for research/education)
3. **Diverse buildings** (offices, labs, hospitals, etc.)
4. **Rich metadata** (building age, size, use type)
5. **Weather data** (temperature, humidity, wind)

### **How We Use It:**
```
Raw Data (GitHub) → Download → Clean → Engineer Features → Train Models
```

---

## 🔬 EDA (Exploratory Data Analysis) - What We're Doing

### **What Is EDA?**
EDA = Understanding your data BEFORE building models.

### **What We Check:**

#### **1. Data Quality**
```python
# Missing values?
df.isnull().sum()

# Outliers? (negative energy readings?)
df[df['meter_reading'] < 0]

# Time gaps? (missing hours?)
df['timestamp'].diff().value_counts()
```

#### **2. Distributions**
```python
# How is energy consumption distributed?
df['meter_reading'].hist()

# Are there extreme values?
df['meter_reading'].describe()
```

#### **3. Patterns**
```python
# Daily pattern (energy higher during work hours?)
df.groupby('hour')['meter_reading'].mean().plot()

# Weekly pattern (weekends lower?)
df.groupby('day_of_week')['meter_reading'].mean().plot()

# Seasonal pattern (summer vs winter?)
df.groupby('month')['meter_reading'].mean().plot()
```

#### **4. Correlations**
```python
# Does temperature affect energy use?
df[['air_temperature', 'meter_reading']].corr()

# Which features matter most?
df.corr()['meter_reading'].sort_values()
```

### **Where We Do EDA:**
- **Implicitly** in `src/data/preprocessing.py` (data cleaning, outlier removal)
- **Explicitly** in the feature engineering step (understanding patterns)
- **Visually** in the Streamlit dashboard (showing patterns to users)

---

## ⚙️ Feature Engineering - What We're Doing

### **What Is Feature Engineering?**
Transforming raw data into features that help the model learn patterns.

### **Our Features:**

#### **1. Calendar Features** (`src/features/calendar_features.py`)
```python
# Extract time components
df['hour'] = df['timestamp'].dt.hour          # 0-23
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = df['timestamp'].dt.month        # 1-12
df['is_weekend'] = df['day_of_week'] >= 5     # True/False

# Cyclical encoding (hour 23 and hour 0 are close!)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

**Why?** Energy use has daily/weekly cycles (offices empty at night, weekends).

#### **2. Weather Features** (`src/features/weather_features.py`)
```python
# Use raw weather
df['air_temperature']
df['humidity']

# Rolling averages (smooth out noise)
df['temp_rolling_24h'] = df.groupby('building_id')['air_temperature'].rolling(24).mean()
```

**Why?** Hot days → more AC → higher energy use.

#### **3. Building Features** (`src/features/building_features.py`)
```python
# Building metadata
df['building_age'] = 2016 - df['year_built']
df['log_square_feet'] = np.log1p(df['square_feet'])  # Log transform for scale

# Encode building type (Office, Hospital, etc.)
df['primary_use_encoded'] = df['primary_use'].astype('category').cat.codes
```

**Why?** Older buildings use more energy, hospitals run 24/7, etc.

#### **4. Lag Features** (`src/features/lag_features.py`)
```python
# What was the energy use 1 hour ago?
df['meter_reading_lag_1h'] = df.groupby('building_id')['meter_reading'].shift(1)

# What was it 24 hours ago (same time yesterday)?
df['meter_reading_lag_24h'] = df.groupby('building_id')['meter_reading'].shift(24)

# Rolling average (last 24 hours)
df['meter_reading_rolling_24h'] = df.groupby('building_id')['meter_reading'].rolling(24).mean()
```

**Why?** Energy use is autocorrelated (if it was high yesterday at 9am, it'll be high today at 9am).

---

## 🎓 Summary: Answering Your Questions

### **1. "We're using raw data from GitHub, so how is this Federated?"**
- **Answer:** We're **simulating** Federated Learning. In a real deployment, each building would have its own server and data. We're using a public dataset to **demonstrate** the algorithm.

### **2. "What dataset are we using?"**
- **Answer:** Building Data Genome 2 (BDG2) from GitHub. It has 1,636 buildings, 2 years of hourly energy data, plus weather and metadata.

### **3. "What EDA are we doing?"**
- **Answer:** We check for missing values, outliers, time gaps, and visualize patterns (daily/weekly cycles, temperature correlations). This happens in preprocessing and the dashboard.

### **4. "What is feature engineering?"**
- **Answer:** Creating new features from raw data:
  - **Calendar:** Hour, day, month, cyclical encoding
  - **Weather:** Temperature, humidity, rolling averages
  - **Building:** Age, size, type
  - **Lag:** Past energy values (1h ago, 24h ago)

---

## 🎤 How to Explain This in Your Presentation

### **Federated Learning:**
*"In a real deployment, each building would train a model locally and only share model weights—not raw data. Since we don't have 100 real buildings, we simulate this by treating each site in the BDG2 dataset as a separate building. The algorithm is the same as real Federated Learning."*

### **Dataset:**
*"We use the Building Data Genome 2 dataset from GitHub. It has 1,636 buildings across 19 sites with 2 years of hourly energy data, weather, and building metadata."*

### **EDA:**
*"We explored the data to find patterns: energy use peaks during work hours, drops on weekends, and correlates with temperature. This guided our feature engineering."*

### **Feature Engineering:**
*"We created features like hour-of-day (cyclical), lag values (past energy use), and weather rolling averages. These features help the model learn temporal and seasonal patterns."*

---

## 📚 Further Reading

- **Federated Learning:** [Google AI Blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- **BDG2 Dataset:** [Paper](https://www.nature.com/articles/s41597-020-00712-x)
- **Feature Engineering:** [Kaggle Guide](https://www.kaggle.com/learn/feature-engineering)
- **Time-Series EDA:** [Towards Data Science](https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a)

---

**You now understand the full picture!** 🎯
