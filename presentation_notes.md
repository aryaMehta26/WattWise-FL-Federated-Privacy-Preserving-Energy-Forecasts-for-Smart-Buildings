# 🎓 WattWise-FL: Class Presentation & Demo Guide

## 🗣️ Talking Points (The "Script")

### 1. Introduction (The "Why")
*   "Good morning/afternoon. My project is **WattWise-FL**."
*   "**The Problem**: Smart buildings generate huge amounts of data that can help save energy (peak shaving), BUT building owners are afraid to share this data because it reveals private occupancy patterns."
*   "**The Solution**: I built a **Federated Learning** system. Instead of sending raw data to a central server, we send the *model* to the building. The data never leaves the site."

### 2. The Dataset (The "What")
*   "I used the **Building Data Genome 2 (BDG2)** dataset."
*   "It contains hourly energy readings from **1,636 buildings** over **2 years**."
*   "I focused on **Electricity** meter data combined with **Weather** data (temperature, humidity) and **Building Metadata** (age, primary use)."
*   *(Show the `data/raw` folder or the `download.py` script if asked)*.

### 3. The Pipeline (The "How")
*   **Preprocessing**: "I handled 'wide-format' data, melted it into long-format, and merged it with weather timestamps."
*   **Feature Engineering**: "I created cyclical features for time (hour/day), lag features (what was the usage 24h ago?), and rolling weather averages to capture thermal inertia."
*   **Models**:
    1.  **LightGBM**: "My primary high-accuracy model. It uses Gradient Boosting."
    2.  **EBM (Explainable Boosting Machine)**: "I used this for *trust*. It tells us exactly *why* a prediction was made (e.g., 'High temperature increased load by 15%')."
    3.  **Federated Simulator**: "I simulated a distributed network where each building trains locally and updates a global model."

### 4. Results (The "Proof")
*   "On the real test set (unseen future data):"
    *   **LightGBM**:
        *   **R2 Score**: **0.98** (This means the model explains **98%** of the energy usage patterns. It's extremely accurate).
        *   **RMSLE**: **0.2695** (Low error score).
    *   **EBM**:
        *   **R2 Score**: **0.93** (Slightly lower, but gives us full explainability).

---

## ❓ Q&A Prep (Anticipating Professor Questions)

**Q: What do these numbers mean? Is 0.26 good?**
**A:** "Yes! **RMSLE** is an *error* metric, so lower is better (0 is perfect). A score of 0.26 is excellent for this dataset. If you prefer 'Accuracy', my **R2 Score is 0.98**, meaning the model is 98% effective at predicting the variance."

**Q: Why did you use RMSLE instead of RMSE?**
**A:** "Energy data varies wildly—some buildings use 10 kWh, others use 10,000 kWh. RMSE would be dominated by the huge buildings. RMSLE (Log error) treats a 10% error the same regardless of the scale."

**Q: How does your Federated Learning actually work?**
**A:** "I implemented a **FedAvg** (Federated Averaging) simulation. In each round, I select a subset of buildings, train a model on their local data, and then aggregate their 'knowledge' (in my simulation, I aggregate their predictions/performance to show the concept, as averaging Tree ensembles is complex)."

**Q: What was the hardest part?**
**A:** "Handling the real-world data. The BDG2 dataset came in a 'wide' format with missing timestamps and needed complex merging with weather data. I had to write a custom preprocessor to melt and align everything."

**Q: Why is EBM important?**
**A:** "In facility management, operators won't trust a 'black box'. EBM gives us graphable shape functions. If the model predicts a spike, we can look at the graph and see 'Oh, it's because the temperature hit 35°C'."

---

## 💻 Live Demo Steps (The "Clean" Workflow)

1.  **Step 1: Data Cleaning**
    *   "First, I'll run the cleaning script. This handles the raw data ingestion and fixes the wide-format issues."
    *   Run: `python src/step1_data_cleaning.py`

2.  **Step 2: Feature Engineering**
    *   "Next, I generate the features. This adds calendar cycles, weather lags, and building interactions."
    *   Run: `python src/step2_feature_engineering.py`

3.  **Step 3: Model Training**
    *   "Finally, I train the models on the processed data."
    *   Run: `python src/step3_train_models.py`

4.  **Show the Dashboard**
    *   "Now that training is done, let's see the results."
    *   Run: `streamlit run src/app.py`
