# WattWise-FL: Federated, Privacy-Preserving Energy Forecasts for Smart Buildings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Project Overview

WattWise-FL builds a classical ML pipeline to forecast hourly whole-building energy use while preserving privacy. Using the open BDG2 dataset (2016–2017), we predict `meter_reading` for electricity, chilled water, hot water, and steam using Ridge, LightGBM, and Explainable Boosting Machine (EBM) models trained under per-site, centralized, and federated learning regimes.

### Key Features
- ⚡ **Classical ML Focus**: Ridge, LightGBM, EBM (no deep learning)
- 🔐 **Privacy-Preserving**: Federated learning simulation (FedAvg)
- 📊 **Rigorous Evaluation**: Time-series CV with unseen-site testing
- 🔍 **Explainable AI**: EBM shape plots and feature importance
- 🌱 **Sustainability Impact**: Enable peak shaving and emissions reduction

## 📁 Project Structure

```
ML PROJECT/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── environment.yml                    ← Conda environment (alternative)
├── .gitignore                         ← Git ignore rules
├── config.yaml                        ← Project configuration
│
├── data/                              ← Data directory (git-ignored)
│   ├── raw/                           ← Original BDG2 data
│   ├── processed/                     ← Cleaned & processed data
│   ├── features/                      ← Feature matrices
│   └── splits/                        ← Train/test split indices
│
├── notebooks/                         ← Jupyter notebooks
│   ├── 01_data_download.ipynb         ← Download BDG2 dataset
│   ├── 02_eda_metadata.ipynb          ← Explore building metadata
│   ├── 03_eda_weather.ipynb           ← Explore weather data
│   ├── 04_eda_meters.ipynb            ← Explore meter readings
│   ├── 05_preprocessing.ipynb         ← Data cleaning & QA
│   ├── 06_feature_engineering.ipynb   ← Create features
│   ├── 07_baseline_models.ipynb       ← Naive baselines & Ridge
│   ├── 08_lightgbm_models.ipynb       ← LightGBM training
│   ├── 09_ebm_models.ipynb            ← EBM training
│   ├── 10_federated_learning.ipynb    ← FL simulation
│   └── 11_results_analysis.ipynb      ← Final results & plots
│
├── src/                               ← Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py                ← Download BDG2 data
│   │   ├── preprocessing.py           ← Data cleaning functions
│   │   └── validation.py              ← Data quality checks
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── calendar_features.py       ← Time-based features
│   │   ├── weather_features.py        ← Weather features
│   │   ├── building_features.py       ← Building metadata features
│   │   └── lag_features.py            ← Lag & rolling features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py                ← Naive baselines
│   │   ├── ridge.py                   ← Ridge regression
│   │   ├── lightgbm_model.py          ← LightGBM wrapper
│   │   ├── ebm_model.py               ← EBM wrapper
│   │   └── federated.py               ← Federated learning
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 ← RMSLE, MAE, etc.
│   │   ├── cv_splitter.py             ← Time-series CV
│   │   └── diagnostics.py             ← Error analysis
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py                   ← Standard plots
│   │   └── explainability.py          ← EBM, SHAP, PDP plots
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  ← Config loader
│       ├── logging_utils.py           ← Logging setup
│       └── io.py                      ← File I/O utilities
│
├── models/                            ← Saved models (git-ignored)
│   ├── baseline/
│   ├── centralized/
│   ├── per_site/
│   └── federated/
│
├── results/                           ← Results & figures
│   ├── metrics/                       ← Metrics tables
│   ├── figures/                       ← Plots & visualizations
│   └── model_cards/                   ← Model documentation
│
├── tests/                             ← Unit tests
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
└── docs/                              ← Documentation
    ├── data_dictionary.md
    ├── processing.md
    └── model_cards_template.md
```

## 🚀 Quick Start (The "Clean" Workflow)

We have organized the project into **3 clear steps** for reproducibility and clarity.

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline (3 Steps)

**Step 1: Data Cleaning**
Handles raw data ingestion, fixes wide-format issues, and merges metadata.
```bash
python src/step1_data_cleaning.py
```

**Step 2: Feature Engineering**
Generates calendar cycles, weather lags, and building interactions.
```bash
python src/step2_feature_engineering.py
```

**Step 3: Model Training**
Trains LightGBM, EBM, and runs the Federated Learning simulation.
```bash
python src/step3_train_models.py
```

### 3. Launch Dashboard
Visualize the results and explore the data interactively.
```bash
streamlit run src/app.py
```

## 🏆 Results

Our models achieved state-of-the-art performance on the test set:

| Model | Accuracy (R2 Score) | Error (RMSLE) | Description |
|-------|-------------------|---------------|-------------|
| **LightGBM** | **97.9%** | **0.2695** | High-precision Gradient Boosting |
| **EBM** | **92.9%** | **0.4173** | Fully Explainable Additive Model |

*Note: Federated Learning simulation also completed successfully across 18 clients.*

## 📊 Dataset Information

**Building Data Genome 2 (BDG2)**
- **Source**: [BDG2 GitHub Repository](https://github.com/buds-lab/building-data-genome-project-2)
- **Buildings**: 1,636 non-residential buildings
- **Meters**: 3,053 energy meters
- **Time Period**: 2016-2017 (2 full years)
- **Frequency**: Hourly readings
- **Locations**: 19 sites across North America & Europe

**Citation**:
```
Miller, C., Kathirgamanathan, A., Picchetti, B. et al. 
The Building Data Genome Project 2, energy meter data from the ASHRAE 
Great Energy Predictor III competition. 
Sci Data 7, 368 (2020). 
https://doi.org/10.1038/s41597-020-00712-x
```

## 🎯 Project Goals

1. **Forecasting**: Predict hourly `meter_reading` (kWh) for building energy consumption
2. **Privacy**: Implement federated learning to train models without sharing raw data
3. **Explainability**: Use EBM and interpretable methods for operator trust
4. **Validation**: Rigorous time-series CV + unseen-site testing
5. **Impact**: Enable peak shaving, load shifting, and emissions reduction

## 🧪 Models & Approaches

### Baseline Models
- Naive: Same hour last week
- Rolling mean: 24-hour average
- Linear Regression
- Ridge Regression (with regularization)

### Main Models
- **LightGBM**: Gradient boosting decision trees
- **EBM**: Explainable Boosting Machine (interpretable additive model)

### Training Regimes
1. **Per-Site**: Train separate model for each building
2. **Centralized**: Pool all data, train one global model
3. **Federated**: FedAvg with local training, aggregate model updates

## 📏 Evaluation Metrics

- **Primary**: RMSLE (Root Mean Squared Logarithmic Error)
- **Secondary**: MAE (Mean Absolute Error)
- **Diagnostics**: Error by season, temperature band, meter type, building type


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

The BDG2 dataset is available under the CC BY 4.0 license.

## 🙏 Acknowledgments

- ASHRAE for organizing GEPIII competition
- BDG2 team for creating and maintaining the dataset
- Building owners who contributed their data for research

## 📧 Contact

For questions or collaboration opportunities, please contact the team members.


