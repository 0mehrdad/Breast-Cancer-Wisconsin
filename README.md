# Breast Cancer Wisconsin – Machine Learning & Visualisation

This project explores the **Breast Cancer Wisconsin (Diagnostic)** dataset using data analysis, feature exploration, and XGBoost-based classification models. It also includes an interactive **Streamlit dashboard** for visualisation and model evaluation.

---

## 🚀 Project Features

### 🔎 Exploratory Data Analysis
- Class imbalance inspection  
- Histograms, KDE plots, and boxplots  
- Correlation heatmaps  
- Scatter-plot matrices

### 🧭 Dimensionality Reduction
- PCA with the first 10 features  
- PCA with all 30 features  

### 🤖 Machine Learning (XGBoost)
- Baseline models:
  - First 10 features (mean)
  - All 30 features
  - PCA components
- Handling imbalance:
  - scale_pos_weight
  - Oversampling (resampling)
- Feature selection using XGBoost feature importance

### 🧪 Model Evaluation
- Confusion matrices  
- Classification reports  
- 10-fold cross-validation  
- Comparison of imbalanced vs balanced results  

### 📊 Streamlit Dashboard
- Feature exploration  
- XGBoost model training & evaluation  
- Optional prediction for new patient data  

---

## 📁 Repository Structure

Project directory layout:

    ├── WDBC.csv                  # Cleaned dataset used in the dashboard
    ├── wdbc.data                 # Original UCI data file
    ├── wdbc.names                # Original UCI metadata/description
    ├── dashboard.py              # Streamlit dashboard (EDA + modelling + prediction)
    ├── data visualisation.ipynb # Jupyter notebook with full analysis
    ├── AE2 Mehrdad-Madadi.pdf   # Academic report with full methodology
    ├── LICENSE                   # GPL-3.0 license
    └── README.md                 # (This file)

---

## 🧬 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset (569 samples, 30 numerical features).

Target labels:
- M — Malignant  
- B — Benign  

All features represent cell-nucleus characteristics extracted from digitized microscopic images.

---

## ▶️ Running the Dashboard

Install dependencies:

    pip install -r requirements.txt

Launch the Streamlit app:

    streamlit run dashboard.py

---

## 🧠 Key Findings

- Oversampling reduces **false negatives**, which is critical in medical classification.  
- XGBoost reaches about **98% accuracy** with very high recall for malignant cases.  
- PCA is helpful for visualisation but slightly reduces performance.  
- A compact model using only important “_worst” features still performs well.
