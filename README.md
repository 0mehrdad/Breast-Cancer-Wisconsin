# Breast Cancer Wisconsin – Machine Learning & Visualisation

This project explores the **Breast Cancer Wisconsin (Diagnostic)** dataset using data analysis, feature exploration, and XGBoost-based classification models. It also includes an interactive **Streamlit dashboard** for visualisation and model evaluation.

---

## 🚀 Project Features

### 🔎 Exploratory Data Analysis
- Check class imbalance  
- Histograms, KDE plots, boxplots  
- Correlation heatmaps  
- Scatter-plot matrix (pair plots)

### 🧭 Dimensionality Reduction
- PCA using 10 features (mean values)  
- PCA using all 30 features  

### 🤖 Machine Learning (XGBoost)
- Baseline models with:
  - First 10 features (mean)
  - All 30 features
  - PCA components
- Class imbalance handling:
  - scale_pos_weight
  - Oversampling (resampling the minority class)
- Feature selection using XGBoost feature importance

### 🧪 Model Evaluation
- Confusion matrices  
- Classification reports  
- 10-fold cross-validation  
- Comparison of original vs balanced data  

### 📊 Streamlit Dashboard
- Explore feature distributions  
- Train and evaluate models with different settings  
- Optional prediction for new patient data  

---

## 📁 Repository Structure

- `WDBC.csv` – Cleaned dataset used in the dashboard  
- `wdbc.data` / `wdbc.names` – Original UCI dataset files  
- `dashboard.py` – Streamlit app (EDA + model evaluation + prediction)  
- `data visualisation.ipynb` – Notebook with analysis and modelling  
- `AE2 Mehrdad-Madadi.pdf` – Full project report  
- `LICENSE` – GPL-3.0 license  

---

## 🧬 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset (569 samples, 30 numerical features).

Target classes:
- **M** – Malignant  
- **B** – Benign  

Features describe properties of cell nuclei (radius, texture, area, smoothness, concavity, etc.).

---

## ▶️ How to Run the Dashboard

Install dependencies (example):

    pip install -r requirements.txt

Run the Streamlit app:

    streamlit run dashboard.py

Then open the local URL shown in the terminal.

---

## 🧠 Summary of Results

- Oversampling significantly reduces **false negatives**, which is critical for cancer detection.  
- XGBoost achieves around **98% accuracy** with very high recall for malignant cases.  
- PCA is useful for visualisation but slightly weaker for final performance.  
- A compact model using a subset of important “_worst” features still performs strongly.

---

## 📌 License

This project is released under the **GPL-3.0** license.
