import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

file_path = os.path.dirname(os.path.abspath(__file__))
csv_path = file_path + '\WDBC.csv'
df = pd.read_csv(csv_path)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Evaluation"])

# ========== 1. DATA OVERVIEW ==========
if page == "Data Overview":
    st.title("Breast Cancer Dataset Overview")
    
    st.subheader("Select a Feature to Explore Distribution")
    selected_feature = st.selectbox("Choose a feature", df.columns[1:-1])
    cl1 , cl2 = st.columns(2)
    with cl1:
            st.subheader("Histogram")
            fig , ax = plt.subplots()
            sns.histplot(df[selected_feature], kde=True)
            st.pyplot(fig)
    with cl2:
            st.subheader("Box plot")
            fig2 ,ax2 = plt.subplots()
            palette = {'B': 'tab:cyan', 'M': 'tab:blue'}
            sns.boxplot(x=df['diagnosis'], y=df[selected_feature], hue=df['diagnosis'],palette = palette)
            st.pyplot(fig2)

    st.subheader("Dataset Info")
    st.write(df.describe())

# ========== 2. MODEL EVALUATION ==========
elif page == "Model Evaluation":
    st.title("XGBoost Model Evaluation")

    use_balanced = st.radio("Use balanced data frame",
         ['Use the original data without balancing',
          'Use balanced dataset','Apply Class Weight (scale_pos_weight)' ])

    features = st.selectbox('Select an option',('First 10 features (mean)','Last 10 features (worst)','All 30 features'))

    
    if use_balanced == 'Use balanced dataset':
        df_majority = df[df.diagnosis == 'B']
        df_minority = df[df.diagnosis == 'M']
        df_minority_upsampled = resample(df_minority, 
                                        replace=True,     
                                        n_samples=len(df_majority),    
                                        random_state=42) 
        df1 = pd.concat([df_majority, df_minority_upsampled])
    else :
        df1 = df
    
    if features == 'First 10 features (mean)':
        X = df1.iloc[:, 2:12].values
    elif features == 'Last 10 features (worst)':
        X = df1.iloc[:, -11:-1].values
    elif features == 'All 30 features':
        X = df1.iloc[:, 2:-1].values
    y = df1['Y'].values
    
    
    
    
    if use_balanced == 'Apply Class Weight (scale_pos_weight)':
        weight = (y == 0).sum() / (y == 1).sum()
    else:
        weight = 1.0

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=weight)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_test, preds, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Predict New Data")
    st.write("Enter the features of the new data point:")
    new_data = {}
    for feature in df.columns[1:11]:
        new_data[feature] = st.number_input(feature, value=0.0, step=0.1)
    x_new = np.array(list(new_data.values())).reshape(1, -1)
    if st.button("Predict"):
        prediction = model.predict(x_new)
        st.write(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")
        proba = model.predict_proba(x_new)
        st.write(f"Probability: {proba[0][1]:.2f} Malignant, {proba[0][0]:.2f} Benign")