import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt # Still needed for SHAP's internal workings, even if not directly plotting

# Set Matplotlib backend to Agg for Streamlit compatibility
plt.switch_backend('Agg')

st.set_page_config(layout="wide", page_title="Diabetic Risk Assessment")

# --- Model Training and Evaluation Functions ---
@st.cache_resource
def train_model(df):
    # Prepare data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split data before SMOTE to prevent data leakage into test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_resampled, y_train_resampled)

    # XGBoost Model
    scale_pos_weight_val = (y_train.value_counts()[0] / y_train.value_counts()[1])
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                              scale_pos_weight=scale_pos_weight_val)
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Store necessary components for SHAP explanation and feature names
    return rf_model, xgb_model, X.columns

# --- SHAP Explanation Function ---
def get_shap_explanation(model, data_point, feature_names):
    explainer = shap.TreeExplainer(model)
    # SHAP values for prediction (output value)
    shap_values = explainer.shap_values(data_point)

    # Ensure shap_values is a list for consistency (TreeExplainer outputs list for classifiers)
    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    # For binary classification, we typically look at SHAP values for the positive class (class 1).
    shap_values_for_class1 = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # Ensure shap_values_for_class1 is 1D for single prediction
    if shap_values_for_class1.ndim > 1:
        shap_values_for_class1 = shap_values_for_class1[0] # Take the first (and only) prediction

    explanation_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values_for_class1
    }).sort_values(by='SHAP Value', ascending=False)

    return explanation_df

# --- Streamlit App Layout ---
st.title("🏥 **Smart Diabetes Prediction System Using Hybrid ML**")
st.markdown("---")
st.markdown("This system helps assess the risk of diabetes based on patient health indicators. It uses advanced machine learning to provide a clear risk evaluation.")

# Load Data
try:
    df = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    st.error("Error: 'diabetes.csv' not found. Please upload the dataset or ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Train models and get necessary components
rf_model, xgb_model, feature_columns = train_model(df) # Removed X_test_df, y_test_series as they are not used in display

st.sidebar.header("Patient Health Profile")
st.sidebar.markdown("Please enter the patient's details below:")
# Input form for new prediction
with st.sidebar.form("prediction_form"):
    pregnancies = st.number_input("Number of Pregnancies (0 - 20)", min_value=0, max_value=20, value=1, help="Number of times pregnant")
    glucose = st.number_input("Glucose Level (mg/dL) (50 - 500)", min_value=0, max_value=500, value=120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    blood_pressure = st.number_input("Blood Pressure (mmHg) (60 - 200)", min_value=0, max_value=200, value=70, help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm) (10 - 100)", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness")
    insulin = st.number_input("Insulin Level (mu U/ml) (15 - 900)", min_value=0, max_value=900, value=79, help="2-Hour serum insulin")
    bmi = st.number_input("BMI (kg/m²) (15.0 - 90.0)", min_value=0.0, max_value=90.0, value=30.0, format="%.1f", help="Body Mass Index")
    dpf = st.number_input("Diabetes Pedigree Function (0.1 - 2.5)", min_value=0.078, max_value=2.5, value=0.372, format="%.3f", help="A function that accounts for diabetes history in relatives and genetic influence")
    age = st.number_input("Age (21 - 100)", min_value=20, max_value=100, value=30)

    submitted = st.form_submit_button("Assess Risk")

st.markdown("---") # Horizontal line for separation

if submitted:
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                              columns=feature_columns)

    # Get predictions from both models
    proba_rf = rf_model.predict_proba(input_data)[:, 1]
    proba_xgb = xgb_model.predict_proba(input_data)[:, 1]

    # Hybrid prediction
    hybrid_proba = (proba_rf + proba_xgb) / 2
    hybrid_prediction = (hybrid_proba > 0.5).astype(int)[0]

    st.subheader("📊 **Risk Assessment Result:**")
    if hybrid_prediction == 1:
        st.error(f"## **Patient is at HIGH RISK of Diabetes**")
        st.write(f"Based on the provided health profile, the model suggests a significant likelihood of diabetes (Probability: **{hybrid_proba[0]:.2f}**).")
        st.markdown("---")
        st.write("### **Key Factors Contributing to High Risk:**")
    else:
        st.success(f"## **Patient is at LOW RISK of Diabetes**")
        st.write(f"Based on the provided health profile, the model suggests a low likelihood of diabetes (Probability: **{hybrid_proba[0]:.2f}**).")
        st.markdown("---")
        st.write("### **Key Factors Influencing Low Risk:**")


    # SHAP Explanations (using XGBoost for explanation as it's often more interpretable)
    shap_explanation_df = get_shap_explanation(xgb_model, input_data, feature_columns)

    # Display the most influential features with clear language
    positive_shap = shap_explanation_df[shap_explanation_df['SHAP Value'] > 0].sort_values(by='SHAP Value', ascending=False)
    negative_shap = shap_explanation_df[shap_explanation_df['SHAP Value'] < 0].sort_values(by='SHAP Value', ascending=True)

    if hybrid_prediction == 1: # If predicted high risk
        st.write("The model identified the following factors as **increasing** the patient's risk of diabetes:")
        if not positive_shap.empty:
            for index, row in positive_shap.head(3).iterrows(): # Show top 3 contributors
                st.markdown(f"- **{row['Feature']}** appears to be a significant contributing factor.")
        else:
            st.write("- The overall combination of health indicators, rather than one dominant factor, suggests a high risk.")

        if not negative_shap.empty:
            st.write("Conversely, some factors that might slightly **reduce** the risk include:")
            for index, row in negative_shap.head(2).iterrows():
                st.markdown(f"- **{row['Feature']}**.")

    else: # If predicted low risk
        st.write("The model identified the following factors as **decreasing** the patient's risk of diabetes:")
        if not negative_shap.empty:
            for index, row in negative_shap.head(3).iterrows(): # Show top 3 contributors
                st.markdown(f"- **{row['Feature']}** appears to be a significant contributing factor to low risk.")
        else:
            st.write("- The overall combination of health indicators suggests a low risk.")

        if not positive_shap.empty:
            st.write("However, some factors that might slightly **increase** the risk are:")
            for index, row in positive_shap.head(2).iterrows():
                st.markdown(f"- **{row['Feature']}**.")


st.markdown("---")
st.info("""
**Important Medical Disclaimer:** This system provides a **risk assessment** based on a predictive model and **is not a diagnostic tool**. It cannot replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment.
""")