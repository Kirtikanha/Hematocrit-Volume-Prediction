# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:35:42 2025

@author: Kirti
"""
# streamlit_app.py
import streamlit as st
import pandas as pd
import json

# Load the saved model

import xgboost as xgb

model = xgb.XGBRegressor()

model.load_model("xgb_model.json")


st.title("Hematocrit Volume Prediction using XGBoost")

st.write("""
This app predicts Hematocrit Volume (HV) based on input features.
You can either enter values manually or upload a CSV file.
""")

# Sidebar for input selection
input_option = st.sidebar.selectbox("Select Input Method", ["Manual Entry", "CSV Upload"])

# Define feature names (same as your training features)
feature_names = ['Gc in mg/dL', 'Tp in sec', 'Ip in microAmp']  # replace with your actual feature column names

if input_option == "Manual Entry":
    # Create input fields
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)
    
    input_df = pd.DataFrame([user_input])
    
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Hematocrit Volume (HV): {prediction:.3f}")

else:  # CSV Upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(input_df)
        
        if st.button("Predict"):
            predictions = model.predict(input_df)
            input_df['Predicted_HV'] = predictions
            st.write("Predictions:")
            st.dataframe(input_df)
            
            # Allow CSV download
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )

