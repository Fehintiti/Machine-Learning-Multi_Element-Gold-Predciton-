# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from joblib import load
import streamlit as st
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))

#scaler_path = r"C:\Users\USER\OneDrive - Thor Exploration\Desktop\Streamlit_Gold Prediction\scaler.joblib"
#pca_path = r"C:\Users\USER\OneDrive - Thor Exploration\Desktop\Streamlit_Gold Prediction\pca.joblib"
#model_path = r"C:\Users\USER\OneDrive - Thor Exploration\Desktop\Streamlit_Gold Prediction\Ramdom_Gold_model.joblib"

scaler_path = os.path.join(working_dir, "scaler.joblib")
pca_path = os.path.join(working_dir, "pca.joblib")
model_path = os.path.join(working_dir, "Ramdom_Gold_model.joblib")

# Load the scaler, PCA, and model using their direct paths
scaler = load(scaler_path)
pca = load(pca_path)
model = load(model_path)


# Project Introduction
st.header('Project Introduction')
st.write("""
We aim to build a predictive model that can accurately predict the presence of gold in the rock samples based on the results of past historic Multi-element assay data.
Please enter the required input values in the designated fields and press the 'Predict' button to see the results.
""")

# Streamlit app layout
st.title('Multi-Element Gold Prediction App')

st.subheader('Instructions')
st.write("""
- Enter your input values in the following format: value1, value2, value3, ..., valueN.
- Press the 'Predict' button to submit your data and see the prediction.
- The model will indicate whether gold is likely present or not based on the input data.
""")


# Mapping of elements to their full names
elements_full_names = {
    'Al': 'Aluminum', 'As': 'Arsenic', 'B': 'Boron', 'Ba': 'Barium', 'Be': 'Beryllium', 'Bi': 'Bismuth', 'Ca': 'Calcium', 
    'Cd': 'Cadmium', 'Co': 'Cobalt', 'Cr': 'Chromium', 'Cu': 'Copper', 'Fe': 'Iron', 'Ga': 'Gallium', 'Hg': 'Mercury', 
    'K': 'Potassium', 'La': 'Lanthanum', 'Mg': 'Magnesium', 'Mn': 'Manganese', 'Mo': 'Molybdenum', 'Na': 'Sodium', 
    'Ni': 'Nickel', 'P': 'Phosphorus', 'Pb': 'Lead', 'S': 'Sulfur', 'Sb': 'Antimony', 'Sc': 'Scandium', 'Sr': 'Strontium', 
    'Th': 'Thorium', 'Ti': 'Titanium', 'Tl': 'Thallium', 'V': 'Vanadium', 'W': 'Tungsten', 'Zn': 'Zinc', 'Zr': 'Zirconium','Ag': 'Silver'
}
percentage_elements = ['Fe', 'Ca', 'Al', 'S', 'Ti']

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
# Creating columns for input
# Creating columns for input
inputs = {}
for index, (symbol, full_name) in enumerate(elements_full_names.items()):
    unit = "pc" if symbol in percentage_elements else "ppm"
    if index % 7 == 0:
        col = col1
    elif index % 7 == 1:
        col = col2
    elif index % 7 == 2:
        col = col3
    elif index % 7 == 3:
        col = col4
    elif index % 7 == 4:
        col = col5
    elif index % 7 == 5:
        col = col6
    elif index % 7 == 6:
        col = col7
    inputs[symbol] = col.text_input(f"{symbol} ({full_name}) in {unit}", key=f"{symbol}_key")

if st.button('Predict'):
    input_list = []
    error_detected = False
    for symbol in elements_full_names.keys():
        input_value = inputs[symbol].strip()
        try:
            # Convert to float, treating blanks as zeros
            input_list.append(float(input_value) if input_value else 0.0)
        except ValueError:
            st.error(f"Invalid input for {symbol}: '{input_value}' is not a numeric value.")
            error_detected = True
            break

    if not error_detected:
        try:
            new_data = np.array(input_list).reshape(1, -1)
            new_data_scaled = scaler.transform(new_data)
            new_data_pca = pca.transform(new_data_scaled)
            prediction = model.predict(new_data_pca)
            if prediction == 0:
                st.write("No gold present")
            else:
                st.write("Gold present")
        except Exception as e:
            st.error(f"An error occurred: {e}")
