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
We aim to build a predictive model that can accurately predict the presence of gold in the rock samples based on the results of past historic  Multi-elemet assay data. 


Please enter the required input values separated by commas and press the 'Predict' button to see the results.
""")

# Instructions
st.subheader('Instructions')
st.write("""
- Enter your input values in the following format: value1, value2, value3, ..., valueN.
- Press the 'Predict' button to submit your data and see the prediction.
- The model will indicate whether gold is likely present or not based on the input data.
""")

# Streamlit app layout
st.title('Multi-Element Gold Prediction App')


# Mapping of elements to their full names
elements_full_names = {
    'Al': 'Aluminum',
    'As': 'Arsenic',
    'B': 'Boron',
    'Ba': 'Barium',
    'Be': 'Beryllium',
    'Bi': 'Bismuth',
    'Ca': 'Calcium',
    'Cd': 'Cadmium',
    'Co': 'Cobalt',
    'Cr': 'Chromium',
    'Cu': 'Copper',
    'Fe': 'Iron',
    'Ga': 'Gallium',
    'Hg': 'Mercury',
    'K': 'Potassium',
    'La': 'Lanthanum',
    'Mg': 'Magnesium',
    'Mn': 'Manganese',
    'Mo': 'Molybdenum',
    'Na': 'Sodium',
    'Ni': 'Nickel',
    'P': 'Phosphorus',
    'Pb': 'Lead',
    'S': 'Sulfur',
    'Sb': 'Antimony',
    'Sc': 'Scandium',
    'Sr': 'Strontium',
    'Th': 'Thorium',
    'Ti': 'Titanium',
    'Tl': 'Thallium',
    'V': 'Vanadium',
    'W': 'Tungsten',
    'Zn': 'Zinc',
    'Zr': 'Zirconium'
}

# Elements measured in percentage
percentage_elements = ['Fe', 'Ca', 'Al', 'S', 'Ti']

# Creating 7 columns for input
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

# Function to assign each element to its column with full name
def element_input(col, symbol, full_name):
    unit = "pc" if symbol in percentage_elements else "ppm"
    return col.text_input(f"{symbol} ({full_name}) in {unit}", key=f"{symbol}_key")

# Distribute elements across columns
for index, (symbol, full_name) in enumerate(elements_full_names.items()):
    if index % 7 == 0:
        element_input(col1, symbol, full_name)
    elif index % 7 == 1:
        element_input(col2, symbol, full_name)
    elif index % 7 == 2:
        element_input(col3, symbol, full_name)
    elif index % 7 == 3:
        element_input(col4, symbol, full_name)
    elif index % 7 == 4:
        element_input(col5, symbol, full_name)
    elif index % 7 == 5:
        element_input(col6, symbol, full_name)
    elif index % 7 == 6:
        element_input(col7, symbol, full_name)



#  taking input values through Streamlit's input methods
# This is a placeholder for where you'd collect or upload new data
input_values = st.text_input('Enter values separated by commas:')

if st.button('Predict'):
    # Process the input values
    input_list = list(map(float, input_values.split(',')))
    new_data = np.array(input_list).reshape(1, -1)
    
    # Scale, PCA transform, and predict
    new_data_scaled = scaler.transform(new_data)
    new_data_pca = pca.transform(new_data_scaled)
    prediction = model.predict(new_data_pca)
    
    # Display the prediction
    if prediction == 0:
        st.write("No gold present")
    else:
        st.write("Gold present")

#"C:\Users\USER\OneDrive - Thor Exploration\Desktop\Streamlit_Gold Prediction\scaler.joblib"

#"C:\Users\USER\OneDrive - Thor Exploration\Desktop\Streamlit_Multiple Prediction System\Saved models\parkinsons_model.sav"
