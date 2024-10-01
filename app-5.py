
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle


# Load your model file
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Real Estate Price Prediction')

# Input fields for the model's features
locality = st.selectbox('Locality', locality_options)
residential = st.selectbox('Residential Type', residential_options)
num_rooms = st.number_input('Number of Rooms', min_value=1, max_value=10, step=1)
num_bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, step=1)

# Convert categorical features to numerical encoding as per your trained model
# Ensure these mappings match how you encoded your training data
locality_mapping = {val: idx for idx, val in enumerate(locality_options)}
residential_mapping = {val: idx for idx, val in enumerate(residential_options)}

# Prepare the input data for prediction
input_data = [[
    locality_mapping[locality],
    residential_mapping[residential],
    num_rooms,
    num_bathrooms
]]

# Button for prediction
if st.button('Predict Sale Price'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Sale Price: ${prediction[0]:,.2f}')

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
real_estate = pd.read_csv(file_path)

# Applying forward fill permanently to 'Locality'
real_estate['Locality'] = real_estate['Locality'].ffill()


# Get unique values for 'Locality' and 'Residential'
locality_options = real_estate['Locality'].unique().tolist()
residential_options = real_estate['Residential'].unique().tolist()

print("Locality Options:", locality_options)
print("Residential Options:", residential_options)


