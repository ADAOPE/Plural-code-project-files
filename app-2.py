
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Initialize label encoders for categorical features
locality_encoder = LabelEncoder()
property_encoder = LabelEncoder()
residential_encoder = LabelEncoder()

# Fit encoders on the dataset
locality_encoder.fit(localities)        # Corrected variable name
property_encoder.fit(property_types)    # Corrected variable name
residential_encoder.fit(residential_types)  # Corrected variable name


real_estate['Property'] = real_estate['Property'].replace('?', pd.NA)
real_estate['Property'] = real_estate['Property'].fillna(method='ffill')
real_estate['Locality'] = real_estate['Locality'].fillna(method='ffill')

# Prepare options for dropdowns based on the dataset
localities = sorted(real_estate["Locality"].dropna().unique())
property_types = sorted(real_estate["Property"].dropna().unique())
residential_types = sorted(real_estate["Residential"].dropna().unique())

# Title of the app
st.title('Real Estate Sale Price Predictor App')

# Add input widgets for user inputs
locality = st.selectbox("Select Locality", options=localities)
property_type = st.selectbox("Select Property Type", options=property_types)
residential_type = st.selectbox("Select Residential Type", options=residential_types)

# Slider widgets for numerical inputs
num_rooms = st.slider("Number of Rooms", min_value=1, max_value=10, value=3)
num_bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=5, value=2)

# Function to make predictions
def make_prediction(locality, property_type, residential_type, num_rooms, num_bathrooms):
    # Prepare the input data
    data = {
        "Locality": [locality],
        "Property": [property_type],
        "Residential": [residential_type],
        "num_rooms": [num_rooms],
        "num_bathrooms": [num_bathrooms]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)

# Encode categorical features
    df['Locality'] = locality_encoder.transform(df['Locality'])
    df['Property'] = property_encoder.transform(df['Property'])
    df['Residential'] = residential_encoder.transform(df['Residential'])

    # Make the prediction
    prediction = model.predict(df).round(2)[0]
    
    return f"Predicted Sale Price: {prediction} dollars"

# When the 'Predict' button is clicked
if st.button("Predict"):
    # Get the prediction
    prediction_text = make_prediction(locality, property_type, residential_type, num_rooms, num_bathrooms)
    st.write(prediction_text)


