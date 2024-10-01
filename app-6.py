
import pickle
import streamlit as st
import pandas as pd

# Load your model file
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title('Real Estate Sale Price Predictor App')

# Add input widgets for user inputs
locality = st.selectbox(
    "Select Locality",
    options=["Waterbury", "Norwalk", "Bridgeport", "Fairfield", "Stamford", "West Hartford", "Greenwich"]
)

property_type = st.selectbox(
    "Select Property Type",
    options=["Single Family", "Four Family", "Three Family", "Two Family"]
)

residential_type = st.selectbox(
    "Select Residential Type",
    options=["Detached House", "Duplex", "Fourplex", "Triplex"]
)

num_rooms = st.slider("Number of Rooms", min_value=1, max_value=10, value=3)
num_bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=5, value=2)

# When the 'Predict' button is clicked
if st.button("Predict"):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'Locality': [locality],
        'Property': [property_type],
        'Residential': [residential_type],
        'num_rooms': [num_rooms],
        'num_bathrooms': [num_bathrooms]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0].round(2)
    st.write(f'The predicted Real Estate House Price: {prediction} thousand dollars')
