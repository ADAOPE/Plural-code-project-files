import streamlit as st
import pandas as pd
import joblib
import numpy as np

#Loading the model into a variable
model = joblib.load("real_estate")

#Loading the Encoder into a variable
encoder = joblib.load("Binary Encoder")

# Defining function to predict prices
def predict_price(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit app layout
st.title("REAL ESTATE SALE PRICE PREDICTION APP")



st.header(" Property Characteristics:")

# Input fields for the categorical variables
property_type = st.selectbox("Property",["Single Family","Two Family","Three Family","Four Family"])
residential_type = st.selectbox("Residential",["Detached House","Duplex","Triplex","Fourplex"])
locality = st.selectbox("Locality", ["Bridgeport", "Waterbury", "Fairfield", "West Hartford", "Greenwich", "Stamford", "Norwalk"])

# Numeric input fields for number of rooms and bathrooms
num_rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3, step=1)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2, step=1)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Property': [property_type],
    'Residential': [residential_type],
    'Locality': [locality],
    'Number of Rooms': [num_rooms],
    'Number of Bathrooms': [num_bathrooms]

})

# Encode the input data using the loaded encoder for categorical variables only
categorical_data = input_data[['Property','Residential',"Locality"]]
encoded_input = encoder.transform(categorical_data)

# Combine the encoded categorical data with the numeric inputs
numeric_data = input_data[['Number of Rooms', 'Number of Bathrooms']]
final_input = pd.concat([encoded_input, numeric_data], axis=1)

# Predict the price when the user clicks the button
if st.button("Predict Price"):
    input_features = final_input.iloc[0].values  # Ensure it matches the expected format
    price = predict_price(input_features)
    st.write(f"The predicted price for the property is: ${price:,.2f}")
