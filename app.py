import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

# Set page config
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Load model and data
@st.cache_resource
def load_model():
    try:
        return pk.load(open('model.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('Cardetails.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

model = load_model()
cars_data = load_data()

if model is None or cars_data is None:
    st.stop()

# Helper function for brand name extraction
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

# Prepare data
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# App header
st.title('Car Price Prediction ML Model')
st.write('Enter the details of your car to predict its price')

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    name = st.selectbox('Select Car Brand', sorted(cars_data['name'].unique()))
    year = st.slider('Car Manufactured Year', 1994, 2024, 2020)
    km_driven = st.number_input('Kilometers Driven', min_value=11, max_value=200000, value=50000)
    fuel = st.selectbox('Fuel Type', sorted(cars_data['fuel'].unique()))
    seller_type = st.selectbox('Seller Type', sorted(cars_data['seller_type'].unique()))

with col2:
    transmission = st.selectbox('Transmission Type', sorted(cars_data['transmission'].unique()))
    owner = st.selectbox('Owner Type', sorted(cars_data['owner'].unique()))
    mileage = st.slider('Car Mileage (km/l)', 10, 40, 15)
    engine = st.slider('Engine (CC)', 700, 5000, 1500)
    max_power = st.slider('Max Power (bhp)', 0, 200, 100)
    seats = st.slider('Number of Seats', 4, 10, 5)

# Mapping dictionaries for categorical variables
owner_map = {
    'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
    'Fourth & Above Owner': 4, 'Test Drive Car': 5
}

fuel_map = {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}
seller_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
transmission_map = {'Manual': 1, 'Automatic': 2}
brand_map = {brand: idx + 1 for idx, brand in enumerate(sorted(cars_data['name'].unique()))}

if st.button("Predict Price", type="primary"):
    try:
        # Create input dataframe
        input_data = pd.DataFrame([[
            name, year, km_driven, fuel, seller_type,
            transmission, owner, mileage, engine, max_power, seats
        ]], columns=[
            'name', 'year', 'km_driven', 'fuel', 'seller_type',
            'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'
        ])

        # Apply mappings
        input_data['owner'] = input_data['owner'].map(owner_map)
        input_data['fuel'] = input_data['fuel'].map(fuel_map)
        input_data['seller_type'] = input_data['seller_type'].map(seller_map)
        input_data['transmission'] = input_data['transmission'].map(transmission_map)
        input_data['name'] = input_data['name'].map(brand_map)

        # Make prediction
        car_price = model.predict(input_data)[0]

        # Display prediction
        st.success(f"Estimated Car Price: ${car_price:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")