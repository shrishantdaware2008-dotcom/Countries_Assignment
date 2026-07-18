import streamlit as st
import pandas as pd
import joblib

model = joblib.load("LR_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)
year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2025,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=50000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    max_value=1000,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    max_value=100.0,
    value=55.4
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.0,
    max_value=10.0,
    value=1.5
)
transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)
model_name = st.text_input("Car Model")

predict = st.button("Predict Price")
if predict:

    input_data = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(columns=encoded_columns, fill_value=0)

    input_data = pd.DataFrame(
    scaler.transform(input_data),
    columns=input_data.columns
)

    prediction = model.predict(input_data)

    st.success(f"Predicted Car Price: £{prediction[0]:,.2f}")
