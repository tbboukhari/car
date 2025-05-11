import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the trained pipeline
pipeline = joblib.load("model_pipeline.pkl")

# Extract categories for dynamic dropdowns
ohe = pipeline.named_steps['preproc'].named_transformers_['cat']
cat_cols = ['gearbox','energy','make','model']
categories = dict(zip(cat_cols, ohe.categories_))

# Display logo (place 'logo.png' in the same folder)
logo = Image.open("logo.png")
st.image(logo, width=200)

# App title
st.title("Vehicle Price Prediction")

# User inputs: real-world values
mileage    = st.number_input("Mileage (km)",               0, 1_000_000, 50_000, step=1_000)
year       = st.number_input("Year of Registration",      1980, 2025,   2015,   step=1)
hp         = st.number_input("Horsepower (hp)",           10,   1_000,   100,   step=10)
avg_price  = st.number_input("Average Market Price (€)", 0,   200_000, 20_000, step=500)
max_price  = st.number_input("Maximum Market Price (€)",  0,   300_000, 25_000, step=500)
min_price  = st.number_input("Minimum Market Price (€)",  0,   150_000, 15_000, step=500)

gearbox = st.selectbox("Gearbox Type", categories['gearbox'])
energy  = st.selectbox("Fuel Type",    categories['energy'])
make    = st.selectbox("Make",         categories['make'])
model   = st.selectbox("Model",        categories['model'])

# Prediction button
if st.button("Predict Price"):
    # Construct input dataframe
    data = {
        'gearbox':   [gearbox],
        'energy':    [energy],
        'make':      [make],
        'model':     [model],
        'mileage':   [mileage],
        'age':       [2025 - year],  # convert to age
        'hp':        [hp],
        'AVG_price': [avg_price],
        'MIN_price': [min_price],
        'MAX_price': [max_price],
    }
    input_df = pd.DataFrame(data)

    # Predict
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Estimated Price: €{prediction:,.0f}")
