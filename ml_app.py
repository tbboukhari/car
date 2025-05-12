import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 2rem;
        font-weight: bold;
        color: #047857;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ECFDF5;
    }
    .section {
        padding: 1rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load models and data
@st.cache_resource
def load_models():
    pipeline = joblib.load("improved_model_pipeline.pkl")
    make_model_mapping = joblib.load("make_model_mapping.pkl")
    return pipeline, make_model_mapping

try:
    # Load the trained pipeline and make-model mapping
    pipeline, make_model_mapping = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Extract categories for dynamic dropdowns
ohe = pipeline.named_steps['preproc'].named_transformers_['cat']
cat_cols = ['make', 'model', 'gearbox', 'energy']
categories = dict(zip(cat_cols, ohe.categories_))

# Display logo and title
try:
    logo = Image.open("logo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=250)
except:
    pass  # No logo available

st.markdown("<h1 class='main-header'>Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>Get an instant estimate of your car's market value</p>", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Car Details</h2>", unsafe_allow_html=True)
    
    # Make dropdown
    make = st.selectbox("Make", sorted(categories['make']))
    
    # Filter models based on selected make
    available_models = make_model_mapping.get(make, [])
    model = st.selectbox("Model", sorted(available_models))
    
    # Year with min and max limits
    current_year = 2025
    year = st.slider("Year of Registration", 2000, current_year, current_year - 3)
    
    # Calculate age for the model
    age = current_year - year
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Technical Specifications</h2>", unsafe_allow_html=True)
    
    # Other inputs
    mileage = st.number_input("Mileage (km)", 0, 500_000, 30_000, step=1_000)
    hp = st.slider("Horsepower (hp)", 50, 500, 120)
    
    gearbox = st.selectbox("Gearbox Type", categories['gearbox'])
    energy = st.selectbox("Fuel Type", categories['energy'])
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add a divider
st.markdown("---")

# Center the prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Calculate Price Estimate", use_container_width=True)

# Prediction logic
if predict_button:
    try:
        # Construct input dataframe with the selected features
        data = {
            'make': [make],
            'model': [model],
            'gearbox': [gearbox],
            'energy': [energy],
            'mileage': [mileage],
            'age': [age],
            'hp': [hp]
        }
        input_df = pd.DataFrame(data)
        
        # Show a spinner while predicting
        with st.spinner('Calculating price...'):
            # Predict
            prediction = pipeline.predict(input_df)[0]
            
            # Format prediction
            formatted_prediction = f"€{prediction:,.0f}"
            
            # Display result
            st.markdown("<div class='result-text'>", unsafe_allow_html=True)
            st.markdown(f"### Estimated Price: {formatted_prediction}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show price range for better context (estimated ±15%)
            lower_bound = prediction * 0.85
            upper_bound = prediction * 1.15
            
            st.info(f"Price range: €{lower_bound:,.0f} - €{upper_bound:,.0f}")
            
            # Add explanation of factors affecting the price
            st.markdown("### Key Price Factors")
            
            # Create factors explanation
            factors = {
                "Age": f"{age} years old" + (" (newer cars typically cost more)" if age < 5 else ""),
                "Mileage": f"{mileage:,} km" + (" (lower mileage typically increases value)" if mileage < 50000 else ""),
                "Make & Model": f"{make} {model}",
                "Fuel Type": energy,
                "Gearbox": gearbox,
                "Horsepower": f"{hp} hp"
            }
            
            # Display factors as a nice looking table
            factor_df = pd.DataFrame({"Factor": factors.keys(), "Value": factors.values()})
            st.table(factor_df)
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Add some information at the bottom
st.markdown("---")
st.markdown("""
**Note**: This price estimate is based on our advanced machine learning model trained on thousands of vehicle listings. 
Actual market prices may vary based on additional factors such as vehicle condition, optional equipment, color, and local market demand.
""")
