Vehicle Price Prediction App

A simple Streamlit web application that allows users to estimate the price of a used vehicle based on its characteristics such as mileage, age, horsepower, market price range, make, model, energy type, and gearbox type.

📦 Features

Real-time vehicle price prediction using a trained XGBoost model

Scaled numerical features and encoded categorical features via scikit-learn pipelines

Dynamic dropdowns based on training data categories (make, model, etc.)

Image/logo support for branding

🧠 Tech Stack

Python 3.10+

Streamlit

Pandas / Numpy

Scikit-learn

XGBoost

Joblib

Pyarrow

🚀 Setup

1. Clone the repo

git clone https://github.com/your-username/vehicle-price-predictor.git
cd vehicle-price-predictor

2. Install dependencies

pip install -r requirements.txt

3. Train the model

python train_model.py

4. Launch the app

streamlit run mlrediction_app.py

📁 Files

xg.py : preprocessing, training and saving the pipeline

ml_app.py : Streamlit interface for prediction

model_pipeline.pkl : serialized model pipeline

requirements.txt : dependencies

logo.png : optional logo to display in the UI
