

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
df = pd.read_parquet('DS-USED-FOR-V.03-MODEL.parquet', engine='pyarrow')

# Basic cleaning & feature engineering
df = df.dropna(subset=['price', 'mileage', 'year', 'hp'])
df['age'] = 2025 - df['year']  # Calculate age from year

# Define the most relevant columns for users
cat_cols = ['make', 'model', 'gearbox', 'energy']  # Most relevant categorical features
num_cols = ['mileage', 'age', 'hp']                # Most relevant numerical features

# Split features and target
X = df[cat_cols + num_cols]
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=55
)

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
], remainder='drop')

# Full pipeline with XGBoost
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('xgb', XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=55,
        tree_method='hist',
        eval_metric='rmse'
    ))
])

# Train the model
print("Training model...")
pipeline.fit(X_train, y_train)

# Model evaluation
preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"Model Performance:")
print(f"MAE: €{mae:.2f}")
print(f"RMSE: €{rmse:.2f}")
print(f"R²: {r2:.4f}")

# Create a mapping of make to its available models
make_model_mapping = {}
for make in df['make'].unique():
    make_model_mapping[make] = df[df['make'] == make]['model'].unique().tolist()

# Save the pipeline and the make-model mapping
joblib.dump(pipeline, "improved_model_pipeline.pkl")
joblib.dump(make_model_mapping, "make_model_mapping.pkl")

print("✅ Model and make-model mapping saved successfully!")
