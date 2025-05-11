# train_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1️⃣ Chargement des données
df = pd.read_parquet('DS-USED-FOR-V.03-MODEL.parquet', engine='pyarrow')

# 2️⃣ Nettoyage basique & feature engineering
df = df.dropna(subset=['price','mileage','year','hp','AVG_price','MIN_price','MAX_price'])
df['age'] = 2025 - df['year']

# 3️⃣ Définition des colonnes
cat_cols = ['gearbox','energy','make','model']
num_cols = ['mileage','age','hp','AVG_price','MIN_price','MAX_price']

# 4️⃣ Séparation X / y
X = df[cat_cols + num_cols]
y = df['price']

# 5️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=55
)

# 6️⃣ Préprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), num_cols),
    # Note : à partir de sklearn 1.2, `sparse` → `sparse_output`
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
], remainder='drop')

# 7️⃣ Pipeline complète avec XGBoost
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('xgb', XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.1,
        colsample_bytree=0.8,
        random_state=55,
        tree_method='hist'
    ))
])

# 8️⃣ Entraînement
pipeline.fit(X_train, y_train)

# 9️⃣ Évaluation
preds = pipeline.predict(X_test)
print(f"MAE  : {mean_absolute_error(y_test, preds):.2f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, preds)):.2f}")

# 🔟 Sauvegarde du pipeline
joblib.dump(pipeline, "model_pipeline.pkl")
print("👉 Pipeline saved to model_pipeline.pkl")
