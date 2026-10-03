import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

print("--- LOADING CLEANED DATASET ---")
df = pd.read_csv('cleaned_pakwheels_macro.csv')

# Select Features and Target Variable
feature_cols = [
    'Year', 'Millage_KM', 'Engine_CC', 
    'Exchange_Rate_PKR_USD', 'Inflation_YoY', 
    'Oil_Price_USD_Barrel', 'Interest_Rate', 'Low_Tax_Bracket'
]

# Drop rows with missing feature values
df_model = df.dropna(subset=feature_cols + ['Price_PKR']).copy()

X = df_model[feature_cols]
y = df_model['Price_PKR']

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training sample size: {len(X_train)} | Testing sample size: {len(X_test)}")

# 1. TRAIN RANDOM FOREST REGRESSOR
print("\n--- TRAINING RANDOM FOREST ---")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print(f"Random Forest MAE: PKR {rf_mae:,.2f}")
print(f"Random Forest R² Score: {rf_r2:.4f}")

# 2. TRAIN XGBOOST REGRESSOR
print("\n--- TRAINING XGBOOST ---")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_r2 = r2_score(y_test, xgb_preds)

print(f"XGBoost MAE: PKR {xgb_mae:,.2f}")
print(f"XGBoost R² Score: {xgb_r2:.4f}")

# 3. FEATURE IMPORTANCE ANALYSIS
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'XGBoost_Importance': xgb.feature_importances_
}).sort_values(by='XGBoost_Importance', ascending=False)

print("\n--- FEATURE IMPORTANCE RANKING (XGBoost) ---")
print(importance_df.to_string(index=False))