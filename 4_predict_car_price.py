import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

print("--- TRAINING MODEL FOR INTERACTIVE PREDICTOR ---")

# 1. Load Cleaned Dataset
df = pd.read_csv('cleaned_pakwheels_macro.csv')

feature_cols = [
    'Year', 'Millage_KM', 'Engine_CC', 
    'Exchange_Rate_PKR_USD', 'Inflation_YoY', 
    'Oil_Price_USD_Barrel', 'Interest_Rate', 'Low_Tax_Bracket'
]

df_model = df.dropna(subset=feature_cols + ['Price_PKR']).copy()
X = df_model[feature_cols]
y = df_model['Price_PKR']

# Train XGBoost Model on Full Dataset for Best Accuracy
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X, y)

print("Model trained and ready!\n")

# Get latest macroeconomic values from dataset defaults (e.g. latest year available)
latest_macro = df.sort_values(by='Year', ascending=False).iloc[0]
default_usd_rate = float(latest_macro['Exchange_Rate_PKR_USD'])
default_inflation = float(latest_macro['Inflation_YoY'])
default_oil_price = float(latest_macro['Oil_Price_USD_Barrel'])
default_interest = float(latest_macro['Interest_Rate'])

print("==================================================")
print("       PAKISTAN CAR PRICE PREDICTOR TOOL          ")
print("==================================================")

try:
    year = int(input("Enter Model Year (e.g. 2021): "))
    engine_cc = float(input("Enter Engine Capacity in CC (e.g. 1300, 1500, 660): "))
    mileage = float(input("Enter Mileage in KM (e.g. 45000): "))
    
    # Tax bracket rule
    low_tax = 1 if engine_cc <= 1000 else 0

    # Build input dataframe
    input_data = pd.DataFrame([{
        'Year': year,
        'Millage_KM': mileage,
        'Engine_CC': engine_cc,
        'Exchange_Rate_PKR_USD': default_usd_rate,
        'Inflation_YoY': default_inflation,
        'Oil_Price_USD_Barrel': default_oil_price,
        'Interest_Rate': default_interest,
        'Low_Tax_Bracket': low_tax
    }])

    # Predict
    predicted_price_pkr = model.predict(input_data)[0]
    predicted_price_lacs = predicted_price_pkr / 100_000
    predicted_price_crores = predicted_price_pkr / 10_000_000

    print("\n--------------------------------------------------")
    print("               PREDICTION RESULT                  ")
    print("--------------------------------------------------")
    print(f"Vehicle: {year} | {engine_cc:.0f} CC | {mileage:,.0f} KM")
    print(f"Estimated Price: PKR {predicted_price_pkr:,.2f}")
    
    if predicted_price_pkr >= 10_000_000:
        print(f"In Lacs / Crores: {predicted_price_crores:.2f} Crore PKR")
    else:
        print(f"In Lacs: {predicted_price_lacs:.2f} Lac PKR")
    print("--------------------------------------------------\n")

except Exception as e:
    print(f"\nError: Invalid input. Details: {e}")