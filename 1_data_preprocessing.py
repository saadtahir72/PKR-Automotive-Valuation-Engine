import pandas as pd
import numpy as np

print("--- REBUILDING CLEANED DATASET WITH MAKE & MODEL ---")

# 1. LOAD DATASETS
cars_df = pd.read_csv('PakWheels Dataset.csv')
macro_df = pd.read_csv('pakistan_inflation_data.csv')

# 2. CLEAN PAKWHEELS DATASET
def parse_pkr_price(price_str):
    if pd.isna(price_str):
        return np.nan
    text = str(price_str).lower().replace('pkr', '').strip()
    try:
        if 'crore' in text:
            num = float(text.replace('crore', '').strip())
            return num * 10_000_000
        elif 'lac' in text or 'lacs' in text:
            num = float(text.replace('lacs', '').replace('lac', '').strip())
            return num * 100_000
        else:
            return float(text.replace(',', ''))
    except:
        return np.nan

cars_df['Price_PKR'] = cars_df['Price'].apply(parse_pkr_price)

cars_df['Millage_KM'] = (
    cars_df['Millage']
    .astype(str)
    .str.replace('km', '', case=False)
    .str.replace(',', '')
    .str.strip()
)
cars_df['Millage_KM'] = pd.to_numeric(cars_df['Millage_KM'], errors='coerce')

cars_df['Engine_CC'] = (
    cars_df['Engine Capacity']
    .astype(str)
    .str.replace('cc', '', case=False)
    .str.replace(',', '')
    .str.strip()
)
cars_df['Engine_CC'] = pd.to_numeric(cars_df['Engine_CC'], errors='coerce')

# Extract Make (First word) and Model (Second word) from car title column 'nam'
cars_df['Make'] = cars_df['nam'].astype(str).str.split().str[0]
cars_df['Model'] = cars_df['nam'].astype(str).str.split().str[1]

# Filter valid records
cars_clean = cars_df[
    (cars_df['Price_PKR'] > 50_000) & 
    (cars_df['Year'] >= 1980) & 
    (cars_df['Engine_CC'] > 0) & 
    (cars_df['Millage_KM'] >= 0) &
    (cars_df['Make'].notna()) &
    (cars_df['Model'].notna())
].copy()

# 3. PROCESS MACROECONOMIC DATASET
macro_df['Year'] = pd.to_datetime(macro_df['Date'], format='%b-%Y').dt.year

macro_annual = macro_df.groupby('Year').agg({
    'Exchange_Rate_PKR_USD': 'mean',
    'Inflation_YoY': 'mean',
    'Oil_Price_USD_Barrel': 'mean',
    'Interest_Rate': 'mean'
}).reset_index()

# 4. MERGE DATASETS
df_final = pd.merge(cars_clean, macro_annual, on='Year', how='inner')

df_final['Price_USD'] = df_final['Price_PKR'] / df_final['Exchange_Rate_PKR_USD']
df_final['Low_Tax_Bracket'] = (df_final['Engine_CC'] <= 1000).astype(int)

# Overwrite CSV with new columns
df_final.to_csv('cleaned_pakwheels_macro.csv', index=False)

print("\n--- DATASET SUCCESSFULLY REBUILT ---")
print(f"Columns saved: {list(df_final.columns)}")