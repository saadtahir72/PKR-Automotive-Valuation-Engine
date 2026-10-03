import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Set plot style
sns.set_theme(style="whitegrid")

print("--- LOADING DATA & RUNNING PLOTS ---")
df = pd.read_csv('cleaned_pakwheels_macro.csv')

feature_cols = [
    'Year', 'Millage_KM', 'Engine_CC', 
    'Exchange_Rate_PKR_USD', 'Inflation_YoY', 
    'Oil_Price_USD_Barrel', 'Interest_Rate', 'Low_Tax_Bracket'
]

df_model = df.dropna(subset=feature_cols + ['Price_PKR']).copy()
X = df_model[feature_cols]
y = df_model['Price_PKR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)

# Figure setup (3 plots side by side / stacked)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Feature Importance Bar Chart
importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values()
importances.plot(kind='barh', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('XGBoost Feature Importance (PakWheels & Macro Data)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Relative Importance Score')

# Plot 2: Actual vs Predicted Prices (in Lakhs/Crores)
axes[1].scatter(y_test / 100_000, preds / 100_000, alpha=0.3, color='teal')
axes[1].plot([0, y_test.max()/100_000], [0, y_test.max()/100_000], 'r--', lw=2)
axes[1].set_title('Actual vs Predicted Prices (Lakh PKR)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Actual Price (Lakh PKR)')
axes[1].set_ylabel('Predicted Price (Lakh PKR)')

plt.tight_layout()
plt.savefig('car_price_analysis_plots.png', dpi=300)
print("\n--- PLOTS CREATED & SAVED AS 'car_price_analysis_plots.png' ---")
plt.show()