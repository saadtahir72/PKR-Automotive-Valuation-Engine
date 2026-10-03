import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# 1. PAGE SETUP
st.set_page_config(
    page_title="PKR Automotive Valuation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS STYLING
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0e1626 100%);
        color: #f0f6fc;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f6beb 0%, #238636 100%);
        color: white !important;
        font-weight: bold;
        font-size: 18px;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. MODEL & DATA LOADING
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('cleaned_pakwheels_macro.csv')
    
    # Filter top brands to keep UI clean
    top_makes = df['Make'].value_counts().head(15).index
    df = df[df['Make'].isin(top_makes)].copy()
    
    # Label Encoders for Categorical Variables
    le_make = LabelEncoder()
    le_model = LabelEncoder()
    
    df['Make_Enc'] = le_make.fit_transform(df['Make'])
    df['Model_Enc'] = le_model.fit_transform(df['Model'])
    
    feature_cols = [
        'Make_Enc', 'Model_Enc', 'Year', 'Millage_KM', 'Engine_CC', 
        'Exchange_Rate_PKR_USD', 'Inflation_YoY', 
        'Oil_Price_USD_Barrel', 'Interest_Rate', 'Low_Tax_Bracket'
    ]
    
    df_model = df.dropna(subset=feature_cols + ['Price_PKR']).copy()
    X = df_model[feature_cols]
    y = df_model['Price_PKR']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X, y)
    
    latest_macro = df.sort_values(by='Year', ascending=False).iloc[0]
    return model, df, le_make, le_model, latest_macro

model, df, le_make, le_model, latest_macro = load_data_and_model()

# 4. SIDEBAR - MACRO ECONOMIC INDICATORS
with st.sidebar:
    st.header("📈 Macro Indicators")
    st.caption("Current economic context driving Pakistani vehicle prices:")
    st.metric("USD / PKR Rate", f"PKR {latest_macro['Exchange_Rate_PKR_USD']:.1f}")
    st.metric("YoY Inflation", f"{latest_macro['Inflation_YoY']:.1f}%")
    st.metric("Crude Oil", f"${latest_macro['Oil_Price_USD_Barrel']:.1f} / bbl")
    st.metric("Policy Rate", f"{latest_macro['Interest_Rate']:.1f}%")

# 5. MAIN CONTENT AREA
st.title("🚗 Pakistan Used Car Price Predictor")
st.markdown("##### *Select vehicle brand, model, year, and usage to predict current market price.*")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🛠️ Select Vehicle Specifications")
    
    # Company Dropdown
    makes = sorted(df['Make'].unique())
    selected_make = st.selectbox("Car Brand / Company", makes, index=makes.index("Suzuki") if "Suzuki" in makes else 0)
    
    # Model Dropdown (Filtered based on chosen brand)
    available_models = sorted(df[df['Make'] == selected_make]['Model'].unique())
    selected_model = st.selectbox("Car Model", available_models)
    
    # Auto-populate average CC for selected model
    avg_cc = float(df[(df['Make'] == selected_make) & (df['Model'] == selected_model)]['Engine_CC'].median())
    if np.isnan(avg_cc):
        avg_cc = 1300.0
        
    year = st.slider("Model Year", min_value=1990, max_value=2026, value=2021)
    engine_cc = st.number_input("Engine Capacity (CC)", min_value=600, max_value=5000, value=int(avg_cc), step=50)
    mileage = st.number_input("Mileage Driven (KM)", min_value=0, max_value=500000, value=60000, step=5000)

with col_right:
    st.subheader("📊 Market Valuation Output")
    st.write(f"Evaluating **{selected_make} {selected_model} ({year})**")
    
    predict_btn = st.button("🚀 Calculate Estimated Value", use_container_width=True)
    
    if predict_btn:
        make_enc = le_make.transform([selected_make])[0]
        model_enc = le_model.transform([selected_model])[0]
        low_tax = 1 if engine_cc <= 1000 else 0
        
        input_df = pd.DataFrame([{
            'Make_Enc': make_enc,
            'Model_Enc': model_enc,
            'Year': year,
            'Millage_KM': mileage,
            'Engine_CC': engine_cc,
            'Exchange_Rate_PKR_USD': float(latest_macro['Exchange_Rate_PKR_USD']),
            'Inflation_YoY': float(latest_macro['Inflation_YoY']),
            'Oil_Price_USD_Barrel': float(latest_macro['Oil_Price_USD_Barrel']),
            'Interest_Rate': float(latest_macro['Interest_Rate']),
            'Low_Tax_Bracket': low_tax
        }])
        
        predicted_price = model.predict(input_df)[0]
        lacs = predicted_price / 100_000
        crores = predicted_price / 10_000_000
        
        st.markdown("---")
        if predicted_price >= 10_000_000:
            st.metric(label="Estimated Market Value", value=f"{crores:.2f} Crore PKR")
        else:
            st.metric(label="Estimated Market Value", value=f"{lacs:.2f} Lac PKR")
            
        st.info(f"💡 Exact Valuation Output: **PKR {predicted_price:,.2f}**")