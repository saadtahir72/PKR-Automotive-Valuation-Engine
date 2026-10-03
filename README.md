# 🚗 PKR Automotive Valuation Engine

An interactive machine learning dashboard powered by **Python**, **XGBoost**, and **Streamlit** to estimate used car prices in Pakistan. The model integrates technical vehicle attributes with macroeconomic indicators (USD/PKR exchange rates, YoY inflation, crude oil prices, and interest rates) to deliver inflation-adjusted market valuations.

---

## 📌 Features
- **Machine Learning Core**: XGBoost Regressor ($R^2 \approx 0.745$, MAE $\approx$ PKR 1.32M).
- **Macroeconomic Integration**: Evaluates vehicle values dynamically against macroeconomic volatility.
- **Interactive UI**: Dark-themed Streamlit dashboard with dynamic brand/model filtering and instantaneous valuation in Lacs and Crores.

---

## 🛠️ Project Structure
- `1_data_preprocessing.py` - Cleans raw PakWheels listings, extracts Make/Model, and merges annual macro trends.
- `2_train_models.py` - Trains Random Forest and XGBoost regressors and evaluates feature importances.
- `3_evaluate_and_plot.py` - Generates evaluation charts (`car_price_analysis_plots.png`).
- `4_predict_car_price.py` - CLI prediction tool.
- `app.py` - Streamlit Web Dashboard application.

---

## 🚀 Local Setup & Execution
```powershell
# Clone Repository
git clone [https://github.com/saadtahir72/PKR-Automotive-Valuation-Engine.git](https://github.com/saadtahir72/PKR-Automotive-Valuation-Engine.git)
cd PKR-Automotive-Valuation-Engine

# Virtual Environment Setup
python -m venv venv
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Launch Streamlit Application
streamlit run app.py