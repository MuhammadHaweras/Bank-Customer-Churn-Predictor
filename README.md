# Bank Customer Churn Prediction

This project predicts whether a bank customer is likely to churn using various machine learning models. It includes data analysis, feature engineering, model training, and a user-friendly Streamlit web app for live predictions.

## Features

- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and preprocessing
- Multiple ML models: SVM, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, CatBoost
- Model evaluation and comparison
- Interactive Streamlit app for real-time churn prediction

## Streamlit App

Try the live demo here:  
[Bank Customer Churn Predictor](https://bank-customer-churn-predictor-haweras.streamlit.app/)

## How to Use

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/bank-customer-churn-project.git
   cd bank-customer-churn-project
   ```
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Run locally**
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py` — Streamlit web app
- `bank_customer_churn.ipynb` — Jupyter notebook for analysis and modeling
- `best_model.pkl` — Saved trained model
- `requirements.txt` — Python dependencies

## Data

The dataset contains customer information such as age, credit score, balance, products, card type, complaints, satisfaction score, and more.
