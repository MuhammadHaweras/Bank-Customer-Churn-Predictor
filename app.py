import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load('./RandomForestClassifier.pkl')


st.markdown("""
# 🏛 Bank Customer Churn Predictor
<hr style="border:1px solid #ccc">

## 👨‍💻 Made with ❤ by: **Haweras**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-haweras-7aa6b11b2/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/MuhammadHaweras)

<hr style="border:1px solid #ccc">
""", unsafe_allow_html=True)

with st.form("customer_form"):
    st.markdown("## 📝 Enter Customer Details")
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("💳 Credit Score", min_value=0, help="Customer's credit score")
        age = st.number_input("📅 Age", min_value=16, max_value=100, value=18)
        tenure = st.number_input("⏳ Tenure (years)", min_value=0, max_value=80)
        balance = st.number_input("💰 Balance", min_value=0.0, value=50000.0)
        num_of_products = st.selectbox("🛒 Number of Products", [1, 2, 3, 4], index=0)
        points_earned = st.number_input("🏆 Points Earned", min_value=0)
        satisfaction_score = st.slider("😊 Satisfaction Score", min_value=1, max_value=5)
    with col2:
        gender = st.radio("👤 Gender", ['Male', 'Female'], horizontal=True)
        geography = st.selectbox("🌍 Geography", ['France', 'Germany', 'Spain'])
        card_type = st.selectbox("💳 Card Type", ['Silver', 'Gold', 'Platinum'])
        has_cr_card = st.radio("💳 Has Credit Card", ["Yes", "No"], horizontal=True)
        is_active_member = st.radio("✅ Is Active Member", ["Yes", "No"], horizontal=True)
        complain = st.radio("❗ Complain", ["Yes", "No"], horizontal=True)
        estimated_salary = st.number_input("💸 Estimated Salary", min_value=0.0, value=60000.0)

    st.markdown("---")
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        gender_encoded = 1 if gender == 'Male' else 0
        geo_map = {'France': 0, 'Germany': 1, 'Spain': 2}
        geography_encoded = geo_map[geography]
        card_type_map = {'Silver': 0, 'Gold': 1, 'Platinum': 2}
        card_type_encoded = card_type_map[card_type]
        has_cr_card_encoded = 1 if has_cr_card == 'Yes' else 0
        is_active_member_encoded = 1 if is_active_member == 'Yes' else 0
        complain_encoded = 1 if complain == 'Yes' else 0

        input_data = pd.DataFrame([{
            'CreditScore': credit_score,
            'Geography': geography_encoded,
            'Gender': gender_encoded,
            'Age': age,
            'Tenure(Years)': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCard': has_cr_card_encoded,
            'IsActiveMember': is_active_member_encoded,
            'EstimatedSalary': estimated_salary,
            'Complain': complain_encoded,
            'Satisfaction Score': satisfaction_score,
            'Card Type': card_type_encoded,
            'Point Earned': points_earned,
        }])

        prob = model.predict_proba(input_data)[0]
        pred = model.predict(input_data)[0]
        pred_label = "Churned" if pred == 1 else "Not Churned"
        st.markdown(f"### 🏦 Prediction: <span style='color: {'red' if pred == 1 else 'green'};'>{pred_label}</span>", unsafe_allow_html=True)
        if pred == 1:
            st.progress(prob[1])
            st.metric(label="Chance of the Customer will Churn", value=f"{prob[1]*100:.2f}%")
            st.metric(label="Chance of the Customer will NOT Churn", value=f"{prob[0]*100:.2f}%")
        else:
            st.progress(prob[0])
            st.metric(label="Chance of the Customer will NOT Churn", value=f"{prob[0]*100:.2f}%")
            st.metric(label="Chance of the Customer will Churn", value=f"{prob[1]*100:.2f}%")

