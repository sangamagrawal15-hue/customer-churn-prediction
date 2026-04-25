import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Title
st.title("✈️ Customer Churn Prediction")
st.markdown("### Travel Industry — Powered by Random Forest")
st.markdown("---")

st.markdown("Fill in the customer details below to predict whether they are likely to **churn**.")

# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", min_value=18, max_value=70, value=30)
    frequent_flyer = st.selectbox("✈️ Frequent Flyer?", options=["No", "Yes"])
    annual_income = st.selectbox("💰 Annual Income Class",
                                  options=["Low Income", "Middle Income", "High Income"])

with col2:
    services_opted = st.slider("🛎️ Services Opted", min_value=1, max_value=6, value=3)
    account_synced = st.selectbox("📱 Account Synced to Social Media?", options=["No", "Yes"])
    booked_hotel = st.selectbox("🏨 Booked Hotel?", options=["No", "Yes"])

st.markdown("---")

# Encode inputs
def encode_input():
    ff = 1 if frequent_flyer == "Yes" else 0
    ac = 1 if account_synced == "Yes" else 0
    bh = 1 if booked_hotel == "Yes" else 0
    income_map = {"Low Income": 0, "Middle Income": 1, "High Income": 2}
    inc = income_map[annual_income]
    return pd.DataFrame([[age, ff, inc, services_opted, ac, bh]],
                        columns=['Age', 'FrequentFlyer', 'AnnualIncomeClass',
                                 'ServicesOpted', 'AccountSyncedToSocialMedia',
                                 'BookedHotelOrNot'])

if st.button("🔍 Predict Churn", use_container_width=True):
    input_df = encode_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **This customer is likely to CHURN**")
        st.metric("Churn Probability", f"{probability * 100:.1f}%")
        st.markdown("**Recommended Action:** Offer personalized discounts or loyalty rewards.")
    else:
        st.success(f"✅ **This customer is NOT likely to churn**")
        st.metric("Churn Probability", f"{probability * 100:.1f}%")
        st.markdown("**Status:** Customer appears satisfied. Keep engagement high!")

    # Show input summary
    with st.expander("📋 Input Summary"):
        st.write(input_df)

st.markdown("---")
st.caption("Built with ❤️ using Random Forest | B.Tech Gen AI – Final Project")
