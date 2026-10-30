
import streamlit as st
import joblib
import pandas as pd
from datetime import date

# Load the trained model and feature names
@st.cache_resource
def load_model():
    artifact = joblib.load("freight_rate_model.pkl")
    return artifact["model"], artifact["features"]

model, features = load_model()

st.set_page_config(
    page_title="Freight Rate Predictor",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Freight Rate Prediction")
st.write(
    "Enter shipment details to estimate the freight rate "
    "using the trained Machine Learning model."
)

with st.form("freight_form"):
    pickup = st.text_input("Pickup city", value="Lexington")
    delivery = st.text_input("Delivery city", value="Fort Wayne")
    distance = st.number_input(
        "Distance (miles)", min_value=1, value=360
    )
    equipment = st.selectbox(
        "Equipment",
        ["Dry Van", "Reefer", "Flatbed"]
    )
    weight = st.number_input(
        "Weight (lbs)", min_value=1, value=32000
    )
    shipment_date = st.date_input(
        "Shipment date", value=date(2025, 12, 1)
    )

    submitted = st.form_submit_button("Predict Freight Rate")

if submitted:
    input_data = pd.DataFrame([{
        "pickup": pickup,
        "delivery": delivery,
        "distance": distance,
        "equipment": equipment,
        "weight": weight,
        "year": shipment_date.year,
        "month_num": shipment_date.month,
        "day": shipment_date.day,
        "day_of_week": shipment_date.weekday()
    }])

    # Match the feature order used during training
    input_data = input_data[features]
    prediction = float(model.predict(input_data)[0])

    st.success(f"Predicted freight rate: ${prediction:,.2f}")
    st.caption(
        "This is a model estimate, not a guaranteed market quote."
    )
