"""Streamlit app for Wellness Tourism Package purchase prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "wellness_tourism_model.joblib"

FEATURE_COLUMNS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("The trained model is not available yet. Run the GitHub Actions pipeline first.")
        st.stop()
    return joblib.load(MODEL_PATH)


def yes_no_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


st.set_page_config(page_title="Wellness Tourism Predictor", layout="wide")
st.title("Wellness Tourism Predictor")

model = load_model()

with st.form("customer_form"):
    left, middle, right = st.columns(3)

    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
        type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3], index=0)
        duration_of_pitch = st.number_input(
            "Duration of pitch", min_value=0, max_value=120, value=15, step=1
        )
        occupation = st.selectbox(
            "Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"]
        )
        gender = st.selectbox("Gender", ["Female", "Male"])

    with middle:
        number_of_person_visiting = st.number_input(
            "People visiting", min_value=1, max_value=10, value=3, step=1
        )
        number_of_followups = st.number_input(
            "Follow-ups", min_value=0, max_value=10, value=3, step=1
        )
        product_pitched = st.selectbox(
            "Product pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
        )
        preferred_property_star = st.selectbox("Preferred property star", [3, 4, 5], index=0)
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        number_of_trips = st.number_input("Annual trips", min_value=0, max_value=30, value=2, step=1)

    with right:
        passport = st.selectbox("Passport", ["No", "Yes"])
        pitch_satisfaction_score = st.slider("Pitch satisfaction score", 1, 5, 3)
        own_car = st.selectbox("Own car", ["No", "Yes"])
        number_of_children_visiting = st.number_input(
            "Children visiting", min_value=0, max_value=10, value=1, step=1
        )
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        monthly_income = st.number_input(
            "Monthly income", min_value=0, max_value=500000, value=22000, step=1000
        )

    submitted = st.form_submit_button("Predict purchase likelihood", use_container_width=True)

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": type_of_contact,
                "CityTier": city_tier,
                "DurationOfPitch": duration_of_pitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": number_of_person_visiting,
                "NumberOfFollowups": number_of_followups,
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": preferred_property_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": number_of_trips,
                "Passport": yes_no_to_int(passport),
                "PitchSatisfactionScore": pitch_satisfaction_score,
                "OwnCar": yes_no_to_int(own_car),
                "NumberOfChildrenVisiting": number_of_children_visiting,
                "Designation": designation,
                "MonthlyIncome": monthly_income,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    probability = model.predict_proba(input_df)[0, 1]
    prediction = int(probability >= 0.5)

    metric_col, action_col = st.columns([1, 2])
    metric_col.metric("Purchase probability", f"{probability:.1%}")

    if prediction == 1:
        action_col.success("High-potential customer. Prioritize outreach for this campaign.")
    else:
        action_col.info("Lower purchase likelihood. Consider nurturing before direct sales outreach.")

    st.dataframe(input_df, use_container_width=True, hide_index=True)
