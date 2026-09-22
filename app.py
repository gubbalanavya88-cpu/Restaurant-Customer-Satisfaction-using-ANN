import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf


# ==========================================
# Load Saved Model and Preprocessing Files
# ==========================================

model = tf.keras.models.load_model(
    "restaurant_satisfaction_ann.h5"
)

scaler = joblib.load(
    "restaurant_satisfaction_scaler.pkl"
)

feature_columns = joblib.load(
    "restaurant_satisfaction_features.pkl"
)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Restaurant Customer Satisfaction",
    page_icon="🍽️",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("🍽️ Restaurant Customer Satisfaction Prediction")

st.write(
    "Enter customer and restaurant visit details "
    "to predict whether the customer is highly satisfied."
)


# ==========================================
# Customer Information
# ==========================================

st.header("Customer Information")


age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


income = st.number_input(
    "Income",
    min_value=0.0,
    value=40000.0
)


visit_frequency = st.selectbox(
    "Visit Frequency",
    ["Daily", "Weekly", "Monthly", "Rarely"]
)


average_spend = st.number_input(
    "Average Spend",
    min_value=0.0,
    value=800.0
)


preferred_cuisine = st.selectbox(
    "Preferred Cuisine",
    [
        "Indian",
        "Chinese",
        "Italian",
        "Mexican",
        "Other"
    ]
)


time_of_visit = st.selectbox(
    "Time of Visit",
    [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]
)


group_size = st.number_input(
    "Group Size",
    min_value=1,
    max_value=20,
    value=2
)


dining_occasion = st.selectbox(
    "Dining Occasion",
    [
        "Casual",
        "Birthday",
        "Business",
        "Date",
        "Family",
        "Other"
    ]
)


meal_type = st.selectbox(
    "Meal Type",
    [
        "Breakfast",
        "Lunch",
        "Dinner",
        "Snack"
    ]
)


online_reservation = st.selectbox(
    "Online Reservation",
    ["Yes", "No"]
)


delivery_order = st.selectbox(
    "Delivery Order",
    ["Yes", "No"]
)


loyalty_program = st.selectbox(
    "Loyalty Program Member",
    ["Yes", "No"]
)


wait_time = st.number_input(
    "Wait Time (minutes)",
    min_value=0,
    max_value=180,
    value=10
)


service_rating = st.slider(
    "Service Rating",
    min_value=1,
    max_value=5,
    value=4
)


food_rating = st.slider(
    "Food Rating",
    min_value=1,
    max_value=5,
    value=4
)


ambiance_rating = st.slider(
    "Ambiance Rating",
    min_value=1,
    max_value=5,
    value=4
)


# ==========================================
# Prediction
# ==========================================

if st.button("Predict Customer Satisfaction"):

    # Create input data
    input_data = {
        'Age': [age],
        'Gender': [gender],
        'Income': [income],
        'VisitFrequency': [visit_frequency],
        'AverageSpend': [average_spend],
        'PreferredCuisine': [preferred_cuisine],
        'TimeOfVisit': [time_of_visit],
        'GroupSize': [group_size],
        'DiningOccasion': [dining_occasion],
        'MealType': [meal_type],
        'OnlineReservation': [online_reservation],
        'DeliveryOrder': [delivery_order],
        'LoyaltyProgramMember': [loyalty_program],
        'WaitTime': [wait_time],
        'ServiceRating': [service_rating],
        'FoodRating': [food_rating],
        'AmbianceRating': [ambiance_rating]
    }

    # Convert input into DataFrame
    input_df = pd.DataFrame(input_data)

    # Convert categorical variables into numerical variables
    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )

    # Match training feature columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Apply the saved scaler
    input_array = scaler.transform(input_df)

    # Make prediction
    predictions = model.predict(
        input_array,
        verbose=0
    )

    # Convert prediction probability into class
    predicted_class = predictions[0][0] >= 0.5


    # ==========================================
    # Display Prediction
    # ==========================================

    st.subheader("Prediction Result")

    if predicted_class:

        st.success(
            "😊 Customer is Highly Satisfied"
        )

    else:

        st.warning(
            "😐 Customer is Not Highly Satisfied"
        )