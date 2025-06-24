import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

# Load data
df = pd.read_excel("measurements2.xlsx", engine='openpyxl')

# Ensure all columns are numerical
df = df.apply(pd.to_numeric, errors='coerce')

# Check for NaN values and handle them
if df.isnull().values.any():
    st.warning("Data contains NaN values. Handling NaN values by filling with column mean.")
    df.fillna(df.mean(), inplace=True)

# Check for infinite values and handle them
if np.isinf(df.values).any():
    st.warning("Data contains infinite values. Handling infinite values by replacing them with NaN and then filling with column mean.")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

# Check for very large values (optional step)
if (df > 1e308).values.any():
    st.warning("Data contains very large values. Please ensure data is in a reasonable range.")
    # Handle very large values here (e.g., cap them to a maximum value)

# Prepare features and target variable
features = df[["distance", "speed", "temp_inside", "temp_outside", "AC", "rain", "sun"]]
target = df["consume"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Fuel Consumption Prediction")

st.write("""
This app predicts the fuel consumption of a car based on user input values.
""")

# User input fields
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
speed = st.number_input("Speed (km/h)", min_value=0.0, step=0.1)
temp_inside = st.number_input("Temperature Inside (°C)", min_value=-10.0, step=0.1)
temp_outside = st.number_input("Temperature Outside (°C)", min_value=-50.0, step=0.1)
AC = st.selectbox("AC Usage", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
rain = st.selectbox("Rain", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
sun = st.selectbox("Sun", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Prediction
if st.button("Predict"):
    user_input = np.array([[distance, speed, temp_inside, temp_outside, AC, rain, sun]])
    prediction = model.predict(user_input)
    st.write(f"Predicted Fuel Consumption: {prediction[0]:.2f} liters")

# Plotting predictions vs actual values (optional)
if st.checkbox("Show Actual vs Predicted Fuel Consumption"):
    y_pred_test = model.predict(X_test)
    plt.scatter(y_test, y_pred_test)
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.title("Actual vs Predicted Fuel Consumption")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    st.pyplot(plt)

# Run the app using the command: streamlit run your_script.py
# Open your terminal or command prompt, navigate to the directory where your script is saved, and run the following command:
#cd C:\Users\PRERANA\Desktop\PDEU\Internships\Brainybeam Project
#streamlit run fuel_consumption.py