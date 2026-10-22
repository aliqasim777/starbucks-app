import streamlit as st
import pandas as pd
import joblib

# Saved model load karein
model = joblib.load('starbucks_model.pkl')

st.title("Starbucks Decision Tree Predictor")
st.write("Yeh app Starbucks dataset par trained Decision Tree model ka use karti hai.")

# User se input lene ke liye fields
calories = st.number_input("Calories", min_value=0.0, max_value=500.0, value=200.0)
protein = st.number_input("Protein (g)", min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict"):
    # Input data ko DataFrame mein convert karein
    input_data = pd.DataFrame([[calories, protein]], columns=['num__Calories', 'num__Protein (g)'])
    
    # Prediction karein
    prediction = model.predict(input_data)
    st.success(f"Predicted Result: {prediction[0]}")
