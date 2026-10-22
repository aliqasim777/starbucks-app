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
    import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Starbucks Decision Tree Predictor",
    page_icon="☕",
    layout="centered",
)

st.title("☕ Starbucks Decision Tree Predictor")
st.write("Enter nutritional values to predict the Starbucks beverage category.")

# App ke andar hi dataset load karke model train kar lenge (No version error!)
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('starbucks.csv')
    
    target_column = 'Beverage_category' if 'Beverage_category' in df.columns else df.columns[0]
    
    X = df[['Calories', 'Protein (g)']].copy()
    X.fillna(X.median(), inplace=True)
    y = df[target_column]
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X, y)
    return model

try:
    model = load_and_train_model()
except Exception as exc:
    st.error(f"Dataset load ya model training mein masla hai: {exc}. Kindly check karein ke 'starbucks.csv' GitHub repository mein mojood hai.")
    st.stop()

st.subheader("Input Features")
calories = st.number_input(
    "Calories", min_value=0.0, max_value=1000.0, value=200.0, step=1.0
)
protein = st.number_input(
    "Protein (g)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
)

if st.button("Predict", type="primary"):
    try:
        input_data = pd.DataFrame({
            "Calories": [calories],
            "Protein (g)": [protein],
        })
        prediction = model.predict(input_data)
        st.success(f"Predicted Result: {prediction[0]}")
    except Exception as exc:
        st.error(f"Prediction nahi ho saki: {exc}")
