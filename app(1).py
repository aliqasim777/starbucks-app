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
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Starbucks Decision Tree Predictor",
    page_icon="☕",
    layout="centered",
)

st.title("☕ Starbucks Decision Tree Predictor")
st.write("Enter nutritional values to predict the Starbucks beverage category.")

# Folder mein jo bhi .pkl file hogi, yeh khud-b-khud utha lega!
current_dir = Path(__file__).resolve().parent
pkl_files = list(current_dir.glob("*.pkl"))

if not pkl_files:
    st.error("Koi bhi model (.pkl) file nahi mili! Kindly GitHub par model file upload karein.")
    st.stop()

MODEL_PATH = pkl_files[0]

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Model load nahi ho saka: {exc}")
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
