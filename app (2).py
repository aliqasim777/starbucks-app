import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Starbucks Decision Tree Predictor", page_icon="☕")

st.title("Starbucks Decision Tree Predictor")
st.write("Yeh app saved Decision Tree model se prediction karti hai.")

MODEL_PATH = Path(__file__).with_name("starbucks_model.pkl")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error(
        "Model file 'starbucks_model.pkl' nahi mili. "
        "Is file ko app.py ke same folder/repository mein upload karein."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Model load nahi ho saka: {e}")
    st.stop()

# The saved model expects only one feature: Calories.
st.subheader("Input Feature")
calories = st.number_input(
    "Calories",
    min_value=0.0,
    max_value=500.0,
    value=200.0,
    step=1.0
)

if st.button("Predict", type="primary"):
    try:
        # Use the exact feature name expected by the saved model.
        input_data = pd.DataFrame({"Calories": [calories]})
        prediction = model.predict(input_data)
        st.success(f"Predicted Result: {prediction[0]}")
    except Exception as e:
        st.error(
            "Prediction nahi ho saki. Saved model ke expected input format ko check karein. "
            f"Technical detail: {e}"
        )
