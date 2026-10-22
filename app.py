import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Starbucks Decision Tree Predictor",
    page_icon="☕",
    layout="centered",
)

# Dark, compact style inspired by the reference screenshot.
st.markdown("""
<style>
    .stApp { background: #0e1117; color: #f5f5f5; }
    [data-testid="stHeader"] { background: rgba(14,17,23,0); }
    [data-testid="stWidgetLabel"] p { color: #f5f5f5; }
    div.stButton > button {
        background: #262833; color: white; border: 1px solid #4b4e5b;
        border-radius: 8px;
    }
    div.stButton > button:hover {
        background: #343744; color: white; border-color: #777b8a;
    }
    .note {
        padding: 12px 14px; border-radius: 8px;
        background: #22232b; border: 1px solid #393b47;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("☕ Starbucks Decision Tree Predictor")
st.write("Enter nutritional values to predict the Starbucks beverage category.")

MODEL_PATH = Path(__file__).resolve().parent / "starbucks_model.pkl"

if not MODEL_PATH.exists():
    st.error("Model file 'starbucks_model.pkl' nahi mili. Isay app.py ke saath same GitHub folder mein upload karein.")
    st.stop()

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model load nahi ho saka: {exc}")
    st.stop()

# The uploaded model was inspected: it expects exactly Calories and Protein (g).
expected = list(getattr(model, "feature_names_in_", []))
if expected and expected != ["Calories", "Protein (g)"]:
    st.warning(
        "Uploaded model ke features app ke expected features se different hain: "
        + ", ".join(expected)
    )

st.subheader("Input Features")
calories = st.number_input(
    "Calories", min_value=0.0, max_value=1000.0, value=200.0, step=1.0
)
protein = st.number_input(
    "Protein (g)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
)

st.markdown(
    '<div class="note">This saved model was trained using <b>Calories</b> and '
    '<b>Protein (g)</b>. Sugar, Caffeine, Fat and Price are not included in this '
    'model, so they are not shown as prediction inputs.</div>',
    unsafe_allow_html=True,
)

if st.button("Predict", type="primary", use_container_width=False):
    try:
        input_data = pd.DataFrame({
            "Calories": [calories],
            "Protein (g)": [protein],
        })
        prediction = model.predict(input_data)
        st.success(f"Predicted Result: {prediction[0]}")
    except Exception as exc:
        st.error(f"Prediction nahi ho saki: {exc}")
        st.caption("Confirm karein ke GitHub wali model file nayi two-feature model hai.")
