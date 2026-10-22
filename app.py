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
    st.error(f"Model load nahi hua: {e}")
    st.stop()

st.subheader("Input Features")
calories = st.number_input("Calories", min_value=0.0, max_value=500.0, value=200.0, step=1.0)
protein = st.number_input("Protein (g)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

if st.button("Predict", type="primary"):
    # For a scikit-learn Pipeline, feature_names_in_ usually contains the
    # original input columns. For an estimator trained on transformed data,
    # feature_names_in_ may contain names such as num__Calories.
    expected = getattr(model, "feature_names_in_", None)

    if expected is not None:
        expected = list(expected)

        if expected == ["Calories", "Protein (g)"]:
            input_data = pd.DataFrame(
                [[calories, protein]],
                columns=["Calories", "Protein (g)"]
            )
        elif expected == ["num__Calories", "num__Protein (g)"]:
            input_data = pd.DataFrame(
                [[calories, protein]],
                columns=["num__Calories", "num__Protein (g)"]
            )
        else:
            st.error(
                "Saved model in app ke 2 inputs se match nahi karta. "
                f"Model ko ye features chahiye: {expected}. "
                "Model training ke waqt use hone wale tamam features app mein dene honge, "
                "ya model ko sirf Calories aur Protein (g) par dobara train/save karein."
            )
            st.stop()
    else:
        # If the model has no feature names, check the number of expected features.
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and n_features != 2:
            st.error(
                f"Saved model {n_features} features expect karta hai, lekin app sirf 2 bhej rahi hai. "
                "Training aur app ke features same hone chahiye."
            )
            st.stop()

        input_data = pd.DataFrame(
            [[calories, protein]],
            columns=["Calories", "Protein (g)"]
        )

    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Result: {prediction[0]}")
    except Exception as e:
        st.error(
            "Prediction nahi ho saki. Model aur app ke input features match nahi kar rahe. "
            f"Technical detail: {e}"
        )
