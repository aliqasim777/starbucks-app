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
    st.error("Model file 'starbucks_model.pkl' app.py ke same folder mein nahi mili.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Model load nahi ho saka: {e}")
    st.stop()

st.subheader("Input Features")
calories = st.number_input(
    "Calories", min_value=0.0, max_value=500.0, value=200.0, step=1.0
)
protein = st.number_input(
    "Protein (g)", min_value=0.0, max_value=50.0, value=5.0, step=0.5
)

if st.button("Predict", type="primary"):
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        expected = list(expected)
        if expected == ["Calories"]:
            st.warning(
                "Protein ka input show ho raha hai, lekin current saved model sirf "
                "Calories par trained hai. Prediction ke liye Protein use nahi hoga. "
                "Dono features use karne ke liye model ko Calories aur Protein (g) "
                "dono par dobara train karke starbucks_model.pkl replace karein."
            )
            input_data = pd.DataFrame({"Calories": [calories]})
        elif expected == ["Calories", "Protein (g)"]:
            input_data = pd.DataFrame({
                "Calories": [calories],
                "Protein (g)": [protein]
            })
        elif expected == ["num__Calories", "num__Protein (g)"]:
            input_data = pd.DataFrame({
                "num__Calories": [calories],
                "num__Protein (g)": [protein]
            })
        else:
            st.error(
                f"Model ko ye features chahiye: {expected}. "
                "App aur training ke feature columns same hone chahiye."
            )
            st.stop()
    else:
        n_features = getattr(model, "n_features_in_", None)
        if n_features == 1:
            st.warning(
                "Current model sirf 1 feature (Calories) par trained hai. "
                "Protein input filhal prediction mein use nahi hoga."
            )
            input_data = pd.DataFrame({"Calories": [calories]})
        elif n_features == 2:
            input_data = pd.DataFrame({
                "Calories": [calories],
                "Protein (g)": [protein]
            })
        else:
            st.error(
                f"Model {n_features} features expect karta hai. "
                "Training code ke mutabiq app inputs set karne honge."
            )
            st.stop()

    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Result: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction nahi ho saki: {e}")import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.title("Starbucks Decision Tree Predictor")
st.write("Yeh app dataset ko load karke on-the-fly model train karti hai aur prediction deti hai.")

# Model ko cache kar lete hain taake bar bar train na ho
@st.cache_resource
def get_trained_model():
    # GitHub repository mein starbucks.csv honi chahiye
    df = pd.read_csv('starbucks.csv')
    
    target_column = 'Beverage_category' if 'Beverage_category' in df.columns else df.columns[0]
    
    # Columns auto-detect karna
    cal_col = [c for c in df.columns if 'cal' in c.lower()][0]
    prot_col = [c for c in df.columns if 'prot' in c.lower()][0]
    
    X = df[[cal_col, prot_col]].copy()
    X.fillna(X.median(), inplace=True)
    y = df[target_column]
    
    # Decision Tree Model Train karna
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X, y)
    
    return model, cal_col, prot_col

try:
    model, cal_col, prot_col = get_trained_model()

    # User inputs
    calories = st.number_input("Calories", min_value=0.0, max_value=500.0, value=200.0)
    protein = st.number_input("Protein (g)", min_value=0.0, max_value=50.0, value=5.0)

    if st.button("Predict"):
        input_data = pd.DataFrame([[calories, protein]], columns=[cal_col, prot_col])
        prediction = model.predict(input_data)
        st.success(f"Predicted Result: {prediction[0]}")

except Exception as e:
    st.error(f"Error: {e}. Kindly ensure 'starbucks.csv' is uploaded in your GitHub repository.")
