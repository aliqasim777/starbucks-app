import streamlit as st
import pandas as pd
import joblib

# Saved model load karein
model = joblib.load('starbucks_model.pkl')

st.title("Starbucks Decision Tree Predictor")
st.write("Yeh app Starbucks dataseimport streamlit as st
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Starbucks Decision Tree Predictor",
    page_icon="☕",
    layout="centered",
)

st.title("☕ Starbucks Decision Tree Predictor")
st.write("Decision Tree Machine Learning Model Prediction App")

@st.cache_resource
def get_model():
    df = pd.read_csv('starbucks.csv')
    
    target_column = 'Beverage_category' if 'Beverage_category' in df.columns else df.columns[0]
    
    X = df[['Calories', 'Protein (g)']].copy()
    X['Calories'].fillna(X['Calories'].median(), inplace=True)
    X['Protein (g)'].fillna(X['Protein (g)'].median(), inplace=True)
    y = df[target_column]
    
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X, y)
    return clf

try:
    model = get_model()
except Exception as e:
    st.error(f"Dataset error: {e}. Ensure 'starbucks.csv' is uploaded in GitHub.")
    st.stop()

st.subheader("Input Features")
calories = st.number_input("Calories", min_value=0.0, max_value=1000.0, value=200.0, step=1.0)
protein = st.number_input("Protein (g)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

if st.button("Predict", type="primary"):
    try:
        input_df = pd.DataFrame({"Calories": [calories], "Protein (g)": [protein]})
        pred = model.predict(input_df)
        st.success(f"Predicted Result: {pred[0]}")
    except Exception as err:
        st.error(f"Prediction error: {err}")
