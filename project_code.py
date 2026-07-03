import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# -----------------------------------
# Custom CSS Styling
# -----------------------------------
st.markdown("""
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a1b9a 100%);
        font-family: "Segoe UI", sans-serif;
        color: #fff;
    }

    /* Centered title */
    h1 {
        text-align: center;
        font-weight: 800;
        font-size: 2.2rem;
        color: #ffffff;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #e0e0e0;
        margin-bottom: 1.5em;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        animation: fadeIn 1s ease;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #6a1b9a, #1e88e5);
        color: white;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        padding: 12px 25px;
        transition: all 0.3s ease;
        border: none;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #8e24aa, #42a5f5);
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }

    /* Result card */
    .result-card {
        border-radius: 16px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
        font-weight: 700;
        color: #fff;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        animation: slideUp 0.8s ease;
    }
    .setosa { background: linear-gradient(135deg, #43a047, #66bb6a); }
    .versicolor { background: linear-gradient(135deg, #1e88e5, #42a5f5); }
    .virginica { background: linear-gradient(135deg, #8e24aa, #ab47bc); }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #42a5f5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("Iris.csv")
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# -----------------------------------
# Load Model
# -----------------------------------
model = pickle.load(open("model.pkl", "rb"))

# -----------------------------------
# Title
# -----------------------------------
st.markdown("<h1>🌸 Iris Flower Predictor</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">An elegant AI app to classify iris species</p>', unsafe_allow_html=True)

# -----------------------------------
# Glassmorphism Card for Inputs
# -----------------------------------
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)",
            min_value=float(df["SepalLengthCm"].min()),
            max_value=float(df["SepalLengthCm"].max()),
            value=float(round(df["SepalLengthCm"].mean(), 1)),
            step=0.1
        )
        petal_length = st.slider("Petal Length (cm)",
            min_value=float(df["PetalLengthCm"].min()),
            max_value=float(df["PetalLengthCm"].max()),
            value=float(round(df["PetalLengthCm"].mean(), 1)),
            step=0.1
        )

    with col2:
        sepal_width = st.slider("Sepal Width (cm)",
            min_value=float(df["SepalWidthCm"].min()),
            max_value=float(df["SepalWidthCm"].max()),
            value=float(round(df["SepalWidthCm"].mean(), 1)),
            step=0.1
        )
        petal_width = st.slider("Petal Width (cm)",
            min_value=float(df["PetalWidthCm"].min()),
            max_value=float(df["PetalWidthCm"].max()),
            value=float(round(df["PetalWidthCm"].mean(), 1)),
            step=0.1
        )

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🔍 Predict Species", use_container_width=True):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Map species to CSS class
    species_class = {
        "Iris-setosa": "setosa",
        "Iris-versicolor": "versicolor",
        "Iris-virginica": "virginica"
    }.get(prediction[0], "")

    # Result card
    st.markdown(f"""
        <div class="result-card {species_class}">
            <h2>🌼 Predicted Species: <b>{prediction[0]}</b></h2>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Prediction Probability")
    for flower, prob in zip(model.classes_, probability[0]):
        st.progress(float(prob))
        st.write(f"**{flower} : {prob*100:.2f}%**")
