import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    :root {
        --primary-color: #6366f1;
        --secondary-color: #a855f7;
        --background-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    * {
        font-family: 'Outfit', sans-serif !important;
    }

    .stApp {
        background: var(--background-gradient);
        color: #f8fafc;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    .card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .prediction-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #818cf8;
    }

    .prediction-result {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
        color: #fff;
        text-shadow: 0 0 20px rgba(129, 140, 248, 0.5);
    }

    .stSlider > div > div > div > div {
        color: #818cf8;
    }

    .stButton > button {
        background: linear-gradient(to right, #6366f1, #a855f7);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
        transform: scale(1.02);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid var(--glass-border);
    }

    .stAlert {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #fca5a5;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    if not os.path.exists('model.pkl'):
        return None
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

data = load_model()

# Header
st.markdown('<h1 class="main-header">Iris classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Premium Machine Learning Experience using KNN Classifier</p>', unsafe_allow_html=True)

if data is None:
    st.error("⚠️ Model file (model.pkl) not found! Please run the training notebook first.")
    st.info("The application requires a trained model to make predictions. Follow the 'train_model.ipynb' to generate it.")
    st.stop()

model = data['model']
feature_names = data['feature_names']
target_names = data['target_names']

# Sidebar inputs
st.sidebar.title("🧬 Feature Inputs")
st.sidebar.markdown("Adjust the flower measurements below:")

inputs = {}
for i, name in enumerate(feature_names):
    # Map feature names to user-friendly titles
    title = name.replace('(cm)', '').strip().title()
    inputs[name] = st.sidebar.slider(f"{title} (cm)", 
                                     min_value=0.0, 
                                     max_value=10.0, 
                                     value=5.0, 
                                     step=0.1,
                                     key=f"input_{i}")

# Main Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="prediction-title">🌸 Input Summary</p>', unsafe_allow_html=True)
    
    # Display inputs as a nice dataframe or list
    input_df = pd.DataFrame([inputs])
    st.dataframe(input_df, hide_index=True)
    
    predict_btn = st.button("🚀 Predict Species")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="prediction-title">📊 Classification Result</p>', unsafe_allow_html=True)
    
    if predict_btn:
        prediction = model.predict(input_df.values)
        proba = model.predict_proba(input_df.values)
        class_idx = prediction[0]
        species = target_names[class_idx].capitalize()
        confidence = np.max(proba) * 100
        
        st.markdown(f'<div class="prediction-result">{species}</div>', unsafe_allow_html=True)
        st.progress(confidence / 100)
        st.write(f"Confidence: **{confidence:.1f}%**")
        
        # Display some info about the species (placeholder for richness)
        if species == "Setosa":
            st.info("Setosa flowers are known for their small petals and sepals.")
        elif species == "Versicolor":
            st.info("Versicolor is an intermediate species with colorful patterns.")
        else:
            st.info("Virginica flowers typically have larger leaves and stems.")
    else:
        st.write("Click the button to classify the iris flower based on your inputs.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #64748b; font-size: 0.8rem;">'
    'Built with Streamlit & Scikit-Learn | Premium AI Design by Antigravity'
    '</div>', 
    unsafe_allow_html=True
)
