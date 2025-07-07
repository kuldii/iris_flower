import os
import gdown
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# GDrive file ID for iris model (single .pkl)
MODELS_FILE_ID = "1SMbfov7ZwfOzBkU3dk4a6WfpE1e30fLO"

# Download helper
def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Download the model file if not present
if not os.path.exists("models/iris_all_models.pkl"):
    download_from_gdrive(MODELS_FILE_ID, "models/iris_all_models.pkl")

# Load model and scaler
data = joblib.load("models/iris_all_models.pkl")
scaler = data['scaler']
best_models = data['models']

species_names = ['setosa', 'versicolor', 'virginica']
feature_names = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

# --- APP LAYOUT ---
st.markdown("""
## 🌺 Iris Flower Species Prediction
Built with Scikit-learn, Matplotlib, Seaborn & Streamlit — by Kuldii Project

This app predicts the species of an iris flower based on measurements such as:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

🏷️ **Model Selection**: Choose from different classification models (e.g. Random Forest, Logistic Regression, etc.)  
📊 **Visualization**: Displays prediction probabilities as a bar chart to help understand model confidence.

Just adjust the sliders and click **Predict** to see the result!
""")

# Move model selector into main page
model_name = st.selectbox(
    "Select Model",
    list(best_models.keys()),
    index=0
)

# Sliders for input features
sepal_length = st.slider(feature_names[0], 4.0, 8.0, 5.1, 0.1)
sepal_width = st.slider(feature_names[1], 2.0, 4.5, 3.5, 0.1)
petal_length = st.slider(feature_names[2], 1.0, 7.0, 1.4, 0.1)
petal_width = st.slider(feature_names[3], 0.1, 2.5, 0.2, 0.1)

if st.button("🌼 Predict"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_features)
    model = best_models[model_name]
    pred = model.predict(input_scaled)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
    else:
        proba = np.zeros(len(species_names))
        proba[pred] = 1.0

    predicted_species = species_names[pred]
    st.markdown(f"## 🌸 Predicted Species: **{predicted_species.capitalize()}**")

    # Probability chart
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(
        species_names,
        proba,
        color=sns.color_palette("viridis", len(species_names))
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    for bar, p in zip(bars, proba):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{p:.2f}",
            ha='center',
            va='bottom',
            fontsize=12
        )
    plt.tight_layout()
    st.pyplot(fig)
