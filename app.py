import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load all models and scaler from a single .pkl file
data = joblib.load('models/iris_all_models.pkl')
scaler = data['scaler']
best_models = data['models']

species_names = ['setosa', 'versicolor', 'virginica']
feature_names = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

# --- FUNCTIONS ---

# Default probability chart (equal probabilities)
def default_proba_plot():
    fig, ax = plt.subplots(figsize=(5, 3))
    proba = [1/len(species_names)] * len(species_names)
    bars = ax.bar(species_names, proba, color=sns.color_palette("viridis", len(species_names)))
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
    return fig

# Prediction function
def predict_species(model_name, sepal_length, sepal_width, petal_length, petal_width):
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
    result_text = f"🌸 **Predicted Species:** {predicted_species.capitalize()}"

    # Create probability chart
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

    return result_text, fig

# Create default chart
default_chart = default_proba_plot()

# --- GRADIO APP ---

with gr.Blocks() as demo:
    gr.Markdown("""
    ## 🌺 Iris Flower Species Prediction
    Built with Scikit-learn, Matplotlib, Seaborn & Gradio — by Kuldii Project

    This app predicts the species of an iris flower based on measurements such as:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width

    🏷️ **Model Selection**: Choose from different classification models (e.g. Random Forest, Logistic Regression, etc.)  
    📊 **Visualization**: Displays prediction probabilities as a bar chart to help understand model confidence.

    Just adjust the sliders and click **Predict** to see the result!
    """)

    model_dropdown = gr.Dropdown(
        choices=list(best_models.keys()),
        value=list(best_models.keys())[0],
        label="Select Model"
    )
    
    sepal_length = gr.Slider(
        minimum=4.0, maximum=8.0, value=5.1, step=0.1,
        label=feature_names[0]
    )
    sepal_width = gr.Slider(
        minimum=2.0, maximum=4.5, value=3.5, step=0.1,
        label=feature_names[1]
    )
    petal_length = gr.Slider(
        minimum=1.0, maximum=7.0, value=1.4, step=0.1,
        label=feature_names[2]
    )
    petal_width = gr.Slider(
        minimum=0.1, maximum=2.5, value=0.2, step=0.1,
        label=feature_names[3]
    )
    
    with gr.Row():
        output_text = gr.Markdown(
            value="🌸 **Predicted Species:** -",
            label="Prediction Result"
        )
        chart = gr.Plot(
            value=default_chart,
            label="Probability Chart"
        )
    
    btn = gr.Button("🌼 Predict")
    btn.click(
        fn=predict_species,
        inputs=[
            model_dropdown,
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ],
        outputs=[output_text, chart]
    )

demo.launch()
