# Iris Flower Classification App

A professional, production-ready machine learning app for classifying Iris flower species using the classic scikit-learn Iris dataset. Built with robust EDA, multi-model training (Logistic Regression, SVM, Random Forest, KNN), hyperparameter tuning, and modern Gradio & Streamlit UIs. Ready for deployment and reproducible environments.

---

## 🚀 Features

- **Comprehensive EDA**: Summary statistics, distributions, pairplots, violin/swarm plots, correlation heatmap, class balance
- **Multiple Classifiers**: Logistic Regression, SVM, Random Forest, KNN (with GridSearchCV hyperparameter tuning)
- **Clean Model Export**: All models and scaler exported as a single `.pkl` file for easy deployment
- **Interactive Apps**: Gradio and Streamlit UIs for real-time prediction and probability visualization
- **Production-Ready**: Environment files for pip and conda, Google Drive model download for cloud deployment

---

## 🏗️ Project Structure

```
├── app.py                      # Gradio app for prediction (production-ready)
├── streamlit_app.py            # Streamlit app with GDrive model download
├── iris_classification.ipynb   # Full EDA, modeling, and training notebook
├── models/
│   └── iris_all_models.pkl     # Trained models and scaler (joblib)
├── requirements.txt            # Python dependencies (pip)
├── environment.yml             # Conda environment file
└── README.md                   # Project documentation
```

---

## 📊 Data & Preprocessing

- **Dataset**: [Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset)
- **Preprocessing**:
  - Train-test split (80-20)
  - Standardization of features
  - No missing values or outliers in the classic dataset

---

## 🧠 Models

- **Logistic Regression** (with GridSearchCV)
- **Support Vector Machine (SVM)** (with GridSearchCV)
- **Random Forest Classifier** (with GridSearchCV)
- **K-Nearest Neighbors (KNN)** (with GridSearchCV)

All models are trained, tuned, and saved for instant prediction in the apps.

---

## 🖥️ Gradio & Streamlit Apps

- **Sliders** for all features (custom min/max for each)
- **Model selection** dropdown
- **Prediction output**: Predicted species and probability chart
- **Streamlit**: Downloads model from Google Drive if not present (for easy cloud deployment)

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kuldii/iris_flower.git
cd iris_flower
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# or, for conda users:
conda env create -f environment.yml
conda activate iris-flower-classifier
```

### 3. (Optional) Train Models
- All models and scaler are pre-trained and saved in `models/iris_all_models.pkl`.
- To retrain, use the notebook `iris_classification.ipynb` and re-export the models.

### 4. Run the Apps

**Gradio:**
```bash
python app.py
```
- The app will be available at `http://localhost:7860/`

**Streamlit:**
```bash
streamlit run streamlit_app.py
```
- The app will be available at `http://localhost:8501/`
- The model file will be downloaded from Google Drive if not present.

---

## 🖥️ Usage

1. Open the app in your browser.
2. Input flower features (Sepal Length, Sepal Width, Petal Length, Petal Width).
3. Select a classification model.
4. Click **Predict** to get the predicted species and probability chart.

---

## 📊 Visualizations & EDA
- See `iris_classification.ipynb` for:
  - Summary statistics
  - Feature distributions
  - Pairplots, violin and swarm plots
  - Correlation heatmap
  - Class balance
  - Model evaluation (accuracy, classification report, confusion matrix)

---

## 📝 Model Details
- **Preprocessing**: StandardScaler for feature scaling.
- **Models**: LogisticRegression, SVC, RandomForestClassifier, KNeighborsClassifier (all with GridSearchCV).
- **Export**: All models and scaler saved in a single `.pkl` file for easy loading in apps.

---

## 📁 File Descriptions
- `app.py`: Gradio app, loads models, handles prediction and UI.
- `streamlit_app.py`: Streamlit app, downloads model from GDrive if needed.
- `models/iris_all_models.pkl`: Dictionary of trained classifiers and scaler.
- `requirements.txt`: Python dependencies.
- `environment.yml`: Conda environment file.
- `iris_classification.ipynb`: Full EDA, preprocessing, model training, and export.

---

## 🌐 Demo & Credits
- **Author**: Sandikha Rahardi (Kuldii Project)
- **Website**: https://kuldiiproject.com
- **Dataset**: [Scikit-learn Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset)
- **UI**: [Gradio](https://gradio.app/), [Streamlit](https://streamlit.io/)
- **ML**: [Scikit-learn](https://scikit-learn.org/)

---

For questions or contributions, please open an issue or pull request.
