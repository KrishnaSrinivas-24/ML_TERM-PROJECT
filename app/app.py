import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import os
import tensorflow as tf

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="auto"
)

# -------------------- Helper Functions --------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert uploaded image into a feature vector suitable for ML models.
    Steps:
    1. Ensure RGB format
    2. Resize to 64x64
    3. Normalize
    4. Compute color histograms (R, G, B)
    5. Flatten + reshape for model
    """
    img_array = np.array(image)

    # Handle grayscale and RGBA
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Resize + Normalize
    resized_img = cv2.resize(img_array, (64, 64))
    normalized_img = resized_img.astype("float32") / 255.0

    # Compute histograms
    hist_features = []
    for channel in range(3):
        hist = cv2.calcHist([normalized_img], [channel], None, [32], [0, 1])
        hist_features.append(hist)

    feature_vector = np.concatenate(hist_features).flatten()
    return feature_vector.reshape(1, -1)


@st.cache_resource
def load_resources():
    """
    Loads PCA, trained models, and class names.
    Cached for performance.
    """
    base_path = os.path.dirname(__file__)
    models_path = os.path.join(base_path, "..", "models")
    results_path = os.path.join(base_path, "..", "results")

    # PCA
    pca_path = os.path.join(models_path, "pca.joblib")
    if not os.path.exists(pca_path):
        raise FileNotFoundError("PCA file missing. Run training notebook first.")
    pca = joblib.load(pca_path)

    # Class Names
    class_names_path = os.path.join(results_path, "class_names.npy")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError("Class names file missing. Run training notebook first.")
    class_names = np.load(class_names_path, allow_pickle=True)

    # Models
    model_files = {
        "Random Forest": "random_forest.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "Neural Network (MLP)": "neural_network_mlp.h5",
    }

    models = {}
    for name, file in model_files.items():
        path = os.path.join(models_path, file)
        if not os.path.exists(path):
            st.warning(f"⚠️ {name} model file not found. Skipping...")
            continue
        if file.endswith(".h5"):
            models[name] = tf.keras.models.load_model(path)
        else:
            models[name] = joblib.load(path)

    if not models:
        raise FileNotFoundError("No models found in the models directory.")

    return pca, models, class_names


# -------------------- Main App --------------------
st.title("🛰️ Satellite Image Land Use Classifier")
st.markdown(
    """
    Upload a **satellite image** and choose a machine learning model.  
    The app will classify the image into its land use category.
    """
)

try:
    pca, models, class_names = load_resources()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

uploaded_file = st.file_uploader("📂 Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model_choice = st.selectbox("🔍 Select a model for prediction", list(models.keys()))

    if st.button(f"🚀 Classify with {model_choice}"):
        with st.spinner(f"Analyzing with {model_choice}..."):
            try:
                features = preprocess_image(image)
                features_pca = pca.transform(features)

                model = models[model_choice]

                if isinstance(model, tf.keras.Model):
                    prediction_probs = model.predict(features_pca)
                    prediction_index = np.argmax(prediction_probs)
                    confidence = float(np.max(prediction_probs)) * 100
                else:
                    prediction_index = model.predict(features_pca)[0]
                    # For scikit-learn models, check if `predict_proba` exists
                    confidence = (
                        model.predict_proba(features_pca).max() * 100
                        if hasattr(model, "predict_proba")
                        else None
                    )

                prediction_class = class_names[prediction_index]

                st.success(f"✅ Prediction: **{prediction_class}**")
                if confidence:
                    st.info(f"Confidence: {confidence:.2f}%")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

else:
    st.info("👆 Upload an image to get started.")
