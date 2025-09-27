import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import os
import tensorflow as tf

# --- Configuration ---
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Helper Functions ---
def preprocess_image(image):
    """
    Takes a user-uploaded image, processes it to match the training data format.
    """
    # Convert image to a NumPy array
    img_array = np.array(image)
    
    # Ensure image is 3-channel RGB
    if len(img_array.shape) == 2: # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4: # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Resize and Normalize
    resized_img = cv2.resize(img_array, (64, 64))
    normalized_img = resized_img.astype('float32') / 255.0

    # Calculate color histograms for each channel
    hist_r = cv2.calcHist([normalized_img], [0], None, [32], [0, 1])
    hist_g = cv2.calcHist([normalized_img], [1], None, [32], [0, 1])
    hist_b = cv2.calcHist([normalized_img], [2], None, [32], [0, 1])
    
    # Concatenate histograms to create the final feature vector
    feature_vector = np.concatenate((hist_r, hist_g, hist_b)).flatten()
    
    # Return as a 2D array for the model
    return feature_vector.reshape(1, -1)

# --- Load All Models and Resources ---
@st.cache_resource
def load_resources():
    """
    Loads the PCA transformer, all five models, and class names.
    This function is cached to prevent reloading on every interaction.
    """
    # Define paths relative to this script file
    base_path = os.path.dirname(__file__)
    models_path = os.path.join(base_path, '..', 'models')
    results_path = os.path.join(base_path, '..', 'results')

    # Load PCA object
    pca_path = os.path.join(models_path, 'pca.joblib')
    pca = joblib.load(pca_path)
    
    # Load class names
    class_names_path = os.path.join(results_path, 'class_names.npy')
    class_names = np.load(class_names_path, allow_pickle=True)

    # Define model names and their corresponding filenames
    models = {}
    model_files = {
        "Random Forest": "random_forest.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "Neural Network (MLP)": "neural_network_mlp.h5"
    }

    # Loop through and load each model
    for name, file in model_files.items():
        path = os.path.join(models_path, file)
        if os.path.exists(path):
            if file.endswith('.h5'):
                # Load Keras/TensorFlow model
                models[name] = tf.keras.models.load_model(path)
            else:
                # Load scikit-learn model
                models[name] = joblib.load(path)
    
    return pca, models, class_names

# --- Main App Logic ---
try:
    # Load all necessary files when the app starts
    pca, models, class_names = load_resources()
except FileNotFoundError as e:
    st.error(f"Error loading resources: {e}. Please ensure you have run the `model_training.ipynb` notebook and all required files are in their correct directories (`/models` and `/results`).")
    st.stop()


# --- Streamlit App UI ---
st.title("🛰️ Satellite Image Land Use Classifier")
st.markdown("""
Welcome! Upload a satellite image and select a machine learning model. The app will predict the image's land use category.
""")

uploaded_file = st.file_uploader(
    "Choose a satellite image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and models:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    
    # Create a dropdown menu for the user to select a model
    model_choice = st.selectbox(
        "Choose a model to make the prediction:",
        # Use the keys from our loaded models dictionary
        list(models.keys())
    )

    # Create a button to trigger the classification
    if st.button(f"Classify with {model_choice}"):
        selected_model = models[model_choice]
        
        with st.spinner(f'Analyzing the image with {model_choice}...'):
            try:
                # Process the user's image through the exact same pipeline
                features = preprocess_image(image)
                features_pca = pca.transform(features)
                
                # Make a prediction based on the model type
                if isinstance(selected_model, tf.keras.Model):
                    # Keras model predicts probabilities for all classes
                    prediction_probs = selected_model.predict(features_pca)
                    prediction_index = np.argmax(prediction_probs)
                else:
                    # Scikit-learn model predicts a single class index
                    prediction_index = selected_model.predict(features_pca)[0]

                # Get the human-readable class name
                prediction_class = class_names[prediction_index]
                
                # Display the result
                st.success(f"**Prediction from {model_choice}: {prediction_class}**")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

elif not models:
    st.error("No trained models were found. Please run the `model_training.ipynb` notebook and ensure all model files are in the `models` directory.")

