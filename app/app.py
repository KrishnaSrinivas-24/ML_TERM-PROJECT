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
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
def get_custom_css(dark_mode=True):
    if dark_mode:
        return """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
    }
    .model-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #667eea;
        transition: all 0.2s ease;
    }
    .model-card:hover {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        border-color: #7c3aed;
    }
    .result-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .result-success h2 {
        margin-bottom: 0.5rem;
        font-size: 2rem;
    }
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 15px;
        padding: 0.3rem;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background: #f8f9ff;
    }
    .metric-container {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    .footer {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 3rem;
        border-top: 3px solid #667eea;
    }
    .sidebar-section {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #667eea;
    }
    .category-item {
        background: rgba(102, 126, 234, 0.1);
        color: white;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
    .mode-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 25px;
        cursor: pointer;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
</style>
"""
    else:
        return """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    .feature-card {
        background: white;
        color: #333;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.2);
    }
    .model-card {
        background: white;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #667eea;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    .model-card:hover {
        background: #f8f9ff;
        border-color: #7c3aed;
    }
    .result-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .result-success h2 {
        margin-bottom: 0.5rem;
        font-size: 2rem;
    }
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 15px;
        padding: 0.3rem;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background: #f8f9ff;
    }
    .metric-container {
        background: white;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease;
        border: 1px solid #e0e0e0;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
    }
    .footer {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 3rem;
        border-top: 3px solid #667eea;
    }
    .sidebar-section {
        background: white;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .category-item {
        background: #f8f9ff;
        color: #333;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
    .mode-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 25px;
        cursor: pointer;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
</style>
"""

st.markdown(get_custom_css(dark_mode=st.session_state.get('dark_mode', True)), unsafe_allow_html=True)

# -------------------- Helper Functions --------------------
def get_confidence_color(confidence):
    """Return color based on confidence level."""
    if confidence >= 80:
        return "#28a745"  # Green
    elif confidence >= 60:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

def create_confidence_bar(confidence):
    """Create a visual confidence bar."""
    return f"""
    <div style="background: #f0f0f0; border-radius: 10px; padding: 3px; margin: 10px 0;">
        <div style="background: {get_confidence_color(confidence)}; width: {confidence}%; 
                    height: 20px; border-radius: 7px; display: flex; align-items: center; 
                    justify-content: center; color: white; font-weight: bold;">
            {confidence:.1f}%
        </div>
    </div>
    """
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
# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Dark/Light mode toggle
col_toggle1, col_toggle2, col_toggle3 = st.columns([6, 1, 1])
with col_toggle3:
    if st.button("🌙" if st.session_state.dark_mode else "☀️", help="Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Header Section
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Land Use Classifier</h1>
    <p>AI-Powered Land Use Classification using Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Introduction Section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 High Accuracy</h3>
        <p>State-of-the-art ML models trained on satellite imagery data</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🚀 Fast Processing</h3>
        <p>Get instant predictions with optimized algorithms</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🌍 Multiple Classes</h3>
        <p>Classify various land use types from satellite images</p>
    </div>
    """, unsafe_allow_html=True)

try:
    pca, models, class_names = load_resources()
    st.success(f"✅ Successfully loaded {len(models)} models and PCA transformer")
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")
    st.info("💡 Please run the training notebook first to generate the required model files.")
    st.stop()

# Sidebar for model information
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h2 style="margin-top: 0;">🤖 Available Models</h2>
    </div>
    """, unsafe_allow_html=True)
    
    for model_name in models.keys():
        st.markdown(f"""
        <div class="model-card">
            <strong>{model_name}</strong><br>
            <small>Ready for prediction</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-section">
        <h2 style="margin-top: 0;">📊 Land Use Categories</h2>
    </div>
    """, unsafe_allow_html=True)
    
    categories = [
        "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", 
        "Industrial", "Pasture", "Permanent Crop", "Residential", 
        "River", "Sea Lake"
    ]
    
    for i, class_name in enumerate(categories):
        st.markdown(f"""
        <div class="category-item">
            <strong>{i+1}.</strong> {class_name}
        </div>
        """, unsafe_allow_html=True)

# Main content area
st.markdown("## � Upload Your Satellite Image")
uploaded_file = st.file_uploader(
    "Choose a satellite image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a satellite image in JPG, JPEG, or PNG format"
)

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, caption="Your satellite image", use_column_width=True)
        
        # Image information
        st.markdown("**Image Details:**")
        st.write(f"📏 **Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"🎨 **Mode:** {image.mode}")
        st.write(f"📁 **Format:** {image.format}")
    
    with col2:
        st.markdown("### 🔧 Model Selection")
        model_choice = st.selectbox(
            "Choose your AI model:",
            list(models.keys()),
            help="Select a machine learning model for classification"
        )
        
        # Model description
        model_descriptions = {
            "Random Forest": "🌲 Ensemble method using multiple decision trees",
            "Support Vector Machine": "⚡ High-performance classification algorithm",
            "Gradient Boosting": "📈 Advanced boosting technique for high accuracy",
            "Logistic Regression": "📊 Linear classification with probabilistic output",
            "Neural Network (MLP)": "🧠 Deep learning multi-layer perceptron"
        }
        
        st.info(f"**{model_choice}:** {model_descriptions.get(model_choice, 'Advanced ML model')}")
        
        if st.button(f"🚀 Classify with {model_choice}", use_container_width=True):
            with st.spinner(f"🔄 Analyzing image with {model_choice}..."):
                try:
                    features = preprocess_image(image)
                    features_pca = pca.transform(features)

                    model = models[model_choice]

                    if isinstance(model, tf.keras.Model):
                        prediction_probs = model.predict(features_pca, verbose=0)
                        prediction_index = np.argmax(prediction_probs)
                        confidence = float(np.max(prediction_probs)) * 100
                    else:
                        prediction_index = model.predict(features_pca)[0]
                        confidence = (
                            model.predict_proba(features_pca).max() * 100
                            if hasattr(model, "predict_proba")
                            else None
                        )

                    prediction_class = class_names[prediction_index]

                    # Results section
                    st.markdown("---")
                    st.markdown("## 🎯 Classification Results")
                    
                    # Main result card
                    st.markdown(f"""
                    <div class="result-success">
                        <h2>🏆 Prediction: {prediction_class}</h2>
                        <p>Model: {model_choice}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence:
                        st.markdown("### 📊 Confidence Level")
                        st.markdown(create_confidence_bar(confidence), unsafe_allow_html=True)
                        
                        # Confidence interpretation
                        if confidence >= 80:
                            st.success(f"🎯 **High Confidence**: The model is very confident about this prediction ({confidence:.1f}%)")
                        elif confidence >= 60:
                            st.warning(f"⚠️ **Medium Confidence**: The model has moderate confidence ({confidence:.1f}%)")
                        else:
                            st.error(f"🤔 **Low Confidence**: The model has low confidence ({confidence:.1f}%)")

                except Exception as e:
                    st.error(f"❌ **Error during prediction:** {str(e)}")
                    st.info("💡 Please try uploading a different image or contact support.")

else:
    # Welcome message when no image is uploaded
    st.markdown("### 👋 Welcome!")
    st.info("� **Get Started:** Upload a satellite image using the file uploader above to begin classification.")
    
    # Sample images section
    st.markdown("### 🌟 Sample Classifications")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🌾 Agricultural Areas</h4>
            <p>Croplands, pastures, and farming regions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with sample_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🌲 Forest Regions</h4>
            <p>Dense forests and woodland areas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with sample_col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🏘️ Urban Areas</h4>
            <p>Residential, industrial, and highway regions</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🛰️ Satellite Image Classifier</h3>
    <p><strong>Powered by Advanced Machine Learning & AI</strong></p>
    <p>Built with ❤️ using Streamlit, TensorFlow, and Scikit-learn</p>
    <br>
    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
        <div class="metric-container">
            <h4>🎯 Accuracy</h4>
            <p>High-precision models</p>
        </div>
        <div class="metric-container">
            <h4>⚡ Speed</h4>
            <p>Real-time processing</p>
        </div>
        <div class="metric-container">
            <h4>🌍 Coverage</h4>
            <p>Multiple land types</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
