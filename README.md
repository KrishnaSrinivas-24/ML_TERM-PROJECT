# 🛰️ Satellite Image Land Use Classification

> **AI-Powered Land Use Classification using Advanced Machine Learning**

This project provides a comprehensive analysis of multiple machine learning models for land use classification from satellite imagery, featuring a stunning and professional web application built with Streamlit.

## ✨ Features

- 🎯 **High Accuracy Models**: Multiple state-of-the-art ML algorithms
- 🚀 **Real-time Processing**: Instant image classification
- 🌍 **Multiple Land Types**: Comprehensive land use category coverage
- 💻 **Professional UI**: Modern, responsive web interface
- 📊 **Visual Analytics**: Confidence scores and interactive results
- 🔧 **Easy Deployment**: One-click app launch

## 📂 Project Structure
```
.
├── app/
│   ├── app.py              # Enhanced Streamlit web application
│   ├── run_app.bat         # Windows launcher script
│   └── run_app.sh          # Unix/Linux launcher script
├── data/
│   ├── train.csv           # Training dataset metadata
│   ├── test.csv            # Test dataset metadata
│   ├── validation.csv      # Validation dataset metadata
│   ├── label_map.json      # Class label mappings
│   └── [Land Use Folders]/ # Organized satellite images
├── models/
│   ├── random_forest.joblib         # Random Forest model
│   ├── support_vector_machine.joblib # SVM model
│   ├── gradient_boosting.joblib     # Gradient Boosting model
│   ├── logistic_regression.joblib   # Logistic Regression model
│   ├── neural_network_mlp.h5        # Neural Network model
│   └── pca.joblib                   # PCA transformer
├── notebooks/
│   ├── main_pipeline.ipynb          # Complete data analysis pipeline
│   └── model_training.ipynb         # Model training and evaluation
├── results/
│   ├── model_performance.csv        # Comprehensive model comparison
│   ├── confusion_matrix.png         # Best model confusion matrix
│   └── [Model artifacts]            # Training results and metrics
└── README.md
```

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional)

### 1. Clone or Download the Repository
```bash
git clone [your-repo-link]
cd ML_TERM-PROJECT
```

### 2. Create and Activate Virtual Environment
**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start Guide

### 1. Train the Models
First, run the training notebook to generate the required model files:

```bash
jupyter notebook notebooks/model_training.ipynb
```

Run all cells in the notebook to train and save the models.

### 2. Launch the Web Application

**Option A: Use the launcher scripts (Recommended)**

Windows:
```cmd
cd app
run_app.bat
```

Unix/Linux/macOS:
```bash
cd app
chmod +x run_app.sh
./run_app.sh
```

**Option B: Manual launch**
```bash
cd app
streamlit run app.py
```

### 3. Access the Application
Open your web browser and navigate to: `http://localhost:8501`

## 🎯 How to Use the App

1. **Upload Image**: Click "Choose a satellite image..." and select your image file
2. **Select Model**: Choose from 5 available ML models in the dropdown
3. **Classify**: Click the "Classify" button to get predictions
4. **View Results**: See the predicted land use type with confidence scores

## 🤖 Available Models

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | Ensemble | Multiple decision trees for robust predictions |
| **Support Vector Machine** | Kernel-based | High-performance classification algorithm |
| **Gradient Boosting** | Ensemble | Advanced boosting for maximum accuracy |
| **Logistic Regression** | Linear | Fast linear classification with probabilities |
| **Neural Network (MLP)** | Deep Learning | Multi-layer perceptron for complex patterns |

## 📊 Supported Land Use Categories

- 🌾 **Annual Crop** - Agricultural croplands
- 🌲 **Forest** - Dense forest areas  
- 🌿 **Herbaceous Vegetation** - Grasslands and meadows
- 🛣️ **Highway** - Roads and transportation infrastructure
- 🏭 **Industrial** - Industrial and commercial areas
- 🐄 **Pasture** - Grazing lands and pastures
- 🍎 **Permanent Crop** - Orchards and permanent crops
- 🏘️ **Residential** - Urban residential areas
- 🌊 **River** - Rivers and waterways
- 🏞️ **Sea Lake** - Large water bodies

## 🛠️ Technical Details

### Architecture

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with scikit-learn and TensorFlow
- **Image Processing**: OpenCV and PIL
- **Dimensionality Reduction**: PCA transformation
- **Model Persistence**: Joblib and HDF5 formats

### Performance

- **Image Processing**: Real-time feature extraction
- **Model Inference**: Sub-second predictions
- **Memory Efficient**: Optimized model loading
- **Scalable**: Supports multiple concurrent users

## 📁 Project Workflow

1. **Data Preprocessing** (`main_pipeline.ipynb`)
   - Image loading and preprocessing
   - Feature extraction using color histograms
   - PCA dimensionality reduction

2. **Model Training** (`model_training.ipynb`)
   - Multiple algorithm comparison
   - Hyperparameter optimization
   - Model evaluation and selection

3. **Web Deployment** (`app.py`)
   - Interactive user interface
   - Real-time image classification
   - Professional styling and UX

## 🔧 Troubleshooting

### Common Issues

**Models not found error:**
```bash
# Make sure you've run the training notebook first
jupyter notebook notebooks/model_training.ipynb
```

**Port already in use:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**Module import errors:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 📈 Performance Metrics

The models are evaluated using:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1-score for multi-class classification
- **Confusion Matrix**: Detailed per-class performance
- **Confidence Scores**: Prediction certainty levels

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is available under the MIT License.

## 🙏 Acknowledgments

- **EuroSAT Dataset** - European Space Agency satellite imagery
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **TensorFlow** - Deep learning framework

---

**Made with ❤️ for Satellite Image Classification**