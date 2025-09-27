# Satellite Image Land Use Classification

This project provides a comprehensive analysis of five different machine learning models for land use classification from satellite imagery. The best-performing model is deployed in a user-friendly web application built with Streamlit.

## 📂 Project Structure

```
.
├── app
│   └── app.py              # The Streamlit application script
├── data
│   └── EuroSAT_RGB         # The unzipped dataset folder
├── models
│   ├── champion_model.joblib # The saved best-performing model
│   └── pca.joblib            # The saved PCA transformer
├── notebooks
│   └── main_pipeline.ipynb   # Jupyter Notebook with all analysis
├── results
│   ├── model_performance.csv # CSV with model scores
│   └── confusion_matrix.png  # Confusion matrix of the champion model
└── README.md
```

## 🛠️ Setup & Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository
```bash
git clone [your-repo-link]
cd [your-repo-name]
```

### 2. Create and Activate a Virtual Environment
*   **On Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
*   **On macOS/Linux:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```
*(Note: If a `requirements.txt` file is not available, you will need to create one using `pip freeze > requirements.txt` after installing the necessary packages manually.)*

## 🚀 How to Run

### 1. Download the Data
- Download the EuroSAT dataset from a source like [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset).
- Unzip the contents into the `data/` folder, ensuring the final path is `data/EuroSAT_RGB`.

### 2. Run the Analysis (Optional)
To understand how the models were trained and evaluated, you can run the Jupyter Notebook:
```bash
jupyter notebook notebooks/main_pipeline.ipynb
```

### 3. Run the Streamlit Web Application
1.  Ensure your terminal is in the root directory of the project.
2.  Run the following command:
    ```bash
    streamlit run app/app.py
    ```
3.  The application will open in your web browser. You can then upload a satellite image to get a classification.