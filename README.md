Satellite Image Land Use Classification
This project is a comprehensive analysis of five different machine learning models for the task of land use classification from satellite imagery. The best-performing model is deployed in a user-friendly web application built with Streamlit.

Project Structure
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
│   └── confusion_matrix.png  # Confusion matrix of the champion
└── README.md

Setup & Installation
Clone the repository:

git clone [your-repo-link]
cd [your-repo-name]

Create and activate a Python virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required libraries:

pip install -r requirements.txt

(You will need to create a requirements.txt file using pip freeze > requirements.txt)

How to Run
Download the Data:

Download the EuroSAT dataset from here.

Unzip the contents into the data/ folder.

Run the Analysis (Optional):

To see how the models were trained, you can run the Jupyter Notebook:

jupyter notebook notebooks/main_pipeline.ipynb

Run the Streamlit Web Application:

Ensure your terminal is in the root directory of the project.

Run the following command:

streamlit run app/app.py

The application will open in your web browser. Upload a satellite image to get a classification.