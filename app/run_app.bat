@echo off
REM Demo script to run the Streamlit app on Windows

echo 🛰️ Starting Satellite Image Classifier...
echo 📂 Make sure you have run the training notebook first!
echo.

REM Check if models exist
if not exist "..\models" (
    echo ❌ Models directory not found. Please run the training notebook first.
    pause
    exit /b 1
)

REM Run the Streamlit app
echo 🚀 Launching the app...
streamlit run app.py --server.port 8501 --server.address localhost

echo ✅ App is running at http://localhost:8501
pause