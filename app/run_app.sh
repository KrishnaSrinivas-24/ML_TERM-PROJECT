#!/bin/bash
# Demo script to run the Streamlit app

echo "🛰️ Starting Satellite Image Classifier..."
echo "📂 Make sure you have run the training notebook first!"
echo ""

# Check if models exist
if [ ! -d "../models" ]; then
    echo "❌ Models directory not found. Please run the training notebook first."
    exit 1
fi

# Run the Streamlit app
echo "🚀 Launching the app..."
streamlit run app.py --server.port 8501 --server.address localhost

echo "✅ App is running at http://localhost:8501"