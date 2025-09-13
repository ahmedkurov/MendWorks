#!/bin/bash

# Start the Python backend for EEG prediction
echo "🚀 Starting EEG Prediction Backend..."

# Navigate to backend directory
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Start the Flask app
echo "🌐 Starting Flask server on http://localhost:5000"
python app.py
