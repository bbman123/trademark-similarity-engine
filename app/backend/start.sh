#!/bin/bash

echo "========================================"
echo "  Trademark Similarity Engine"
echo "  Backend API Server"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if models exist
if [ ! -f "models/cnn_encoder.keras" ]; then
    echo "ERROR: Model files not found in models/ directory!"
    echo "Please ensure all model files are present."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Starting API server..."
echo "Server will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python api.py
