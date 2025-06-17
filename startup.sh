#!/bin/bash

echo "🔁 Installing Python dependencies..."
pip install --target=".python_packages/lib/site-packages" -r requirements.txt

echo "🔁 Exporting Python path..."
export PYTHONPATH="$PYTHONPATH:.python_packages/lib/site-packages"

echo "🚀 Starting FastAPI app with Uvicorn..."
exec python -m uvicorn main:app --host=0.0.0.0 --port=8000
