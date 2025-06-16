#!/bin/bash

# Install Python dependencies at runtime (in case Azure didn't preserve them)
pip install -r requirements.txt

# Run your FastAPI app with uvicorn
exec uvicorn main:app --host=0.0.0.0 --port=8000
