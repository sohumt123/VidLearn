#!/bin/bash

echo "Starting AI Video Tutor Backend..."

# Activate virtual environment
source venv/bin/activate

# Load environment variables from .env file
# (dotenv will handle this in Python)

# Start the FastAPI server
python main.py