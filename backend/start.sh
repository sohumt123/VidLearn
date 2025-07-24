#!/bin/bash

echo "Starting AI Video Tutor Backend..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export OPENAI_API_KEY="${OPENAI_API_KEY:-your_openai_api_key_here}"

# Start the FastAPI server
python main.py