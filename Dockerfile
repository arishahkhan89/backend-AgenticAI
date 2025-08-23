# Base Python image
FROM python:3.13-slim

# Working directory
WORKDIR /app

# Copy Python dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install system-level dependencies (ffmpeg)
RUN apt-get update && apt-get install -y ffmpeg

# Copy project files
COPY . .

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
