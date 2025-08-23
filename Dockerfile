FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (ffmpeg etc.)
RUN apt-get update && apt-get install -y ffmpeg wget tar

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8080

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
