# Use an official Python image
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code
COPY backend /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "main.py"]
