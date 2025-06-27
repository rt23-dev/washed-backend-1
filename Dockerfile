# Use official Python image
FROM python:3.10-slim

# Install dependencies (e.g., ffmpeg for video, libGL for OpenCV)
RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy everything from the backend folder to /app
COPY backend/ /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Start your Flask app
CMD ["python", "app.py"]
