FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/ /app
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Render to detect
EXPOSE 10000

RUN apt-get update && apt-get install -y git && \
    pip install gdown && \
    mkdir -p /app/static && \
    gdown https://drive.google.com/uc?id=1Rk-YjH3whBHa4i-OGOaj2TWIV59gTgfK -O /app/static/pro_cached.pkl

# Use Gunicorn for production server
CMD bash -c "gunicorn app:app --bind 0.0.0.0:${PORT:-10000}"
