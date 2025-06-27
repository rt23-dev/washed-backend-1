FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ Copy contents of backend, not backend folder itself
COPY backend/ /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
