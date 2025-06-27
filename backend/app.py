import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from two_vid_comparision import run_analysis
from pathlib import Path
import pickle
import requests
import gdown

PRO_CACHE_PATH = "static/pro_cached.pkl"
PRO_CACHE_URL = "https://drive.google.com/uc?id=1Rk-YjH3whBHa4i-OGOaj2TWIV59gTgfK"

if not os.path.exists(PRO_CACHE_PATH):
    print("📥 Downloading pro golfer cache...")
    os.makedirs("static", exist_ok=True)
    gdown.download(PRO_CACHE_URL, PRO_CACHE_PATH, quiet=False)
    print("✅ Downloaded and saved .pkl.")

print("⏳ Loading pro golfer video cache...")
with open(PRO_CACHE_PATH, "rb") as f:
    pro_cached = pickle.load(f)
print("✅ Loaded pro golfer video cache.")

app = Flask(__name__, static_url_path='/outputs', static_folder='outputs')
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    user_file = request.files["file"]
    user_filename = f"uploads/{uuid.uuid4()}_user.mp4"
    os.makedirs("uploads", exist_ok=True)
    user_file.save(user_filename)

    output_path, feedback, similarity = run_analysis(user_filename, pro_cached=pro_cached)
    return jsonify({
        "feedback": feedback,
        "videoUrl": f"http://localhost:5000/outputs/{Path(output_path).name}",
        "similarity": similarity
    })

@app.route("/")
def home():
    return "Hello from Washed backend!"

print("✅ Flask app starting...")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
