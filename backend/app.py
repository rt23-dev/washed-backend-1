import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from two_vid_comparision import run_analysis  # Import your analysis function
from two_vid_comparision import process_video  # Import your video processing function
from pathlib import Path

pro_video_path = "static/pro.mp4"
print("⏳ Caching pro golfer video...")
pro_cached = process_video(pro_video_path)
print("✅ Cached pro golfer video.")

app = Flask(__name__, static_url_path='/outputs', static_folder='outputs')
CORS(app)  # Allow requests from frontend

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    user_file = request.files["file"]
    pro_path = "static/pro.mp4"  # Replace with your actual pro video path

    # Save uploaded file
    user_filename = f"uploads/{uuid.uuid4()}_user.mp4"
    os.makedirs("uploads", exist_ok=True)
    user_file.save(user_filename)

    # Run analysis
    output_path, feedback, similarity = run_analysis(user_filename, pro_cached=pro_cached)
    filename_only = Path(output_path).name
    return jsonify({
        "feedback": feedback,
        "videoUrl": f"http://localhost:5000/outputs/{os.path.basename(output_path)}",
        "similarity": similarity
    })
    
@app.route("/")
def home():
    return "Hello from Washed backend!"
    
#@app.route("/outputs/<filename>")
#def serve_output_video(filename):
#    return send_from_directory("outputs", filename, mimetype="video/mp4")

print("✅ Flask app starting...")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
