import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math
import os
import uuid
import openai

# Initialize models
model = YOLO("runs/detect/golfclub_yolov8/weights/best.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3)

client = openai.OpenAI(api_key=("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"))


def smooth_trail(trail, window=5):
    smoothed = []
    for i in range(len(trail)):
        start = max(0, i - window + 1)
        chunk = trail[start:i + 1]
        avg_x = int(np.mean([p[0] for p in chunk]))
        avg_y = int(np.mean([p[1] for p in chunk]))
        smoothed.append((avg_x, avg_y))
    return smoothed


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return [], 0, 0, 0, [], {}

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    club_trail = []
    processed_frames = []
    club_positions = []
    hip_positions = []
    frame_idx = 0

    while ret:
        output = np.zeros_like(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        hips = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            for lm in landmarks:
                cx, cy = int(lm.x * width), int(lm.y * height)
                conf = lm.visibility
                brightness = int(conf * 255)
                cv2.circle(output, (cx, cy), 4, (brightness, brightness, brightness), -1)

            for idx1, idx2 in mp_pose.POSE_CONNECTIONS:
                lm1 = landmarks[idx1]
                lm2 = landmarks[idx2]
                x1, y1 = int(lm1.x * width), int(lm1.y * height)
                x2, y2 = int(lm2.x * width), int(lm2.y * height)
                avg_conf = (lm1.visibility + lm2.visibility) / 2
                brightness = int(avg_conf * 255)
                cv2.line(output, (x1, y1), (x2, y2), (brightness, brightness, brightness), 2)

            try:
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hips = ((l_hip.x + r_hip.x) / 2 * width, (l_hip.y + r_hip.y) / 2 * height)
                hip_positions.append((frame_idx, hips))
            except:
                hip_positions.append((frame_idx, None))
        else:
            hip_positions.append((frame_idx, None))

        results = model(frame, verbose=False)
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            club_trail.append((cx, cy))
            club_positions.append((frame_idx, (cx, cy)))
            break

        smoothed_trail = smooth_trail(club_trail, window=5)
        for i in range(1, len(smoothed_trail)):
            cv2.line(output, smoothed_trail[i - 1], smoothed_trail[i], (0, 0, 255), 2)

        processed_frames.append(output)
        frame_idx += 1
        ret, frame = cap.read()

    cap.release()
    smoothed_trail = smooth_trail(club_trail, window=5)

    max_y = max([y for x, y in smoothed_trail]) if smoothed_trail else 0
    min_y = min([y for x, y in smoothed_trail]) if smoothed_trail else 0
    backswing_height = "high" if min_y < height * 0.3 else "medium" if min_y < height * 0.5 else "low"

    total_dist = sum(euclidean(smoothed_trail[i], smoothed_trail[i + 1]) for i in range(len(smoothed_trail) - 1))
    x_vals = [p[0] for p in smoothed_trail]
    y_vals = [p[1] for p in smoothed_trail]
    horizontal_range = max(x_vals) - min(x_vals) if x_vals else 1
    vertical_range = max(y_vals) - min(y_vals) if y_vals else 1
    steepness_ratio = vertical_range / horizontal_range if horizontal_range else 1
    swing_plane = "steep" if steepness_ratio > 1.3 else "medium" if steepness_ratio > 0.8 else "flat"

    angle_changes = []
    for i in range(2, len(smoothed_trail)):
        a, b, c = np.array(smoothed_trail[i - 2]), np.array(smoothed_trail[i - 1]), np.array(smoothed_trail[i])
        ba, bc = a - b, c - b
        if np.linalg.norm(ba) > 0 and np.linalg.norm(bc) > 0:
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_changes.append(angle)
    avg_curve = np.mean(angle_changes) if angle_changes else 0
    trail_curvature = "smooth arc" if avg_curve < 0.3 else "moderate" if avg_curve < 0.7 else "jerky"

    impact_frame = -1
    min_dist = float("inf")
    for (f_idx, cpos), (_, hpos) in zip(club_positions, hip_positions):
        if hpos:
            dist = euclidean(cpos, hpos)
            if dist < min_dist:
                min_dist = dist
                impact_frame = f_idx

    analysis = {
        "backswing_height": backswing_height,
        "swing_plane": swing_plane,
        "trail_curvature": trail_curvature,
        "impact_frame": impact_frame,
        "trail_distance": total_dist,
        "tempo_ratio": impact_frame / len(processed_frames) if impact_frame > 0 else 1.0
    }

    return processed_frames, width, height, fps, smoothed_trail, analysis


def compute_similarity(trail1, trail2, width1, height1, width2, height2):
    n = min(len(trail1), len(trail2))
    if n == 0:
        return 0.0
    max_dim = max(width1, height1, width2, height2)
    norm1 = np.array([[x / max_dim, y / max_dim] for x, y in trail1[:n]])
    norm2 = np.array([[x / max_dim, y / max_dim] for x, y in trail2[:n]])
    dists = np.linalg.norm(norm1 - norm2, axis=1)
    avg_dist = np.mean(dists)
    similarity = max(0.0, 1.0 - avg_dist)
    return similarity


def run_analysis(user_path, pro_path=None, pro_cached=None):
    frames1, width1, height1, fps1, trail1, analysis1 = process_video(user_path)
    if pro_cached:
        frames2, width2, height2, fps2, trail2, analysis2 = pro_cached
    else:
        frames2, width2, height2, fps2, trail2, analysis2 = process_video(pro_path)

    out_width = width1 + width2
    out_height = max(height1, height2)
    fps = min(fps1, fps2) if fps1 and fps2 else 30
    out_filename = f"outputs/{uuid.uuid4()}_side_by_side.mp4"
    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_filename, fourcc, fps, (out_width, out_height))

    num_frames = min(len(frames1), len(frames2))
    for i in range(num_frames):
        out1 = cv2.resize(frames1[i], (width1, out_height))
        out2 = cv2.resize(frames2[i], (width2, out_height))
        combined = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        combined[:, :width1] = out1
        combined[:, width1:] = out2
        out.write(combined)

    out.release()

    similarity = compute_similarity(trail1, trail2, width1, height1, width2, height2)

    prompt = f"""
    Compare two golf swings based on the following data:

    Player 1:
    - Backswing height: {analysis1['backswing_height']}
    - Swing plane: {analysis1['swing_plane']}
    - Clubhead trail curvature: {analysis1['trail_curvature']}
    - Impact frame: {analysis1['impact_frame']}
    - Trail distance: {analysis1['trail_distance']:.1f}
    - Tempo (impact ratio): {analysis1['tempo_ratio']:.2f}


    Player 2:
    - Backswing height: {analysis2['backswing_height']}
    - Swing plane: {analysis2['swing_plane']}
    - Clubhead trail curvature: {analysis2['trail_curvature']}
    - Impact frame: {analysis2['impact_frame']}
    - Trail distance: {analysis2['trail_distance']:.1f}
    - Tempo (impact ratio): {analysis2['tempo_ratio']:.2f}

    Similarity Score: {similarity:.2f}

    Write a short, coaching-style analysis, giving specific feedback on how the second video compares to the first video. Assume first is a pro named Collin Morikawa.
    Also, please note that both videos are in slo mo. Additionally, the user only gets this message and a similarity score.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        feedback = response.choices[0].message.content
    except Exception as e:
        feedback = f"GPT Error: {e}"

    final_output = out_filename.replace(".mp4", "_final.mp4")
    exit_code = os.system(
        f"ffmpeg -y -i {out_filename} -vcodec libx264 -crf 23 -preset fast {final_output}"
    )

    if exit_code != 0:
        print("❌ ffmpeg failed.")
        print(f"Tried converting: {out_filename} ➝ {final_output}")
        print("⚠️ Skipping deletion for debugging.")
        final_output = out_filename  # fallback to uncompressed file
    else:
        os.remove(out_filename)

    return final_output, feedback, similarity