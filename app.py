"""
app.py - FaceSense AI Application Controller

Full REST API & Controller implementation for:
    1. Camera & Input Module
    2. Face Detection & Real-time Tracking Module
    3. Face Enrollment & Embedding Storage Module
    4. Face Recognition & Similarity Module
    5. User Profile & Preferences Module
    6. Content Management Module
    7. Recommendation Engine Module (TF-IDF + Cosine Similarity)
    8. Unknown User & Privacy-Safe Fallback Module
    9. Web UI Controller
    10. Admin & Data Management Module
"""

import os
from flask import Flask, request, jsonify, render_template
import database
import face_recognition_module as face_module
import recommendation

app = Flask(__name__)


# ---------------- Web UI Route ----------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------- Face Detection & Recognition API ----------------

@app.route("/api/detect", methods=["POST"])
def detect_face():
    """Quick face detection to provide real-time bounding box to the browser."""
    payload = request.get_json(force=True, silent=True) or {}
    frame = payload.get("frame")
    if not frame:
        return jsonify({"detected": False, "error": "No frame received"}), 400

    img = face_module.decode_base64_image(frame)
    if img is None:
        return jsonify({"detected": False, "error": "Decode failed"}), 400

    _, bbox, (h, w) = face_module.detect_face_bbox(img)
    if bbox:
        return jsonify({
            "detected": True,
            "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
            "frame_width": w,
            "frame_height": h
        })
    return jsonify({"detected": False, "frame_width": w, "frame_height": h})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    """
    Identifies a user from a live camera frame.
    If matched -> loads user profile & generates personalized recommendations.
    If unmatched -> loads safe general recommendations.
    """
    payload = request.get_json(force=True, silent=True) or {}
    frame = payload.get("frame")
    category_filter = payload.get("category", "All")

    if not frame:
        return jsonify({"error": "frame is required"}), 400

    result = face_module.recognize_face(frame)
    bbox_dict = None
    if result.get("bbox"):
        b = result["bbox"]
        bbox_dict = {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}

    # Check if a user is recognized
    if result.get("recognized") and result.get("user_id"):
        user_id = result["user_id"]
        user = database.get_user(user_id)
        if user:
            database.update_user_last_seen(user_id)
            recs = recommendation.get_recommendations_for_user(
                user_id, top_n=8, category_filter=category_filter
            )
            prefs = database.get_preferences(user_id)
            return jsonify({
                "face_detected": True,
                "recognized": True,
                "user": {
                    "user_id": user["user_id"],
                    "name": user["name"],
                    "email": user["email"] or "user@example.com",
                    "created_at": user["created_at"],
                    "last_seen": user.get("last_seen", "Just now"),
                    "preferences": prefs
                },
                "confidence_score": result.get("confidence_score"),
                "match_percentage": result.get("match_percentage"),
                "threshold": result.get("threshold", 76.0),
                "bbox": bbox_dict,
                "recommendations": recs
            })

    # If not recognized or no face found
    default_recs = recommendation.get_default_recommendations(
        top_n=8, category_filter=category_filter
    )
    return jsonify({
        "face_detected": result.get("face_detected", False),
        "recognized": False,
        "user": None,
        "confidence_score": result.get("confidence_score", 0),
        "match_percentage": result.get("match_percentage", 0),
        "threshold": result.get("threshold", 76.0),
        "predicted_label": result.get("predicted_label"),
        "bbox": bbox_dict,
        "recommendations": default_recs
    })


# ---------------- Enrollment Pipeline ----------------

@app.route("/api/register", methods=["POST"])
def register():
    """Initializes new user profile with selected interests."""
    payload = request.get_json(force=True, silent=True) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip()
    categories = payload.get("categories") or []

    if not name:
        return jsonify({"error": "User name is required"}), 400
    if not categories:
        return jsonify({"error": "Select at least one interest category"}), 400

    user_id = database.create_user(name, email=email)
    database.set_preferences(user_id, categories)
    return jsonify({
        "status": "success",
        "user_id": user_id,
        "name": name,
        "email": email,
        "categories": categories
    })


@app.route("/api/enroll_sample", methods=["POST"])
def enroll_sample():
    """Captures a single face crop sample and extracts numerical embedding."""
    payload = request.get_json(force=True, silent=True) or {}
    user_id = payload.get("user_id")
    frame = payload.get("frame")

    if not user_id or not frame:
        return jsonify({"error": "user_id and frame are required"}), 400

    ok, message, bbox = face_module.enroll_face_sample(user_id, frame)
    if not ok:
        return jsonify({"error": message}), 422

    database.increment_sample_count(user_id)
    bbox_dict = None
    if bbox:
        bbox_dict = {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}

    return jsonify({
        "status": "success",
        "message": message,
        "bbox": bbox_dict
    })


@app.route("/api/finish_enrollment", methods=["POST"])
def finish_enrollment():
    """Compiles all stored face samples into the LBPH model."""
    ok, message = face_module.retrain_model()
    if not ok:
        return jsonify({"error": message}), 422
    return jsonify({"status": "success", "message": message})


# ---------------- Recommendations API ----------------

@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
def recommendations(user_id):
    category = request.args.get("category", "All")
    recs = recommendation.get_recommendations_for_user(user_id, top_n=8, category_filter=category)
    return jsonify(recs)


@app.route("/api/default_recommendations", methods=["GET"])
def default_recommendations():
    category = request.args.get("category", "All")
    recs = recommendation.get_default_recommendations(top_n=8, category_filter=category)
    return jsonify(recs)


# ---------------- Interaction Logging ----------------

@app.route("/api/interact", methods=["POST"])
def interact():
    payload = request.get_json(force=True, silent=True) or {}
    user_id = payload.get("user_id")
    content_id = payload.get("content_id")
    action = payload.get("action", "click")

    if not user_id or not content_id:
        return jsonify({"error": "user_id and content_id are required"}), 400

    database.log_interaction(user_id, content_id, action)
    return jsonify({"status": "logged", "action": action})


# ---------------- User Management API ----------------

@app.route("/api/users", methods=["GET"])
def get_users():
    users = database.get_all_users()
    for u in users:
        u["preferences"] = database.get_preferences(u["user_id"])
    return jsonify(users)


@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user_profile(user_id):
    user = database.get_user(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    user["preferences"] = database.get_preferences(user_id)
    user["interactions"] = database.get_user_interactions(user_id, limit=10)
    return jsonify(user)


@app.route("/api/users/<int:user_id>/preferences", methods=["POST"])
def update_preferences(user_id):
    payload = request.get_json(force=True, silent=True) or {}
    categories = payload.get("categories") or {}
    database.set_preferences(user_id, categories)
    return jsonify({"status": "updated", "preferences": database.get_preferences(user_id)})


@app.route("/api/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    database.delete_user(user_id)
    face_module.retrain_model()
    return jsonify({"status": "deleted", "user_id": user_id})


# ---------------- Content Management API (Admin) ----------------

@app.route("/api/content", methods=["GET"])
def list_content():
    category = request.args.get("category")
    items = database.get_all_content()
    if category and category.lower() != "all":
        items = [i for i in items if i["category"].lower() == category.lower()]
    return jsonify(items)


@app.route("/api/content", methods=["POST"])
def add_content():
    payload = request.get_json(force=True, silent=True) or {}
    title = (payload.get("title") or "").strip()
    category = (payload.get("category") or "").strip()
    description = (payload.get("description") or "").strip()
    tags = (payload.get("tags") or "").strip()
    image_url = payload.get("image_url")
    read_time = payload.get("read_time", "8 min watch")
    video_id = payload.get("video_id")

    if not title or not category or not description:
        return jsonify({"error": "Title, category, and description are required"}), 400

    new_id = database.add_content(title, description, category, tags, image_url, read_time, video_id=video_id)
    recommendation.invalidate_cache()
    return jsonify({"status": "created", "content_id": new_id})


@app.route("/api/content/<int:content_id>", methods=["DELETE"])
def delete_content(content_id):
    database.delete_content(content_id)
    recommendation.invalidate_cache()
    return jsonify({"status": "deleted", "content_id": content_id})


# ---------------- Admin Analytics & System Info ----------------

@app.route("/api/stats", methods=["GET"])
def stats():
    sys_stats = database.get_system_stats()
    sys_stats["model_trained"] = os.path.exists(face_module.MODEL_PATH)
    return jsonify(sys_stats)


@app.route("/api/interactions", methods=["GET"])
def all_interactions():
    limit = int(request.args.get("limit", 30))
    logs = database.get_all_interactions(limit=limit)
    return jsonify(logs)


@app.route("/api/seed_demo_user", methods=["POST"])
def seed_demo_user():
    """
    Creates a pre-configured demo user profile
    with preset Technology & AI and multimedia interests to simplify live demonstrations.
    """
    user_id = database.create_user("Demo User", "demo.user@example.com")
    database.set_preferences(user_id, {
        "Technology & AI": 1.5,
        "Travel": 1.2,
        "Music & Art": 1.0
    })
    return jsonify({
        "status": "success",
        "user_id": user_id,
        "name": "Demo User",
        "email": "demo.user@example.com",
        "message": "Demo profile initialized. You can now enroll face samples or test recommendations!"
    })


if __name__ == "__main__":
    database.init_db()
    port = int(os.environ.get("PORT", 5000))
    print(f"FaceSense AI starting on http://127.0.0.1:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
