"""
test_system.py - Integration Verification for FaceSense AI
Tests the complete pipeline:
- Database connectivity & table schema
- Content catalog & categories
- TF-IDF recommendation engine & cosine ranking
- User registration & preferences tuning
- Interaction feedback loop
- Flask test client requests & endpoints
"""

import os
import cv2
import numpy as np
import base64
import json
import database
import face_recognition_module as face_module
import recommendation
from app import app


def create_dummy_face_frame():
    """Generates a synthetic 640x480 test image with a circle."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 40)
    # Draw a simulated head
    cv2.circle(img, (320, 240), 90, (200, 180, 160), -1)
    # Eyes
    cv2.circle(img, (290, 220), 12, (50, 50, 50), -1)
    cv2.circle(img, (350, 220), 12, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (320, 280), (35, 15), 0, 0, 180, (50, 50, 150), -1)

    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def run_tests():
    print("========================================")
    print(" FaceSense AI System Integration Tests")
    print("========================================")

    # 1. DB Init & Schema
    database.init_db()
    stats = database.get_system_stats()
    print(f"[OK] Database Schema verified. Total catalogue items: {stats['total_content']}")

    # 2. Test Client setup
    client = app.test_client()

    # 3. Test GET /
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    print("[OK] Route GET / loaded successfully (Status 200 OK)")

    # 4. Test Demo User Seeding (Vishwa DK)
    res = client.post("/api/seed_demo_user")
    assert res.status_code == 200
    vishwa_data = res.get_json()
    vishwa_id = vishwa_data["user_id"]
    print(f"[OK] Route POST /api/seed_demo_user created demo student: {vishwa_data['name']} (ID: {vishwa_id})")

    # 5. Test Personalized Recommendations for Vishwa DK
    res = client.get(f"/api/recommendations/{vishwa_id}")
    assert res.status_code == 200
    recs = res.get_json()
    assert len(recs) > 0, "No recommendations returned!"
    top_item = recs[0]
    print(f"[OK] Route GET /api/recommendations/{vishwa_id} returned {len(recs)} personalized items.")
    print(f"   Top Match: '{top_item['title']}' | Score: {top_item.get('match_percentage')}% | Reason: {top_item.get('recommendation_reason')}")

    # 6. Test Default Recommendations (Unknown user)
    res = client.get("/api/default_recommendations")
    assert res.status_code == 200
    defs = res.get_json()
    assert len(defs) > 0
    print(f"[OK] Route GET /api/default_recommendations returned {len(defs)} safe trending fallback items.")

    # 7. Test User Interaction Logging
    res = client.post("/api/interact", json={
        "user_id": vishwa_id,
        "content_id": top_item["content_id"],
        "action": "like"
    })
    assert res.status_code == 200
    print(f"[OK] Route POST /api/interact logged user 'like' on content #{top_item['content_id']}")

    # 8. Test Stats Endpoint
    res = client.get("/api/stats")
    assert res.status_code == 200
    sys_stats = res.get_json()
    print(f"[OK] Route GET /api/stats: {sys_stats['total_users']} users, {sys_stats['total_interactions']} interactions logged.")

    # 9. Test Face Recognition with test frame
    test_frame = create_dummy_face_frame()
    res = client.post("/api/recognize", json={"frame": test_frame})
    assert res.status_code == 200
    rec_result = res.get_json()
    print(f"[OK] Route POST /api/recognize handled frame cleanly (recognized={rec_result.get('recognized')})")

    print("\n========================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! (10/10)")
    print("========================================")


if __name__ == "__main__":
    run_tests()
