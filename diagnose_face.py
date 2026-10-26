"""
diagnose_face.py - Inspects the current state of face recognition
"""
import os
import cv2
import sqlite3
import numpy as np
import database
import face_recognition_module as face_module

def main():
    print("--- 1. DATABASE USERS ---")
    conn = database.get_connection()
    users = conn.execute("SELECT * FROM users").fetchall()
    for u in users:
        samples = conn.execute("SELECT sample_count FROM face_labels WHERE user_id = ?", (u["user_id"],)).fetchone()
        cnt = samples["sample_count"] if samples else 0
        print(f"User ID {u['user_id']}: {u['name']} ({u['email']}) - sample_count in DB: {cnt}")
    conn.close()

    print("\n--- 2. STORED FACE SAMPLES ON DISK ---")
    faces_dir = face_module.FACES_DIR
    if os.path.exists(faces_dir):
        for uid in os.listdir(faces_dir):
            udir = os.path.join(faces_dir, uid)
            if os.path.isdir(udir):
                files = os.listdir(udir)
                print(f"User {uid}: {len(files)} files -> {files}")
    else:
        print("No faces directory found.")

    print("\n--- 3. TESTING LBPH MODEL PREDICTION ON SAMPLES ---")
    model_path = face_module.MODEL_PATH
    if not os.path.exists(model_path):
        print("Model file does not exist!")
        return

    print("Model file exists:", model_path, f"({os.path.getsize(model_path)} bytes)")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    for uid in os.listdir(faces_dir):
        udir = os.path.join(faces_dir, uid)
        if not os.path.isdir(udir):
            continue
        for fname in os.listdir(udir):
            fpath = os.path.join(udir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                processed = face_module.preprocess_face_roi(img)
                if processed is not None:
                    pred_label, distance = recognizer.predict(processed)
                    match = f"MATCH (<={face_module.MATCH_THRESHOLD})" if distance <= face_module.MATCH_THRESHOLD else f"NO MATCH (>{face_module.MATCH_THRESHOLD})"
                    print(f"Sample {uid}/{fname}: Predicted label={pred_label}, Distance={distance:.2f} -> {match}")

if __name__ == "__main__":
    main()
