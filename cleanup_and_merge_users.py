"""
cleanup_and_merge_users.py - Merges duplicate Vishwa profiles and cleans empty users
"""
import os
import shutil
import sqlite3
import database
import face_recognition_module as fm

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")

def run_cleanup():
    print("Starting cleanup and merge...")
    conn = database.get_connection()
    cur = conn.cursor()

    # 1. Merge User 15 files into User 13
    u13_dir = os.path.join(FACES_DIR, "13")
    u15_dir = os.path.join(FACES_DIR, "15")

    os.makedirs(u13_dir, exist_ok=True)
    if os.path.exists(u15_dir):
        existing_13 = [f for f in os.listdir(u13_dir) if f.endswith(".jpg")]
        next_idx = len(existing_13) + 1

        for fname in sorted(os.listdir(u15_dir)):
            if fname.endswith(".jpg"):
                src = os.path.join(u15_dir, fname)
                dst = os.path.join(u13_dir, f"sample_{next_idx}.jpg")
                shutil.copy2(src, dst)
                print(f"Copied {src} -> {dst}")
                next_idx += 1

        shutil.rmtree(u15_dir, ignore_errors=True)
        print("Removed user 15 samples folder.")

    # 2. Update DB encodings for 15 -> 13
    cur.execute("UPDATE face_encodings SET user_id = 13 WHERE user_id = 15")
    
    # 3. Update sample count for User 13
    total_samples = len([f for f in os.listdir(u13_dir) if f.endswith(".jpg")])
    cur.execute("INSERT OR REPLACE INTO face_labels (user_id, sample_count) VALUES (13, ?)", (total_samples,))

    # 4. Remove child table rows for user 15 first to respect FK
    cur.execute("DELETE FROM face_labels WHERE user_id = 15")
    cur.execute("DELETE FROM preferences WHERE user_id = 15")
    cur.execute("DELETE FROM interactions WHERE user_id = 15")
    cur.execute("DELETE FROM face_encodings WHERE user_id = 15")
    cur.execute("DELETE FROM users WHERE user_id = 15")

    # Clean up empty test users without samples (keep active_uids)
    active_uids = [13, 17, 18]
    placeholders = ",".join(map(str, active_uids))
    cur.execute(f"DELETE FROM face_labels WHERE user_id NOT IN ({placeholders})")
    cur.execute(f"DELETE FROM preferences WHERE user_id NOT IN ({placeholders})")
    cur.execute(f"DELETE FROM interactions WHERE user_id NOT IN ({placeholders})")
    cur.execute(f"DELETE FROM face_encodings WHERE user_id NOT IN ({placeholders})")
    cur.execute(f"DELETE FROM users WHERE user_id NOT IN ({placeholders})")

    conn.commit()
    conn.close()

    print(f"Database cleaned. Remaining active users: {active_uids}")
    print(f"User 13 sample count on disk: {total_samples}")

if __name__ == "__main__":
    run_cleanup()
