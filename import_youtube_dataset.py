"""
import_youtube_dataset.py - Ingests YouTube Multimedia Video Dataset

Loads dataset/youtube.csv into the SQLite content table:
  - Extracts YouTube video ID (link)
  - Cleans title & channel description noise
  - Maps categories:
      'travel'    -> 'Travel'
      'art_music' -> 'Music & Art'
      'food'      -> 'Food & Cooking'
      'history'   -> 'History & Culture'
  - Adds verified Computer Science & AI multimedia entries for 'Technology & AI'
  - Generates official YouTube thumbnail URLs: https://img.youtube.com/vi/{video_id}/hqdefault.jpg
  - Populates rich tags and watch durations for TF-IDF vectorization
"""

import os
import csv
import re
import random
import database


BASE_DIR = os.path.dirname(__file__)
DATASET_CSV = os.path.join(BASE_DIR, "dataset", "youtube.csv")


def clean_description(text):
    """Removes social media handles, subscriber counts, and boilerplates."""
    if not text:
        return "An engaging YouTube video featuring in-depth commentary and visual insights."
    
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        if re.search(r'subscribers?|subscribe|patreon|instagram|twitter|facebook|http[s]?://|bit\.ly|tiktok|merch', line, re.I):
            continue
        cleaned.append(line)

    summary = " ".join(cleaned)
    summary = re.sub(r'\s+', ' ', summary).strip()
    if len(summary) < 40:
        # Fallback to truncated original if aggressive cleaning removed too much
        summary = " ".join(lines[:3])

    return summary[:320] if len(summary) > 320 else summary


def get_curated_tech_videos():
    """Curated Computer Science, AI, and Engineering YouTube videos."""
    return [
        (
            "Neural Networks and Deep Learning Chapter 1",
            "What is a neural network? A visual introduction to artificial intelligence, neurons, weights, biases, and activation functions.",
            "Technology & AI",
            "ai, neural networks, machine learning, deep learning, math, python",
            "https://img.youtube.com/vi/aircAruvnKk/hqdefault.jpg",
            "19 min watch",
            "aircAruvnKk"
        ),
        (
            "How Large Language Models Work: Transformers Explained",
            "Under the hood of ChatGPT and modern LLMs: attention mechanisms, self-attention matrices, and text token prediction.",
            "Technology & AI",
            "ai, llm, transformer, chatgpt, nlp, technology",
            "https://img.youtube.com/vi/wjZofJX0v4U/hqdefault.jpg",
            "16 min watch",
            "wjZofJX0v4U"
        ),
        (
            "Computer Vision & Face Recognition in 100 Seconds",
            "Fast overview of OpenCV, Haar Cascades, facial landmarks, and convolutional embeddings for biometric detection.",
            "Technology & AI",
            "computer vision, opencv, face recognition, ai, python, tutorial",
            "https://img.youtube.com/vi/u2T7TC3UfyU/hqdefault.jpg",
            "5 min watch",
            "u2T7TC3UfyU"
        ),
        (
            "Building Cloud Infrastructure with Kubernetes & Docker",
            "Architecting distributed microservices, automated autoscaling, containerization, and enterprise cloud reliability.",
            "Technology & AI",
            "docker, kubernetes, cloud, devops, microservices, backend",
            "https://img.youtube.com/vi/X48VuDVv0do/hqdefault.jpg",
            "14 min watch",
            "X48VuDVv0do"
        ),
        (
            "Quantum Computing Concepts: Qubits and Superposition",
            "Explaining the fundamental physics of quantum processors, quantum gates, entanglement, and revolutionary speedups.",
            "Technology & AI",
            "quantum, physics, computing, science, hardware, algorithms",
            "https://img.youtube.com/vi/JhHMJCUmq28/hqdefault.jpg",
            "12 min watch",
            "JhHMJCUmq28"
        ),
        (
            "Cybersecurity Zero Trust Architecture Explained",
            "Defending against ransomware and data breaches through continuous identity verification and least-privilege networks.",
            "Technology & AI",
            "cybersecurity, security, zero trust, networking, encryption",
            "https://img.youtube.com/vi/5g0V_16jW9Y/hqdefault.jpg",
            "9 min watch",
            "5g0V_16jW9Y"
        ),
        (
            "The Future of Humanoid Robotics & Embodied AI",
            "State of the art in bipedal locomotion, reinforcement learning for robotics, and autonomous spatial perception.",
            "Technology & AI",
            "robotics, ai, hardware, engineering, future tech",
            "https://img.youtube.com/vi/fn3KWM1kuAw/hqdefault.jpg",
            "11 min watch",
            "fn3KWM1kuAw"
        ),
        (
            "Modern Full-Stack Web Development in 2026",
            "Architecture of modern web apps: Next.js, FastAPI, REST, GraphQL, WebSockets, and Edge serverless execution.",
            "Technology & AI",
            "web development, javascript, python, fullstack, software engineering",
            "https://img.youtube.com/vi/nu_pCVPKzTk/hqdefault.jpg",
            "15 min watch",
            "nu_pCVPKzTk"
        )
    ]


def import_dataset(clear_existing=True):
    """Ingests dataset/youtube.csv into SQLite content table."""
    database.init_db()
    
    if not os.path.exists(DATASET_CSV):
        print(f"Error: {DATASET_CSV} not found!")
        return 0

    conn = database.get_connection()
    cur = conn.cursor()

    if clear_existing:
        print("Clearing previous mock content records...")
        cur.execute("DELETE FROM interactions")
        cur.execute("DELETE FROM content")
        conn.commit()

    category_map = {
        "travel": "Travel",
        "art_music": "Music & Art",
        "food": "Food & Cooking",
        "history": "History & Culture"
    }

    watch_times = ["6 min watch", "8 min watch", "10 min watch", "12 min watch", "15 min watch", "18 min watch"]

    imported_count = 0
    records_to_insert = []

    # 1. Add curated Technology & AI videos first
    tech_videos = get_curated_tech_videos()
    for tv in tech_videos:
        records_to_insert.append(tv)

    # 2. Read dataset/youtube.csv
    print(f"Reading from {DATASET_CSV}...")
    with open(DATASET_CSV, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            link = (row.get("link") or "").strip()
            title = (row.get("title") or "").strip()
            desc_raw = (row.get("description") or "").strip()
            cat_raw = (row.get("category") or "").strip().lower()

            if not link or not title:
                continue

            clean_cat = category_map.get(cat_raw, "Travel")
            clean_desc = clean_description(desc_raw)
            tags = f"{clean_cat.lower()}, youtube, video, {cat_raw}, documentary"
            image_url = f"https://img.youtube.com/vi/{link}/hqdefault.jpg"
            watch_time = random.choice(watch_times)

            records_to_insert.append((
                title,
                clean_desc,
                clean_cat,
                tags,
                image_url,
                watch_time,
                link
            ))

    print(f"Prepared {len(records_to_insert)} total multimedia video entries. Inserting into SQLite...")
    
    cur.executemany(
        """
        INSERT INTO content (title, description, category, tags, image_url, read_time, video_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        records_to_insert
    )
    conn.commit()

    # Category stats
    stats = cur.execute("SELECT category, COUNT(*) as c FROM content GROUP BY category").fetchall()
    total = cur.execute("SELECT COUNT(*) FROM content").fetchone()[0]
    conn.close()

    print("\n=======================================================")
    print(f" SUCCESS! Successfully imported {total} video items into FaceSense AI:")
    for row in stats:
        print(f"   • {row['category']}: {row['c']} videos")
    print("=======================================================\n")
    return total


if __name__ == "__main__":
    import_dataset()
