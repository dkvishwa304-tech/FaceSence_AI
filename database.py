"""
database.py - FaceSense AI Database Layer

Implements SQLite persistence matching the project specifications:
    - Users: user_id, name, email/identifier, created_at, last_seen
    - Face_Encodings: encoding_id, user_id, encoding, sample_path, created_at
    - Preferences: preference_id, user_id, category, preference_value
    - Content: content_id, title, description, category, tags, image_url, read_time
    - Interactions: interaction_id, user_id, content_id, action, timestamp
"""

import sqlite3
import os
import json
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "facesense.db")


def get_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            created_at TEXT NOT NULL,
            last_seen TEXT
        );

        CREATE TABLE IF NOT EXISTS face_encodings (
            encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            encoding TEXT NOT NULL,
            sample_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS face_labels (
            user_id INTEGER PRIMARY KEY,
            sample_count INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS preferences (
            preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            preference_value REAL NOT NULL DEFAULT 1.0,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS content (
            content_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            tags TEXT NOT NULL,
            image_url TEXT,
            read_time TEXT DEFAULT '5 min read'
        );

        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()

    # Check if users has new columns like last_seen, if not add them safely
    cur.execute("PRAGMA table_info(users)")
    user_columns = [row["name"] for row in cur.fetchall()]
    if "last_seen" not in user_columns:
        cur.execute("ALTER TABLE users ADD COLUMN last_seen TEXT")
    conn.commit()

    # Check if content has new columns like image_url, read_time, video_id, if not add them safely
    cur.execute("PRAGMA table_info(content)")
    columns = [row["name"] for row in cur.fetchall()]
    if "image_url" not in columns:
        cur.execute("ALTER TABLE content ADD COLUMN image_url TEXT")
    if "read_time" not in columns:
        cur.execute("ALTER TABLE content ADD COLUMN read_time TEXT DEFAULT '5 min read'")
    if "video_id" not in columns:
        cur.execute("ALTER TABLE content ADD COLUMN video_id TEXT")
    conn.commit()

    # Re-seed if content catalog is empty
    cur.execute("SELECT COUNT(*) AS c FROM content")
    count = cur.fetchone()["c"]
    if count == 0:
        seed_content(conn)

    conn.close()


def seed_content(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM content")  # Refresh with rich catalog

    rich_content = [
        # Technology & AI
        (
            "Deep Learning & Neural Networks: From Zero to Hero",
            "Comprehensive guide to modern deep neural architectures, backpropagation, and CNNs for computer vision.",
            "Technology",
            "ai, deep learning, neural networks, machine learning, python",
            "https://images.unsplash.com/photo-1677442136019-21780efad99a?w=600&auto=format&fit=crop&q=80",
            "8 min read"
        ),
        (
            "Building Cloud-Native Microservices with Docker & Kubernetes",
            "Master container orchestration, automated scaling, service mesh, and resilient distributed architectures.",
            "Technology",
            "cloud, docker, kubernetes, microservices, devops",
            "https://images.unsplash.com/photo-1667372393119-3d4c48d07fc9?w=600&auto=format&fit=crop&q=80",
            "10 min read"
        ),
        (
            "Computer Vision in 2026: Real-Time Face Recognition",
            "How face embeddings, Haar cascades, and deep facial feature representations power modern biometric security.",
            "Technology",
            "computer vision, opencv, face recognition, biometric, ai",
            "https://images.unsplash.com/photo-1507146426996-ef05306b995a?w=600&auto=format&fit=crop&q=80",
            "6 min read"
        ),
        (
            "Cybersecurity Essentials: Zero Trust Network Defense",
            "Protecting organizational assets using least-privilege access, identity-first auth, and end-to-end encryption.",
            "Technology",
            "cybersecurity, zero trust, network, privacy, encryption",
            "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=600&auto=format&fit=crop&q=80",
            "7 min read"
        ),
        (
            "Next-Generation Web Architectures: Edge Computing & WebAssembly",
            "Speed up web applications by deploying code closer to users at the network edge with WASM binaries.",
            "Technology",
            "web development, wasm, edge computing, javascript, performance",
            "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=600&auto=format&fit=crop&q=80",
            "5 min read"
        ),

        # Science & Innovation
        (
            "Quantum Computing: Principles and Practical Algorithms",
            "Understanding qubits, superposition, quantum entanglement, and Shor's algorithm for quantum supremacy.",
            "Science",
            "quantum, physics, computing, science, innovation",
            "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=600&auto=format&fit=crop&q=80",
            "9 min read"
        ),
        (
            "James Webb Space Telescope: Unveiling the Early Universe",
            "Breathtaking deep-field imagery and spectroscopic data revealing the formation of first-generation galaxies.",
            "Science",
            "astronomy, space, jwst, cosmos, nasa",
            "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=600&auto=format&fit=crop&q=80",
            "7 min read"
        ),
        (
            "CRISPR & Genetic Engineering: The Future of Precision Medicine",
            "How targeted gene-editing technology is curing hereditary illnesses and accelerating biotech innovations.",
            "Science",
            "biology, crispr, genetics, medicine, biotechnology",
            "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=600&auto=format&fit=crop&q=80",
            "6 min read"
        ),
        (
            "Renewable Energy Breakthroughs: Perovskite Solar Cells",
            "Next-generation photovoltaic cells delivering high conversion efficiencies at ultra-low production costs.",
            "Science",
            "energy, clean tech, solar, environment, climate",
            "https://images.unsplash.com/photo-1509391365360-2e959784a276?w=600&auto=format&fit=crop&q=80",
            "5 min read"
        ),

        # Sports & Fitness
        (
            "High-Performance Athletic Conditioning: Science of VO2 Max",
            "Optimize endurance, anaerobic threshold, and cardiovascular longevity through structured periodization.",
            "Sports",
            "fitness, conditioning, cardio, training, endurance",
            "https://images.unsplash.com/photo-1517838277536-f5f99be501cd?w=600&auto=format&fit=crop&q=80",
            "6 min read"
        ),
        (
            "Tactical Analysis: Modern Football Pressing Systems",
            "In-depth breakdown of Gegenpressing, mid-block transitions, and defensive shape in top European leagues.",
            "Sports",
            "football, soccer, tactics, analysis, sports",
            "https://images.unsplash.com/photo-1508098682722-e99c43a406b2?w=600&auto=format&fit=crop&q=80",
            "8 min read"
        ),
        (
            "Marathon Training: 16-Week Sub-3-Hour Roadmap",
            "Nutritional fueling strategies, weekly mileage building, and taper schedules for competitive runners.",
            "Sports",
            "running, marathon, athletics, pace, fitness",
            "https://images.unsplash.com/photo-1452626038306-9aae5e071dd3?w=600&auto=format&fit=crop&q=80",
            "9 min read"
        ),
        (
            "Basketball Shot Mechanics & Spatial Shooting Analytics",
            "Biomechanics of the modern perimeter shot and shot-selection heatmaps in modern analytics-driven basketball.",
            "Sports",
            "basketball, shooting, analytics, nba, mechanics",
            "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=600&auto=format&fit=crop&q=80",
            "5 min read"
        ),

        # Cooking & Gastronomy
        (
            "Artisanal Sourdough Masterclass: Fermentation Science",
            "Understanding lactobacillus, wild yeast hydration percentages, and Dutch-oven baking techniques.",
            "Cooking",
            "sourdough, baking, bread, fermentation, culinary",
            "https://images.unsplash.com/photo-1589367920969-ab8e050bbb04?w=600&auto=format&fit=crop&q=80",
            "7 min read"
        ),
        (
            "Gourmet Handmade Pasta: From Semolina to Table",
            "Traditional Italian pasta dough formulas, rolling precision, and authentic regional ragu pairings.",
            "Cooking",
            "pasta, italian, gourmet, culinary, recipe",
            "https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=600&auto=format&fit=crop&q=80",
            "6 min read"
        ),
        (
            "The Science of Flavour: Maillard Reaction & Umami",
            "How molecular gastronomy and thermal browning create deep savory profiles across world cuisines.",
            "Cooking",
            "food science, flavor, umami, cooking techniques, culinary",
            "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600&auto=format&fit=crop&q=80",
            "5 min read"
        ),
        (
            "Quick Nutrient-Dense Mediterranean Weeknight Meals",
            "Healthy olive oil, legume, fresh herbs, and seafood recipes ready in under 25 minutes.",
            "Cooking",
            "healthy, mediterranean, nutrition, recipes, quick meals",
            "https://images.unsplash.com/photo-1540420773420-3366772f4999?w=600&auto=format&fit=crop&q=80",
            "4 min read"
        ),

        # Music & Audio
        (
            "Music Theory for Producers: Modern Chord Progressions",
            "Harmonic structures, modal interchange, and voicing strategies to compose emotionally captivating tracks.",
            "Music",
            "music theory, production, harmony, composition, sound",
            "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=600&auto=format&fit=crop&q=80",
            "7 min read"
        ),
        (
            "The Golden Era of Classic Rock & Electric Guitar Tone",
            "Legendary tube amplifiers, analog effects pedals, and guitar riffs that defined 70s stadium rock.",
            "Music",
            "rock, guitar, vintage audio, instruments, classic rock",
            "https://images.unsplash.com/photo-1465847899084-d164df4dedc6?w=600&auto=format&fit=crop&q=80",
            "6 min read"
        ),
        (
            "Acoustic Engineering & Studio Sound Design",
            "Controlling early reflections, bass traps, and diffuse reverberation in professional audio listening rooms.",
            "Music",
            "audio engineering, acoustics, studio, sound, recording",
            "https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?w=600&auto=format&fit=crop&q=80",
            "8 min read"
        ),
        (
            "The Evolution of Jazz Improvisation and Modal Bebop",
            "Trace the harmonic genius of Miles Davis, John Coltrane, and the revolution of modal improvisation.",
            "Music",
            "jazz, history, improvisation, brass, bebop",
            "https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=600&auto=format&fit=crop&q=80",
            "7 min read"
        )
    ]

    cur.executemany(
        """
        INSERT INTO content (title, description, category, tags, image_url, read_time)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rich_content,
    )
    conn.commit()


# ---------- Users ----------

def create_user(name, email=None):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO users (name, email, created_at, last_seen) VALUES (?, ?, ?, ?)",
        (name, email, now, now),
    )
    user_id = cur.lastrowid
    cur.execute("INSERT OR REPLACE INTO face_labels (user_id, sample_count) VALUES (?, 0)", (user_id,))
    conn.commit()
    conn.close()
    return user_id


def get_user(user_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users():
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT u.user_id, u.name, u.email, u.created_at, u.last_seen,
               COALESCE(fl.sample_count, 0) as sample_count
        FROM users u
        LEFT JOIN face_labels fl ON u.user_id = fl.user_id
        ORDER BY u.user_id DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user_last_seen(user_id):
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE users SET last_seen = ? WHERE user_id = ?", (now, user_id))
    conn.commit()
    conn.close()


def delete_user(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM face_labels WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM interactions WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    # Also clean up disk folder if it exists
    user_dir = os.path.join(DATA_DIR, "faces", str(user_id))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir, ignore_errors=True)


# ---------- Face Encodings Table ----------

def store_face_encoding(user_id, encoding_vector, sample_path=None):
    """
    Saves numerical face representation in face_encodings table as JSON list of floats.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(encoding_vector, (list, tuple)):
        enc_str = json.dumps(list(encoding_vector))
    else:
        enc_str = json.dumps(encoding_vector.tolist())

    cur.execute(
        """
        INSERT INTO face_encodings (user_id, encoding, sample_path, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, enc_str, sample_path, now),
    )
    conn.commit()
    conn.close()


def get_face_encodings_for_user(user_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM face_encodings WHERE user_id = ?", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_total_face_encodings_count():
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) as c FROM face_encodings").fetchone()
    conn.close()
    return row["c"] if row else 0


def increment_sample_count(user_id, by=1):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO face_labels (user_id, sample_count)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET sample_count = sample_count + ?
        """,
        (user_id, by, by),
    )
    conn.commit()
    conn.close()


def all_enrolled_user_ids():
    conn = get_connection()
    rows = conn.execute("SELECT user_id FROM face_labels WHERE sample_count > 0").fetchall()
    conn.close()
    return [r["user_id"] for r in rows]


# ---------- Preferences ----------

def set_preferences(user_id, categories_with_weights=None):
    """
    Accepts either a list of category strings or a dict of {category: weight}
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))

    if isinstance(categories_with_weights, dict):
        for cat, weight in categories_with_weights.items():
            cur.execute(
                "INSERT INTO preferences (user_id, category, preference_value) VALUES (?, ?, ?)",
                (user_id, cat, float(weight)),
            )
    elif isinstance(categories_with_weights, (list, tuple)):
        for cat in categories_with_weights:
            cur.execute(
                "INSERT INTO preferences (user_id, category, preference_value) VALUES (?, ?, ?)",
                (user_id, cat, 1.0),
            )
    conn.commit()
    conn.close()


def get_preferences(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT category, preference_value FROM preferences WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Content CRUD ----------

def get_all_content():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM content ORDER BY content_id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_content_by_id(content_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM content WHERE content_id = ?", (content_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_content(title, description, category, tags, image_url=None, read_time="5 min read", video_id=None):
    conn = get_connection()
    cur = conn.cursor()
    if not image_url:
        if video_id:
            image_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        else:
            # Default placeholder per category
            placeholders = {
                "Technology": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&auto=format&fit=crop&q=80",
                "Science": "https://images.unsplash.com/photo-1507668077129-56e32842fceb?w=600&auto=format&fit=crop&q=80",
                "Sports": "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=600&auto=format&fit=crop&q=80",
                "Cooking": "https://images.unsplash.com/photo-1556910103-1c02745aae4d?w=600&auto=format&fit=crop&q=80",
                "Music": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=600&auto=format&fit=crop&q=80",
                "Travel": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=600&auto=format&fit=crop&q=80",
                "History": "https://images.unsplash.com/photo-1461360370896-922624d12aa1?w=600&auto=format&fit=crop&q=80"
            }
            image_url = placeholders.get(category, "https://images.unsplash.com/photo-1499750310107-5fef28a66643?w=600&auto=format&fit=crop&q=80")

    cur.execute(
        """
        INSERT INTO content (title, description, category, tags, image_url, read_time, video_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (title, description, category, tags, image_url, read_time, video_id),
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


def delete_content(content_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM content WHERE content_id = ?", (content_id,))
    cur.execute("DELETE FROM interactions WHERE content_id = ?", (content_id,))
    conn.commit()
    conn.close()


# ---------- Interactions ----------

def log_interaction(user_id, content_id, action="view"):
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO interactions (user_id, content_id, action, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, content_id, action, now),
    )
    conn.commit()
    conn.close()


def get_interacted_content_ids(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT DISTINCT content_id FROM interactions WHERE user_id = ?", (user_id,)
    ).fetchall()
    conn.close()
    return [r["content_id"] for r in rows]


def get_user_interactions(user_id, limit=20):
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT i.interaction_id, i.user_id, i.content_id, i.action, i.timestamp,
               c.title, c.category
        FROM interactions i
        JOIN content c ON i.content_id = c.content_id
        WHERE i.user_id = ?
        ORDER BY i.interaction_id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_interactions(limit=50):
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT i.interaction_id, i.user_id, i.content_id, i.action, i.timestamp,
               u.name as user_name, c.title as content_title, c.category
        FROM interactions i
        LEFT JOIN users u ON i.user_id = u.user_id
        LEFT JOIN content c ON i.content_id = c.content_id
        ORDER BY i.interaction_id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Analytics & Admin Stats ----------

def get_system_stats():
    conn = get_connection()
    cur = conn.cursor()

    users_count = cur.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    content_count = cur.execute("SELECT COUNT(*) FROM content").fetchone()[0]
    interactions_count = cur.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    encodings_count = cur.execute("SELECT COUNT(*) FROM face_encodings").fetchone()[0]

    category_counts = cur.execute(
        "SELECT category, COUNT(*) as c FROM content GROUP BY category"
    ).fetchall()

    conn.close()

    return {
        "total_users": users_count,
        "total_content": content_count,
        "total_interactions": interactions_count,
        "total_face_encodings": encodings_count,
        "categories": {r["category"]: r["c"] for r in category_counts}
    }
