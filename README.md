# FaceSense AI: Face Recognition-Based Personalized Content Recommendation System

**Academic Year:** 2026  
**Programme:** B.Sc. Computer Science  
**Student Name:** Vishwa DK  
**Register / College ID:** C24UG206CSC052  
**Institution:** Sona College of Arts and Science  
**Affiliated University:** Periyar University  
**Project Guide:** S. Indra, M.Sc.  

---

## 1. Project Overview & Abstract

**FaceSense AI** is a web-based personalized content recommendation system that integrates computer vision biometric identification with information retrieval. 

Unlike conventional recommendation systems that require cumbersome login credentials or manual profile selections, **FaceSense AI** uses a browser-accessible camera (`navigator.mediaDevices.getUserMedia()`) to detect and recognize the user's face in real-time. Upon positive facial verification, the system loads the recognized user's preference profile and computes personalized content recommendations using **Term Frequency-Inverse Document Frequency (TF-IDF)** and **Cosine Similarity**.

For unknown or unrecognized faces, the system activates a **privacy-safe fallback mode**, serving curated trending content and offering an instant biometric enrollment flow without ever exposing another user's personal profile.

---

## 2. All 10 Implemented Project Modules

| # | Module | Implementation Description | Primary Files |
|---|---|---|---|
| **1** | **Camera & Input Module** | Captures live video stream from laptop/phone camera using HTML5 `getUserMedia()`. Supports front/rear cameras, upload fallback, and snapshot framing. | `static/js/camera.js`, `templates/index.html` |
| **2** | **Face Detection Module** | Uses OpenCV Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) to detect faces, calculate bounding boxes `(x, y, w, h)`, and draw real-time tracking HUD overlays. | `face_recognition_module.py` |
| **3** | **Face Enrollment Module** | 2-stage interactive wizard that guides users through capturing 6 multi-angle face samples, extracts 128-d numerical embeddings, and stores them in the `face_encodings` database table. | `face_recognition_module.py`, `app.py` |
| **4** | **Face Recognition Module** | Employs Local Binary Patterns Histograms (LBPH) via `cv2.face.LBPHFaceRecognizer`. Compares live embeddings with stored templates, computes Euclidean distance, and returns a verified match percentage. | `face_recognition_module.py` |
| **5** | **User Profile Module** | Manages user credentials, registration timestamps, last-seen timestamps, and fine-tuneable category interest weights (0.0x to 2.0x). | `database.py`, `templates/index.html` |
| **6** | **Content Management Module** | Real-world multimedia catalog of 3,607 video entries with YouTube video IDs, categories (Technology & AI, Travel, Music & Art, Food & Cooking, History & Culture), tags, watch times, and official YouTube thumbnails. | `database.py`, `import_youtube_dataset.py`, `static/js/camera.js` |
| **7** | **Recommendation Engine** | Scikit-learn `TfidfVectorizer` (unigrams + bigrams) and `cosine_similarity` calculating semantic relevance scores (%) and explainable AI reasons. In-memory caching for sub-second responses. | `recommendation.py` |
| **8** | **Unknown User Module** | Privacy-first architecture: unrecognized faces receive general trending items and an enrollment CTA, completely isolating enrolled users' data. | `recommendation.py`, `app.py` |
| **9** | **Web UI Module** | Modern responsive glassmorphism UI with biometric scanning reticle, animated confidence bars, filter pills, and embedded YouTube video player modal. | `templates/index.html`, `static/css/style.css` |
| **10** | **Admin / Data Management** | Comprehensive administration dashboard displaying real-time metrics, registered users management, content item creation, and live interaction audit logs. | `app.py`, `templates/index.html` |

---

## 2.1 YouTube Multimedia Video Dataset (`dataset/youtube.csv`)

The system integrates a real-world benchmark dataset of **3,607 multimedia video records**:

- **Dataset Source:** `dataset/youtube.csv` (1.9 MB)
- **Features Extracted:** YouTube Video ID (`link`), Title, Cleaned Channel Descriptions, Categories.
- **Dataset Distribution:**
  - ✈️ **Travel & Exploration:** 1,156 videos
  - 🎵 **Music & Art:** 947 videos
  - 🍲 **Food & Cooking:** 903 videos
  - 🏛️ **History & Culture:** 593 videos
  - 💻 **Technology & AI:** 8 curated foundational Computer Science & AI videos (3Blue1Brown, Lex Fridman, Fireship, Two Minute Papers)
- **Interactive Multimedia Embeds:** Selecting any recommendation opens a built-in, responsive **YouTube Video Player (`<iframe>`)** with direct YouTube watch links.

---

## 3. Database Architecture (SQLite Schema)

The SQLite database (`data/facesense.db`) strictly adheres to the 5 suggested tables in Section 10 of the project specifications:

```sql
-- 1. Users Profile Table
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    created_at TEXT NOT NULL,
    last_seen TEXT
);

-- 2. Face Encodings Biometric Table
CREATE TABLE face_encodings (
    encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    encoding TEXT NOT NULL,         -- JSON serialized 128-d numerical feature vector
    sample_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 3. User Category Preferences Table
CREATE TABLE preferences (
    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    category TEXT NOT NULL,
    preference_value REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 4. Content Catalogue Table
CREATE TABLE content (
    content_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    tags TEXT NOT NULL,
    image_url TEXT,
    read_time TEXT DEFAULT '5 min read'
);

-- 5. User Interaction History Table (Feedback Loop)
CREATE TABLE interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    content_id INTEGER NOT NULL,
    action TEXT NOT NULL,          -- 'view', 'like', 'bookmark', 'click'
    timestamp TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
);
```

---

## 4. Mathematical Formulations & Algorithms

### A. Term Frequency - Inverse Document Frequency (TF-IDF)
Converts text descriptions and tags into numerical feature vectors:
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### B. Cosine Similarity Metric
Measures the cosine angle between the user preference vector $\vec{u}$ and candidate content vector $\vec{c}$:
$$\text{Cosine Similarity}(\vec{u}, \vec{c}) = \frac{\vec{u} \cdot \vec{c}}{\|\vec{u}\| \|\vec{c}\|} = \frac{\sum_{i=1}^{n} u_i c_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n c_i^2}}$$

### C. Local Binary Patterns Histograms (LBPH) Face Distance
Computes invariant pixel neighborhoods and compares spatial histograms:
$$D(H_1, H_2) = \sqrt{\sum_{i=1}^N (H_1(i) - H_2(i))^2}$$
- **Threshold:** Set at $\le 75.0$ distance for verified match.

---

## 5. Quick Start & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.12.4)
- Web browser with webcam access (Chrome, Edge, Firefox)

### Step 1: Install Dependencies
Open PowerShell or Terminal inside the `facesense_ai` directory:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
python app.py
```

### Step 3: Open in Browser
Visit **`http://127.0.0.1:5000`** in your browser.

> **Note for Mobile / Remote Testing:**
> Modern mobile browsers restrict camera access to secure origins (`localhost` or `https://`). When testing on a smartphone on the same local Wi-Fi, run an HTTPS tunnel (e.g. `ngrok http 5000`) or test on your local laptop browser.

---

## 6. How to Conduct a Live Project Presentation / Viva

1. **Demonstrate Unrecognized Face (Safe Mode):**
   - Click **Recognize Face** before enrolling. The system will detect an unknown face, display the amber alert *"Unrecognized Face Detected"*, and show safe trending items without revealing any private user profiles.
2. **Demonstrate Face Enrollment:**
   - Click **➕ Enroll New Face**.
   - Enter Name (e.g., `Vishwa DK`) and College ID (`C24UG206CSC052`).
   - Select interests: `Technology` and `Science`.
   - Click **Start Auto-Capture**. Follow the 6 prompt angles as thumbnails appear and the model retrains in real time.
3. **Demonstrate Personalized Recognition:**
   - Click **Recognize Face**.
   - Observe the live green/cyan bounding box, match confidence (e.g. `94.8%`), and immediate personalized recommendations for Technology and Science.
4. **Demonstrate Adaptive Feedback Loop:**
   - Click the **❤️ Like** button on an article.
   - Switch to the **👤 My Profile** tab to view your updated interaction timeline.
   - Adjust the category preference sliders and watch recommendations recalculate in real-time.
5. **Demonstrate Admin Hub (Module 10):**
   - Open the **⚙️ Admin Hub** tab to show the external examiner the database records, enrolled sample count, and live interaction audit logs.
6. **Display Architecture & Viva Guide:**
   - Switch to the **📄 Project Docs** tab to show the examiners the built-in DFD Level 0, DFD Level 1, mathematical equations, and project credentials.

---

## 7. Report Chapter Outline (Periyar University Submission)

- **Chapter 1:** Introduction & Project Motivation
- **Chapter 2:** Literature Survey & Existing System Limitations
- **Chapter 3:** Proposed FaceSense AI Architecture
- **Chapter 4:** Software & Hardware Requirements Analysis
- **Chapter 5:** System Design (DFD Level 0, Level 1, Use Case & ER Diagrams)
- **Chapter 6:** Computer Vision & Recommendation Methodology (Haar Cascade, LBPH, TF-IDF, Cosine Similarity)
- **Chapter 7:** Detailed Module Descriptions (Modules 1 through 10)
- **Chapter 8:** Database Design & SQLite Normalization
- **Chapter 9:** Implementation Details & Technology Stack
- **Chapter 10:** Software Testing & Verification (`test_system.py`)
- **Chapter 11:** Experimental Results, Confidence Metrics & Performance Discussion
- **Chapter 12:** Security, Biometric Privacy & System Limitations
- **Chapter 13:** Future Enhancements (Liveness Detection, DeepFace, Collaborative Filtering)
- **Chapter 14:** Conclusion & Summary
- **References & Appendices**
