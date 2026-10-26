"""
cross_validate.py - Tests prediction distance on each sample against a model trained without that sample.
"""
import os
import cv2
import numpy as np
import face_recognition_module as fm

def test():
    faces_dir = fm.FACES_DIR
    all_samples = []
    for uid in os.listdir(faces_dir):
        udir = os.path.join(faces_dir, uid)
        if not os.path.isdir(udir):
            continue
        for fname in os.listdir(udir):
            fpath = os.path.join(udir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_samples.append((int(uid), fname, cv2.resize(img, fm.FACE_SIZE)))

    print(f"Total samples collected: {len(all_samples)}")

    # For each sample, train a model on all other samples and test on this sample!
    print("\n--- LEAVE-ONE-OUT PREDICTION TEST ---")
    distances = []
    for i, (test_uid, test_fname, test_img) in enumerate(all_samples):
        train_faces = []
        train_labels = []
        for j, (uid, fname, img) in enumerate(all_samples):
            if i != j:
                train_faces.append(img)
                train_labels.append(uid)

        # Train
        rec = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        rec.train(train_faces, np.array(train_labels))

        pred_lbl, dist = rec.predict(test_img)
        is_same = (pred_lbl == test_uid)
        distances.append(dist)
        status = "CORRECT" if is_same else f"WRONG (pred={pred_lbl})"
        print(f"User {test_uid} ({test_fname}): dist={dist:.2f} -> {status}")

    print(f"\nAverage distance on unseen frames: {np.mean(distances):.2f}")
    print(f"Min distance: {np.min(distances):.2f}, Max distance: {np.max(distances):.2f}")

if __name__ == "__main__":
    test()
