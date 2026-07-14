import numpy as np
import os

RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/dataset.bin"
SAMPLES_PER_CLASS = 1000
SEED = 42

CATEGORIES = ["cup",
              "donut",
              "envelope",
              "house",
              "guitar",
              "pants"
              ]

def pack_binary():
    all_images = []
    all_labels = []
    for label, category in enumerate(CATEGORIES):
        filepath = os.path.join(RAW_DIR, category + ".npy")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}. Please download the dataset first.")
        data = np.load(filepath)
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(data), size=SAMPLES_PER_CLASS, replace=False)
        sampled = data[indices]
        all_images.append(sampled)
        all_labels.extend([label] * SAMPLES_PER_CLASS)
        print(f"Sampled {SAMPLES_PER_CLASS} images for '{category}' (label {label})")
    all_images = np.vstack(all_images)
    all_labels = np.array(all_labels, dtype=np.int32)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"\nTotal images: {all_images.shape}")

    with open(OUTPUT_PATH, "wb") as f:
        f.write(np.int32(len(all_labels)).tobytes())
        f.write(np.int32(len(CATEGORIES)).tobytes())
        
        for img, label in zip(all_images, all_labels):
            f.write(img.astype(np.uint8).tobytes())
            f.write(np.uint8(label).tobytes())
    
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    pack_binary()