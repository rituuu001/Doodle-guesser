import os
import requests

CATEGORIES = ["cup",
              "donut",
              "envelope",
              "house",
              "guitar",
              "pants"
              ]

RAW_DIR = "data/raw"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

def download_category(category):
    filename = category + ".npy"
    filepath = os.path.join(RAW_DIR, filename)

    if os.path.exists(filepath):
        print(f"Already downloaded: {filename}")
        return filepath
    
    url_name = category.replace(" ", "_") + ".npy"
    url = BASE_URL + url_name
    print(f"Downloading {filename} ...")

    response = requests.get(url, stream = True)
    response.raise_for_status()

    os.makedirs(RAW_DIR, exist_ok = True)
    with open(filepath,"wb") as f:
        for chunk in response.iter_content(chunk_size = 8192):
            f.write(chunk)

    print(f"Done downloading {filename}")
    return filepath

if __name__ == "__main__":
    for category in CATEGORIES:
        download_category(category)

    print("All downloads completed.")



