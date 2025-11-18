import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import shutil
import time

BRAND_QUERIES = [
    "Coca Cola logo", "Coca Cola brand", "Coca Cola emblem",
    "Coca Cola transparent", "Coca Cola sign",
    "Coca Cola icon", "Coca Cola vector", "Coca Cola classic",
    "Coca Cola red logo", "Coca Cola white logo",
    "Coca Cola bottle", "Coca Cola label", "Coca Cola design",
    "Coca Cola vintage", "Coca Cola retro",
    "Coca Cola 3d", "Coca Cola illustration", "Coca Cola background",
    "Coca Cola print", "Coca Cola image"
]

NEGATIVE_QUERIES = [
    "technology gadgets", "urban architecture", "winter landscapes",
    "desert photography", "workspace photos",
    "vehicle photos", "fashion photography", "books images",
    "flowers photography", "night city lights",
    "beach sunset", "autumn colors", "modern interiors",
    "industrial design", "geometric patterns",
    "water drops", "clouds photography", "vintage objects",
    "neon lights", "minimalist art"
]

NUM_IMAGES_PER_QUERY = 40
OUTPUT_DIR = "dataset"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_image(url, path):
    try:
        r = requests.get(url, stream=True, timeout=5)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def search_images(query, num_images=40):
    print(f"\nSearching images for: {query}")
    headers = {"User-Agent": "Mozilla/5.0"}
    query_string = urllib.parse.quote(query)
    image_urls = []
    offset = 0

    while len(image_urls) < num_images:
        url = f"https://www.bing.com/images/async?q={query_string}&first={offset}&count=100"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        for img in soup.find_all("a", class_="iusc"):
            try:
                m = img.get("m")
                if m:
                    m_json = eval(m)
                    img_url = m_json.get("murl")
                    if img_url and img_url not in image_urls:
                        image_urls.append(img_url)
            except Exception:
                continue
            if len(image_urls) >= num_images:
                break

        offset += 100
        time.sleep(1)
        if offset > num_images * 3:
            break

    print(f"Found {len(image_urls)} images for '{query}'")
    return image_urls

def main():
    create_dir(OUTPUT_DIR)

    brand_dir = os.path.join(OUTPUT_DIR, "positive")
    neg_dir = os.path.join(OUTPUT_DIR, "negative")
    create_dir(brand_dir)
    create_dir(neg_dir)

    idx = 0
    for query in BRAND_QUERIES:
        brand_urls = search_images(query, NUM_IMAGES_PER_QUERY)
        for url in brand_urls:
            path = os.path.join(brand_dir, f"{idx}.jpg")
            download_image(url, path)
            idx += 1

    idx = 0
    for query in NEGATIVE_QUERIES:
        neg_urls = search_images(query, NUM_IMAGES_PER_QUERY)
        for url in neg_urls:
            path = os.path.join(neg_dir, f"{idx}.jpg")
            download_image(url, path)
            idx += 1

    print(f"\nDataset download completed! Data saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()