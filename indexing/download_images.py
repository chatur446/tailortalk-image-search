import os
import io
import pandas as pd
import requests

from PIL import Image
from tqdm import tqdm


CSV_PATH = "byrappa_tejas_31july.csv"
IMAGE_DIR = "data/images"

os.makedirs(IMAGE_DIR, exist_ok=True)


def download_image(url, output_path):
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        image = image.convert("RGB")

        image.save(output_path, format="JPEG", quality=95)

        return True

    except Exception as e:
        print(f"\nFailed: {url}")
        print(f"Reason: {e}")
        return False


def main():

    print("Reading CSV...")

    df = pd.read_csv(CSV_PATH)

    print(f"Total products: {len(df)}")

    successful = 0
    failed = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Downloading images"
    ):

        sku = str(row["SKU"]).strip()
        url = str(row["image_url"]).strip()

        output_path = os.path.join(
            IMAGE_DIR,
            f"{sku}.jpg"
        )

        if os.path.exists(output_path):
            successful += 1
            continue

        if download_image(url, output_path):
            successful += 1
        else:
            failed += 1

    print("\n==============================")
    print("Download complete")
    print("==============================")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(df)}")


if __name__ == "__main__":
    main()