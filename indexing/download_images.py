import io
import os

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


CSV_PATH = "byrappa_tejas_31july.csv"
IMAGE_DIR = "data/images"
METADATA_PATH = "data/metadata.csv"


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

        image = Image.open(
            io.BytesIO(response.content)
        )

        image = image.convert("RGB")

        image.save(
            output_path,
            format="JPEG",
            quality=95
        )

        return True

    except Exception as e:
        print(f"\nFailed: {url}")
        print(f"Reason: {e}")
        return False


def main():

    print("Reading CSV...")

    df = pd.read_csv(CSV_PATH)

    print(f"Total catalogue rows: {len(df)}")

    metadata_rows = []

    successful = 0
    failed = 0

    for index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Downloading images"
    ):

        sku = str(row["SKU"]).strip()
        url = str(row["image_url"]).strip()

        # Unique ID for every catalogue row
        image_id = f"{index + 1:06d}_{sku}"

        output_path = os.path.join(
            IMAGE_DIR,
            f"{image_id}.jpg"
        )

        # Download only if this image doesn't already exist
        if os.path.exists(output_path):
            successful += 1

            metadata_rows.append({
                "image_id": image_id,
                "image_path": output_path,
                "Name": row["Name"],
                "SKU": row["SKU"],
                "Stock": row["Stock"],
                "Retail Price": row["Retail Price"],
                "Discounted Price": row["Discounted Price"],
                "image_url": row["image_url"],
                "Website Link": row["Website Link"]
            })

            continue

        if download_image(url, output_path):

            successful += 1

            metadata_rows.append({
                "image_id": image_id,
                "image_path": output_path,
                "Name": row["Name"],
                "SKU": row["SKU"],
                "Stock": row["Stock"],
                "Retail Price": row["Retail Price"],
                "Discounted Price": row["Discounted Price"],
                "image_url": row["image_url"],
                "Website Link": row["Website Link"]
            })

        else:
            failed += 1

    metadata_df = pd.DataFrame(metadata_rows)

    metadata_df.to_csv(
        METADATA_PATH,
        index=False
    )

    print("\n==============================")
    print("Download complete")
    print("==============================")
    print(f"Total rows:       {len(df)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(f"Metadata records: {len(metadata_df)}")
    print(f"Metadata saved:   {METADATA_PATH}")


if __name__ == "__main__":
    main()