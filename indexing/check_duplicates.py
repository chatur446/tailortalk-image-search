import hashlib
import os
from collections import defaultdict

import pandas as pd


IMAGE_DIR = "data/images"
OUTPUT_PATH = "data/duplicate_images.csv"


def file_hash(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def main():

    print("Checking for duplicate image files...")

    hashes = defaultdict(list)

    files = [
        file
        for file in os.listdir(IMAGE_DIR)
        if file.lower().endswith(".jpg")
    ]

    for i, filename in enumerate(files, start=1):

        path = os.path.join(IMAGE_DIR, filename)

        image_hash = file_hash(path)

        hashes[image_hash].append(filename)

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} images")

    duplicate_groups = []

    for image_hash, filenames in hashes.items():

        if len(filenames) > 1:

            duplicate_groups.append({
                "hash": image_hash,
                "count": len(filenames),
                "files": " | ".join(filenames)
            })

    duplicates = pd.DataFrame(duplicate_groups)

    duplicates.to_csv(
        OUTPUT_PATH,
        index=False
    )

    total_duplicate_files = sum(
        group["count"]
        for group in duplicate_groups
    )

    print("\n==============================")
    print("DUPLICATE IMAGE CHECK")
    print("==============================")
    print(f"Total images       : {len(files)}")
    print(f"Duplicate groups   : {len(duplicate_groups)}")
    print(f"Files in duplicates: {total_duplicate_files}")
    print(f"Report             : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()