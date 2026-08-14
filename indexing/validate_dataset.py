import os

import pandas as pd
from PIL import Image


IMAGE_DIR = "data/images"
METADATA_PATH = "data/metadata.csv"
REPORT_PATH = "data/validation_report.csv"


def validate_image(image_path):
    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format

        return {
            "valid": True,
            "width": width,
            "height": height,
            "format": image_format,
            "error": ""
        }

    except Exception as e:
        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "format": "",
            "error": str(e)
        }


def main():

    print("Reading metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    print(f"Metadata records: {len(metadata)}")

    results = []

    print("\nValidating images...\n")

    for _, row in metadata.iterrows():

        image_id = row["image_id"]

        image_path = row["image_path"]

        result = validate_image(image_path)

        results.append({
            "image_id": image_id,
            "image_path": image_path,
            "SKU": row["SKU"],
            "valid": result["valid"],
            "width": result["width"],
            "height": result["height"],
            "format": result["format"],
            "error": result["error"]
        })

    report = pd.DataFrame(results)

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    valid_count = report["valid"].sum()
    invalid_count = len(report) - valid_count

    print("\n==============================")
    print("DATASET VALIDATION")
    print("==============================")
    print(f"Total images : {len(report)}")
    print(f"Valid images : {valid_count}")
    print(f"Invalid      : {invalid_count}")
    print(f"Report       : {REPORT_PATH}")

    if invalid_count > 0:

        print("\nInvalid images:")

        print(
            report[
                report["valid"] == False
            ][
                ["image_id", "error"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()