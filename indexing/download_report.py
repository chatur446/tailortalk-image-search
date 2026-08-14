import pandas as pd


ORIGINAL_CSV = "byrappa_tejas_31july.csv"
METADATA_CSV = "data/metadata.csv"
OUTPUT_CSV = "data/download_failures.csv"


def main():

    original = pd.read_csv(ORIGINAL_CSV)
    metadata = pd.read_csv(METADATA_CSV)

    downloaded_ids = set(metadata["image_id"])

    failures = []

    for index, row in original.iterrows():

        image_id = f"{index + 1:06d}_{str(row['SKU']).strip()}"

        if image_id not in downloaded_ids:

            failures.append({
                "image_id": image_id,
                "SKU": row["SKU"],
                "Name": row["Name"],
                "image_url": row["image_url"],
                "Website Link": row["Website Link"]
            })

    failures_df = pd.DataFrame(failures)

    failures_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("==============================")
    print("DOWNLOAD REPORT")
    print("==============================")
    print(f"Catalogue rows : {len(original)}")
    print(f"Downloaded     : {len(metadata)}")
    print(f"Failed         : {len(failures_df)}")
    print(f"Report saved   : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()