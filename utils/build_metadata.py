import pandas as pd

from metadata_parser import parse_product_name


INPUT_PATH = "data/metadata.csv"
OUTPUT_PATH = "data/structured_metadata.csv"


def main():

    print("Loading metadata...")

    df = pd.read_csv(INPUT_PATH)

    print("Products:", len(df))

    structured = []

    for _, row in df.iterrows():

        parsed = parse_product_name(
            row["Name"]
        )

        structured.append({
            "image_id": row["image_id"],
            "SKU": row["SKU"],
            "Name": row["Name"],
            "image_path": row["image_path"],
            "Website Link": row["Website Link"],
            "Stock": row["Stock"],
            "Retail Price": row["Retail Price"],
            "Discounted Price": row["Discounted Price"],
            "fabrics": "|".join(parsed["fabrics"]),
            "colours": "|".join(parsed["colours"]),
            "designs": "|".join(parsed["designs"]),
        })

    result = pd.DataFrame(structured)

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n==============================")
    print("STRUCTURED METADATA CREATED")
    print("==============================")
    print("Rows:", len(result))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()