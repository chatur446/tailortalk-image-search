import faiss
import numpy as np
import pandas as pd


INDEX_PATH = "index/image_index.faiss"
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
METADATA_PATH = "data/metadata.csv"


def main():

    print("Loading index...")

    index = faiss.read_index(INDEX_PATH)

    print("Loading metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    print("Loading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype("float32")

    query_indices = [0, 10, 100, 500, 1000]

    for query_index in query_indices:

        query_embedding = embeddings[
            query_index
        ].reshape(1, -1)

        scores, indices = index.search(
            query_embedding,
            6
        )

        print("\n==============================")
        print("QUERY")
        print("==============================")

        print(
            "Image:",
            metadata.iloc[query_index]["image_id"]
        )

        print(
            "SKU:",
            metadata.iloc[query_index]["SKU"]
        )

        print(
            "Name:",
            metadata.iloc[query_index]["Name"]
        )

        print("\nTop 5 similar products:\n")

        result_count = 0

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            # Skip the query image itself
            if idx == query_index:
                continue

            result_count += 1

            row = metadata.iloc[idx]

            print(
                f"{result_count}. "
                f"Score={score:.4f} | "
                f"SKU={row['SKU']} | "
                f"Name={row['Name']} | "
                f"Image={row['image_path']}"
            )

            if result_count == 5:
                break


if __name__ == "__main__":
    main()