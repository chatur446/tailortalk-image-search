import pandas as pd

from reranker import (
    metadata_similarity,
    calculate_final_score
)


METADATA_PATH = "data/structured_metadata.csv"


def main():

    print("Loading metadata...")

    df = pd.read_csv(METADATA_PATH)

    # Use the first product as the query
    query = df.iloc[0]

    print("\n==============================")
    print("RERANKER TEST")
    print("==============================")

    print("Query:")
    print(query["Name"])

    print("\nQuery metadata:")
    print("Fabrics:", query["fabrics"])
    print("Colours:", query["colours"])
    print("Designs:", query["designs"])

    print("\nComparing candidates...\n")

    results = []

    # Test against next 10 products
    for i in range(1, 11):

        candidate = df.iloc[i]

        metadata_score = metadata_similarity(
            query,
            candidate
        )

        # Fake visual score for testing
        visual_score = 0.95 - (i * 0.01)

        final_score = calculate_final_score(
            visual_score,
            metadata_score
        )

        results.append({
            "Name": candidate["Name"],
            "Visual": visual_score,
            "Metadata": metadata_score,
            "Final": final_score
        })

    results.sort(
        key=lambda x: x["Final"],
        reverse=True
    )

    for rank, result in enumerate(results, 1):

        print(
            f"{rank}. "
            f"Visual={result['Visual']:.4f} | "
            f"Metadata={result['Metadata']:.4f} | "
            f"Final={result['Final']:.4f} | "
            f"{result['Name']}"
        )


if __name__ == "__main__":
    main()