import pandas as pd
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)
import faiss
import numpy as np

from search.reranker import (
    metadata_similarity,
    calculate_final_score
)


INDEX_PATH = "index/image_index.faiss"
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
METADATA_PATH = "data/structured_metadata.csv"

TOP_K = 5


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading index...")
index = faiss.read_index(INDEX_PATH)

print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH)


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_image(image_path):

    query_matches = metadata[
        metadata["image_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.endswith(image_path)
    ]

    if query_matches.empty:
        print(f"\nCould not find: {image_path}")
        return

    query_row = query_matches.iloc[0]

    # Find embedding corresponding to this image
    metadata_index = query_matches.index[0]

    query_embedding = embeddings[metadata_index].reshape(1, -1).astype("float32")

    # --------------------------------------------------
    # FAISS visual search
    # --------------------------------------------------

    visual_scores, indices = index.search(
        query_embedding,
        50
    )

    visual_results = []

    for score, idx in zip(
        visual_scores[0],
        indices[0]
    ):

        idx = int(idx)

        candidate = metadata.iloc[idx]

        # Don't return the query image itself
        candidate_path = (
            str(candidate["image_path"])
            .replace("\\", "/")
        )

        if candidate_path.endswith(image_path):
            continue

        visual_results.append({
            "SKU": candidate["SKU"],
            "Name": candidate["Name"],
            "visual_score": float(score),
            "metadata_score": metadata_similarity(
                query_row,
                candidate
            )
        })


    # --------------------------------------------------
    # Metadata reranking
    # --------------------------------------------------

    for result in visual_results:

        result["final_score"] = calculate_final_score(
            result["visual_score"],
            result["metadata_score"]
        )


    reranked_results = sorted(
        visual_results,
        key=lambda x: x["final_score"],
        reverse=True
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print("Image:", image_path)
    print("SKU:", query_row["SKU"])
    print("Name:", query_row["Name"])


    print("\n" + "-" * 80)
    print("CLIP / FAISS TOP 5")
    print("-" * 80)

    for i, result in enumerate(
        visual_results[:TOP_K],
        start=1
    ):

        print(
            f"{i}. "
            f"{result['SKU']} | "
            f"visual={result['visual_score']:.4f} | "
            f"metadata={result['metadata_score']:.4f}"
        )


    print("\n" + "-" * 80)
    print("AFTER METADATA RERANKING")
    print("-" * 80)

    for i, result in enumerate(
        reranked_results[:TOP_K],
        start=1
    ):

        print(
            f"{i}. "
            f"{result['SKU']} | "
            f"visual={result['visual_score']:.4f} | "
            f"metadata={result['metadata_score']:.4f} | "
            f"final={result['final_score']:.4f}"
        )


# --------------------------------------------------
# Test cases
# --------------------------------------------------

TEST_IMAGES = [
    "000011_QA255417.jpg",
    "000101_QS236750.jpg",
    "000505_QW207664.jpg",
    "001006_AA311090.jpg"
]


for image in TEST_IMAGES:
    evaluate_image(image)