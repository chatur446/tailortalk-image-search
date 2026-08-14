import os

import faiss
import numpy as np


EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
INDEX_PATH = "index/image_index.faiss"


def main():

    print("Loading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype("float32")

    print("Embedding shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    print("Creating FAISS index...")

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    print("Adding embeddings...")

    index.add(embeddings)

    os.makedirs(
        os.path.dirname(INDEX_PATH),
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    print("\n==============================")
    print("FAISS INDEX CREATED")
    print("==============================")
    print("Vectors:", index.ntotal)
    print("Dimensions:", dimension)
    print("Index type: IndexFlatIP")
    print("Similarity: Cosine similarity")
    print("Saved to:", INDEX_PATH)


if __name__ == "__main__":
    main()
    