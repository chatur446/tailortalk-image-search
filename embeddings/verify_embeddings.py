import numpy as np
import pandas as pd


METADATA_PATH = "data/metadata.csv"
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"


def main():

    metadata = pd.read_csv(METADATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    print("==============================")
    print("EMBEDDING ALIGNMENT CHECK")
    print("==============================")

    print("Metadata rows:", len(metadata))
    print("Embedding rows:", len(embeddings))
    print("Embedding dimensions:", embeddings.shape[1])

    assert len(metadata) == len(embeddings), (
        "Metadata and embeddings have different row counts!"
    )

    # Verify every metadata image path exists
    missing = [
        path
        for path in metadata["image_path"]
        if not __import__("os").path.exists(path)
    ]

    print("Missing image files:", len(missing))

    assert len(missing) == 0, (
        "Some metadata image paths do not exist!"
    )

    # Check that embeddings are normalized
    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    print(
        "Minimum embedding norm:",
        norms.min()
    )

    print(
        "Maximum embedding norm:",
        norms.max()
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-5
    ), "Some embeddings are not normalized!"

    print("\nFirst 5 mappings:")

    for i in range(min(5, len(metadata))):

        print(
            f"{i}: "
            f"{metadata.iloc[i]['image_id']} → "
            f"{metadata.iloc[i]['SKU']} → "
            f"{metadata.iloc[i]['image_path']}"
        )

    print("\n✓ Embedding alignment verified")
    print("✓ All image files exist")
    print("✓ All embeddings are normalized")


if __name__ == "__main__":
    main()
    