import faiss
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from reranker import (
    metadata_similarity,
    calculate_final_score
)


MODEL_NAME = "openai/clip-vit-base-patch32"

INDEX_PATH = "index/image_index.faiss"
METADATA_PATH = "data/structured_metadata.csv"

TOP_K_VISUAL = 20
TOP_K_FINAL = 5


def load_model():

    print("Loading CLIP model...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model


def generate_embedding(
    image_path,
    processor,
    model
):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.get_image_features(
            **inputs
        )

    # Handle different transformers versions
    if hasattr(outputs, "pooler_output"):
        embedding = outputs.pooler_output
    elif hasattr(outputs, "image_embeds"):
        embedding = outputs.image_embeds
    else:
        embedding = outputs

    embedding = embedding.detach().cpu().numpy()

    embedding = embedding / np.linalg.norm(
        embedding,
        axis=1,
        keepdims=True
    )

    return embedding.astype("float32")


def main():

    print("Loading index...")

    index = faiss.read_index(
        INDEX_PATH
    )

    print("Loading metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print("Loading model...")

    processor, model = load_model()

    query_image_id = "001006_AA311090"

    query_row = metadata[
        metadata["image_id"] == query_image_id
    ].iloc[0]

    print("\n==============================")
    print("QUERY")
    print("==============================")

    print(
        "Image:",
        query_row["image_id"]
    )

    print(
        "SKU:",
        query_row["SKU"]
    )

    print(
        "Name:",
        query_row["Name"]
    )

    print("\nGenerating query embedding...")

    query_embedding = generate_embedding(
        query_row["image_path"],
        processor,
        model
    )

    # ------------------------------------------------
    # STEP 1: Visual search
    # ------------------------------------------------

    print("\nSearching FAISS...")

    visual_scores, indices = index.search(
        query_embedding,
        TOP_K_VISUAL
    )

    candidates = []

    # ------------------------------------------------
    # STEP 2: Metadata reranking
    # ------------------------------------------------

    for rank, (score, idx) in enumerate(
        zip(
            visual_scores[0],
            indices[0]
        )
    ):

        candidate = metadata.iloc[idx]

        # Don't return the query itself
        if candidate["image_id"] == query_image_id:
            continue

        metadata_score = metadata_similarity(
            query_row,
            candidate
        )

        final_score = calculate_final_score(
            float(score),
            metadata_score
        )

        candidates.append({
            "visual_score": float(score),
            "metadata_score": metadata_score,
            "final_score": final_score,
            "candidate": candidate
        })

    # ------------------------------------------------
    # STEP 3: Sort by final score
    # ------------------------------------------------

    candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    print("\n==============================")
    print("VISUAL + METADATA SEARCH")
    print("==============================")

    print("\nTop 5 results:\n")

    for rank, result in enumerate(
        candidates[:TOP_K_FINAL],
        1
    ):

        candidate = result["candidate"]

        print(
            f"{rank}. "
            f"Final={result['final_score']:.4f} | "
            f"Visual={result['visual_score']:.4f} | "
            f"Metadata={result['metadata_score']:.4f}"
        )

        print(
            f"   SKU: {candidate['SKU']}"
        )

        print(
            f"   Name: {candidate['Name']}"
        )

        print(
            f"   Image: {candidate['image_path']}"
        )

        print()


if __name__ == "__main__":
    main()