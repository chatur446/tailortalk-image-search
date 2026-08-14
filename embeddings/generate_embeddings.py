import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"

METADATA_PATH = "data/metadata.csv"
OUTPUT_PATH = "embeddings/image_embeddings.npy"

BATCH_SIZE = 16


def main():

    print("Loading metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    print(f"Images to embed: {len(metadata)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    print("\nLoading CLIP model...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    embeddings = []

    print("\nGenerating embeddings...")

    for start in tqdm(
        range(0, len(metadata), BATCH_SIZE),
        desc="Embedding images"
    ):

        batch = metadata.iloc[
            start:start + BATCH_SIZE
        ]

        images = []

        valid_rows = []

        for _, row in batch.iterrows():

            try:

                image = Image.open(
                    row["image_path"]
                ).convert("RGB")

                images.append(image)
                valid_rows.append(row)

            except Exception as e:

                print(
                    f"\nFailed to load {row['image_path']}: {e}"
                )

        if not images:
            continue

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            batch_embeddings = model.visual_projection(
                outputs.pooler_output
            )

        # Normalize embeddings
        batch_embeddings = (
            batch_embeddings
            / batch_embeddings.norm(
                dim=-1,
                keepdim=True
            )
        )

        embeddings.append(
            batch_embeddings.cpu().numpy()
        )

    embeddings = np.vstack(embeddings)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        embeddings
    )

    print("\n==============================")
    print("EMBEDDING GENERATION COMPLETE")
    print("==============================")
    print("Model:", MODEL_NAME)
    print("Embedding shape:", embeddings.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()