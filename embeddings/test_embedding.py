import torch

from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_PATH = "data/images/000001_QS204820.jpg"


def main():

    print("Loading model...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    model.eval()

    print("Loading image...")

    image = Image.open(IMAGE_PATH).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    print("Generating embedding...")

    with torch.no_grad():

        outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        image_features = model.visual_projection(
            outputs.pooler_output
        )

    # Normalize the embedding
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    embedding = image_features[0]

    print("\n==============================")
    print("EMBEDDING TEST")
    print("==============================")
    print("Model:", MODEL_NAME)
    print("Embedding dimensions:", embedding.shape[0])
    print("Embedding norm:", embedding.norm().item())
    print("First 10 values:")
    print(embedding[:10])


if __name__ == "__main__":
    main()