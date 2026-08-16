from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

from search.reranker import (
    metadata_similarity,
    calculate_final_score
)


MODEL_NAME = "openai/clip-vit-base-patch32"

INDEX_PATH = "index/image_index.faiss"
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
METADATA_PATH = "data/structured_metadata.csv"

TOP_K_VISUAL = 50
TOP_K_FINAL = 5


app = FastAPI(
    title="TailorTalk Image Search API",
    description="Visual saree search using CLIP, FAISS and metadata reranking",
    version="1.0.0"
)


print("Loading index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH)

print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Loading CLIP model...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.eval()


@app.get("/")
def root():
    return {
        "message": "TailorTalk Image Search API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    
    # -----------------------------
    # 0. Validate uploaded file
    # -----------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPG, PNG, or WEBP."
        )

    # -----------------------------
    # 1. Read uploaded image
    # -----------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )


    # -----------------------------
    # 2. Generate CLIP embedding
    # -----------------------------

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output

    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    query_embedding = (
        image_features
        .cpu()
        .numpy()
        .astype("float32")
    )


    # -----------------------------
    # 3. FAISS visual search
    # -----------------------------

    visual_scores, indices = index.search(
        query_embedding,
        TOP_K_VISUAL
    )


    

    query_sku = file.filename.rsplit(".", 1)[0].split("_")[-1]

    query_reference = None
    query_row = None


    # First try to find the exact image
    query_matches = metadata[
        metadata["image_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.endswith(file.filename)
    ]

    if not query_matches.empty:
        query_row = query_matches.iloc[0]

        query_reference = {
            "SKU": query_row["SKU"],
            "Name": query_row["Name"]
        }

    # Fallback to SKU if exact image is not found
    elif not metadata[
        metadata["SKU"].astype(str) == str(query_sku)
    ].empty:

        query_matches = metadata[
            metadata["SKU"].astype(str) == str(query_sku)
        ]

        query_row = query_matches.iloc[0]

        query_reference = {
            "SKU": query_row["SKU"],
            "Name": query_row["Name"]
        }

    candidates = []

    for visual_score, idx in zip(
        visual_scores[0],
        indices[0]
    ):

        idx = int(idx)

        candidate_row = metadata.iloc[idx]
        candidate_image_path = (
            str(candidate_row["image_path"])
            .replace("\\", "/")
        )

        if candidate_image_path.endswith(file.filename):
            continue
 
        metadata_score = (
            metadata_similarity(query_row, candidate_row)
            if query_row is not None
            else 0.0
        )

        final_score = calculate_final_score(
            float(visual_score),
            float(metadata_score)
        )

        candidates.append({
            "visual_score": float(visual_score),
            "metadata_score": float(metadata_score),
            "final_score": float(final_score),
            "SKU": candidate_row["SKU"],
            "Name": candidate_row["Name"],
            "image_path": candidate_row["image_path"]
        })


    # -----------------------------
    # 6. Sort by final score
    # -----------------------------

    candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )


    # -----------------------------
    # 7. Return top results
    # -----------------------------

    return {
        "filename": file.filename,
        "query_reference": query_reference,
        "results": candidates[:TOP_K_FINAL]
    }