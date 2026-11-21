"""


Standalone Python version of the original analysis notebook.

Dependencies:
    pip install langchain-chroma==0.2.6 langchain-community==0.3.31 langchain==0.3.27 numpy pillow sentence-transformers==2.7.0 torch torchvision transformers

Configuration (environment variables, no values hardcoded):
    ARCHIVE_URL  - set to your data file location
"""

import os
import shutil
import tarfile
import urllib.request
import zipfile

# ---- Configuration -------------------------------------------------
# Values are read from the environment so nothing is hardcoded.
ARCHIVE_URL = os.environ.get("ARCHIVE_URL", "")  # set to your data file location


# ---- Data helpers --------------------------------------------------
def download_file(url, dest):
    """Download `url` to `dest` (skips if the file already exists)."""
    if not url:
        raise RuntimeError("No source URL configured - set the matching environment variable.")
    if os.path.exists(dest):
        print("Already present:", dest)
        return dest
    print("Downloading", dest, "...")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as handle:
        shutil.copyfileobj(response, handle)
    return dest


def extract_archive(archive_path, dest_dir="."):
    """Extract a .tar/.tar.gz/.zip archive into `dest_dir`."""
    os.makedirs(dest_dir, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(dest_dir)
    else:
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(path=dest_dir)
    print("Extracted", archive_path, "->", dest_dir)
    return dest_dir


def download_and_extract(url, dest_dir="."):
    """Download an archive and extract it into `dest_dir`."""
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, os.path.basename(url.split("?")[0]) or "dataset.tar")
    download_file(url, archive_path)
    return extract_archive(archive_path, dest_dir)

# ---- Construct a Multimodal Vector Index -------------------------------

# ---- Introduction ------------------------------------------------------
# Retrieval-Augmented Generation (RAG) systems depend critically on the quality of their retrieval layer to produce grounded and contextually accurate responses. In real-world applications, user queries often span heterogeneous data sources, including structured textual information and visual content. As a result, modern RAG pipelines increasingly adopt multimodal retrieval architectures capable of indexing and searching across multiple data modalities.
# To support these scenarios, multimodal systems encode different data types into dense vector representations that enable efficient similarity-based search. A well-designed retrieval layer ensures that high-quality evidence is identified before any downstream reasoning or generation occurs.
# In this lab, you will build the retrieval backbone of a multimodal RAG system for restaurant and food discovery. Using structured outputs from Module 1 together with raw food images, you will construct vectorized, searchable indexes that support cross-modal evidence retrieval.
# The lab emphasizes retrieval pipeline engineering, focusing on how heterogeneous data is transformed, embedded, and organized within scalable vector databases using ChromaDB. This modular design mirrors production RAG systems, where the retrieval layer operates as a reusable component independent of the generation model.
# Upon completion, you will have a functional multimodal vector index that forms the foundation for similarity search, metadata-constrained retrieval, and multimodal fusion in subsequent lessons.

# ---- Objectives --------------------------------------------------------
# After completing this lab, you will be able to:
# - Construct retrieval-ready documents from structured restaurant data
# - Generate dense embeddings for both text and image modalities
# - Build separate vector indexes for articles and food images using Chroma
# - Persist multimodal vector databases for downstream retrieval tasks
# - Prepare the retrieval backbone for similarity search and metadata filtering
# These capabilities form the foundation of a production-style multimodal RAG retrieval pipeline.

# ---- Set up the lab environment ----------------------------------------
# In this step, you'll install the core libraries required to build a multimodal retrieval system.
# The environment supports both text and image processing and prepares the runtime for vector database construction. These components will be used throughout the module.
# The environment includes:
# - PyTorch (CPU) for model inference and tensor operations
# - LangChain + Chroma for vector database construction and persistence
# - Sentence-Transformers for generating text embeddings
# - Transformers (CLIP) for image embedding support
# - Pillow and NumPy for image loading and numerical processing
# After installation completes, the environment will be ready for multimodal vector index construction.

# ================================
# Install dependencies
# ================================

# ---- Import core libraries ---------------------------------------------
# Next, import the core Python libraries used throughout this lab.
# These libraries enable:
# - structured data loading and preprocessing
# - embedding model initialization
# - multimodal document construction
# - vector database creation and management
# Successful import confirms that the runtime environment is correctly configured and ready for the retrieval pipeline.

# ================================
# Import libraries
# ================================

# Standard library
import glob
import json
import os
import shutil
from pathlib import Path

# Third-party library
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")

# ---- Prepare the image dataset -----------------------------------------
# Multimodal retrieval systems require both textual and visual evidence.
# In this step, you'll perform the following operations:
# 1. Download the synthetic recipe image dataset
# 2. Extract the image archive
# 3. Collect image file paths for embedding
# These images will later form the Restaurant & Food Images database, which complements the restaurant text database built from structured data.

# ================================
# Download and prepare image data
# ================================
ZIP_URL = ARCHIVE_URL
ZIP_PATH = "synthetic-recipe-images.zip"
IMG_DIR  = "recipe_images"

download_file(ZIP_URL, ZIP_PATH)
extract_archive(ZIP_PATH, IMG_DIR)

image_paths = sorted(glob.glob(f"{IMG_DIR}/**/*.png", recursive=True))
print(f"✅ Images found: {len(image_paths)}")

# ---- Load the structured data ------------------------------------------
# This module builds directly on the structured outputs produced earlier. Before constructing the multimodal index, load the processed JSON files generated earlier.
# You'll load:
# - Structured restaurant data, which provides the textual retrieval corpus
# - Augmented recipe data, which supplies additional metadata for food items
# These structured records will be transformed into retrieval-ready documents in the next step. Maintaining this pipeline connection ensures the multimodal RAG system remains modular and extensible.

# ================================
# Load structured data from Module 1
# ================================

# NOTE: Ensure the JSON files are in your current working directory.

with open("structured_restaurant_data.json", "r") as f:
    restaurants = json.load(f)

with open("augmented_food_recipe.json", "r") as f:
    recipes = json.load(f)

print(f"✅ Loaded restaurants: {len(restaurants)}")
print(f"✅ Loaded recipes:     {len(recipes)}")

# ---- Initialize embedding models ---------------------------------------
# To support multimodal retrieval, you need to initialize two specialized embedding models.
# The system uses:
# - Sentence-Transformers (384-dimensional) for restaurant text encoding
# - CLIP image encoder (512-dimensional) for food image encoding
# Both embedding outputs are L2-normalized to support cosine similarity search in later retrieval stages. Using modality-appropriate encoders ensures that semantic similarity is preserved within each vector space.

# ================================
# Initialize embedding models
# ================================

# ---- Text embedding model (384-d) ----
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)

print("✅ Text embedder ready")


# ---- Image embedding model (512-d) ----
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

print("✅ Image embedder ready")

# ---- Construct multimodal documents ------------------------------------
# Vector databases operate on structured documents rather than raw files.
# In this step, you'll construct two document sets:
# 1. Article documents derived from structured restaurant data
# 2. Image documents derived from recipe image files
# Each document includes:
# - Retrieval text (page_content)
# - A unique document identifier
# - Metadata fields that enable filtering and constrained search
# These multimodal documents form the foundation of the retrieval index and will be embedded in the next stage.

# ================================
# Build article and image documents
# ================================

# -------- articles --------
article_docs = []

for i, r in enumerate(restaurants):
    name = str(r.get("name", "")).strip()
    if not name:
        continue

    text = (
    f"Restaurant: {name}\n"
    f"Cuisine: {r.get('food_style','')}\n"
    f"Location: {r.get('location','')}"
    )

    # GUARANTEED UNIQUE
    doc_id = f"rest_{i}"

    article_docs.append(
        Document(
            page_content=text.strip(),
            metadata={
                "doc_id": doc_id,
                "cuisine": r.get("food_style"),
                "location": r.get("location"),
                "source": "restaurant",
            },
        )
    )

print("✅ article docs:", len(article_docs))


# -------- images --------
image_docs = []

for i, (p, rec) in enumerate(zip(image_paths, recipes)):
    doc_id = f"img_{i}"

    image_docs.append(
        Document(
            # keeps retrieval results readable
            page_content=rec.get("name", f"recipe image {i}"),
            metadata={
                "doc_id": doc_id,
                "image_path": p,
                "source": "recipe_image",
                "recipe_id": rec.get("id"),
                "cuisine": rec.get("cuisine"),
            },
        )
    )

print("✅ image docs:", len(image_docs))

# ---- Construct the vector index ----------------------------------------
# You'll now construct the retrieval backbone of the multimodal RAG system.
# This step performs the following operations:
# 1. Generate embeddings for both text and image documents
# 2. Create separate Chroma collections for each modality
# 3. Store vectors together with their metadata
# 4. Persist the multimodal index to disk for reuse
# After this step completes, the system is fully prepared for similarity retrieval and metadata filtering in Lesson 2.

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())
if os.path.isdir(DB_DIR):
    shutil.rmtree(DB_DIR)  # Reset vector DB (important for reruns)
# ----- article DB -----
A = embed_texts([d.page_content for d in article_docs])
article_db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)
article_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=A.tolist(),
    documents=[d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs],
)
print("✅ Article DB ready")

# ----- image DB -----
V = embed_images([d.metadata["image_path"] for d in image_docs])
image_db = Chroma(
    collection_name="food_images",
    persist_directory=DB_DIR,
)
image_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in image_docs],
    embeddings=V.tolist(),
    documents=[d.page_content for d in image_docs],
    metadatas=[d.metadata for d in image_docs],
)
print("✅ Image DB ready")
print("🎉 Multimodal Vector Index Construction COMPLETE")

# ---- 📸 Screenshot Submission Requirement ------------------------------
# At the end of this lab, take a screenshot of the final code cell and its output, and name it M2L1_multimodal_vector_index.jpg.
# Your screenshot must clearly show:
# - Successful creation of both Chroma collections:
# - restaurant_articles
# - food_images
# - The final completion message:
# "Multimodal Vector Index Construction COMPLETE"
# This screenshot will later serve as a component for the assessment.

# ---- Conclusion --------------------------------------------------------
# In this lab, you built the multimodal retrieval foundation for the restaurant RAG system. You converted structured restaurant data and food images into retrieval-ready documents, generated embeddings, and constructed persistent vector indexes using Chroma.
# The multimodal index is now prepared for similarity search and metadata filtering.

# ---- Authors -----------------------------------------------------------

# Zikai Dou
