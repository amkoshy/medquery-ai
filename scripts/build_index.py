import os
import json
import uuid
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# === Load Chunks ===
def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks

# === Embedding and FAISS ===
def embed_texts(texts, model_name="BAAI/bge-base-en-v1.5"):
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True)

def build_faiss_index(embeddings, index_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"[✓] FAISS index saved to {index_path}")

# === Main Pipeline ===
def main(input_chunks, faiss_index):
    chunks = load_chunks(input_chunks)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    build_faiss_index(embeddings, faiss_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from document chunks")
    parser.add_argument("--input_chunks", required=True, help="Path to .jsonl chunked input")
    parser.add_argument("--faiss_index", required=True, help="Path to save FAISS index")
    args = parser.parse_args()

    main(args.input_chunks, args.faiss_index)
