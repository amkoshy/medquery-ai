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
def embed_texts(texts, model_name, batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True
    )
    return embeddings

def build_faiss_index(embeddings, index_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"[✓] FAISS index saved to {index_path}")

def save_metadata(chunks, metadata_path):
    metadata = [{"text": c.get("text", ""), "page": c.get("page"), "doc_title": c.get("doc_title")} for c in chunks]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[✓] Metadata saved to {metadata_path}")

# === Main Pipeline ===
def main(input_chunks, faiss_index, metadata_path=None, model_name=None):
    chunks = load_chunks(input_chunks)
    texts = [c["text"] for c in chunks if "text" in c]

    print(f"[INFO] Loaded {len(texts)} valid chunks")
    embeddings = embed_texts(texts, model_name=model_name)
    print(f"[INFO] Embedding shape: {embeddings.shape}")

    build_faiss_index(embeddings, faiss_index)

    if metadata_path:
        save_metadata(chunks, metadata_path)

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from document chunks")
    parser.add_argument("--input_chunks", required=True, help="Path to .jsonl chunked input")
    parser.add_argument("--faiss_index", required=True, help="Path to save FAISS index")
    parser.add_argument("--metadata_path", help="Optional path to save chunk metadata as JSON")
    parser.add_argument("--model_name", required=True, help="SentenceTransformer model name")
    args = parser.parse_args()

    main(args.input_chunks, args.faiss_index, args.metadata_path, args.model_name)
