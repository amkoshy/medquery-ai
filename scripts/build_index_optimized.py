
import argparse
import jsonlines
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from tqdm import tqdm

def auto_batch_size():
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        if total_mem >= 20 * 1024**3:
            return 256
        elif total_mem >= 10 * 1024**3:
            return 128
        else:
            return 64
    except:
        return 32  # Fallback for CPU

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_chunks", type=str, required=True)
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    args = parser.parse_args()

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = SentenceTransformer(args.model_name, device=device)

    # Read input chunks
    texts = []
    with jsonlines.open(args.input_chunks) as reader:
        for obj in reader:
            texts.append(obj["text"])

    # Set batch size
    batch_size = args.batch_size or auto_batch_size()
    print(f"Using batch size: {batch_size}")

    # Encode in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i+batch_size]
        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            device=device,
            convert_to_numpy=True,
            normalize_embeddings=False  # Keep False for L2 FAISS
        )
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)

    # Create FAISS index
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(all_embeddings)

    # Save index
    faiss.write_index(index, args.faiss_index)
    print(f"Index saved to: {args.faiss_index}")

if __name__ == "__main__":
    main()
