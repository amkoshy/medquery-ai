import os
import json
from sentence_transformers import SentenceTransformer, util
import torch

# === Config ===
embedding_model_name = "BAAI/bge-base-en-v1.5"
input_json_path = "queries_answers/results_hybrid_frca_2.json"  # Update to your actual path
output_log_path = input_json_path.replace(".json", "_rag_analysis_log_.txt")

# === Load Embedding Model ===
print(f"[INFO] Loading embedding model: {embedding_model_name}")
model = SentenceTransformer(embedding_model_name)

# === Load RAG Result JSON ===
print(f"[INFO] Loading RAG results from: {input_json_path}")
with open(input_json_path, "r") as f:
    rag_data = json.load(f)

# === Open log file for writing output ===
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

# === Analyze each entry ===
for entry_index, entry in enumerate(rag_data):
    query = entry.get("query", "")
    answer = entry.get("answer", "")
    chunks = entry.get("chunks", [])

    log("=" * 80)
    log(f"[{entry_index+1}] QUERY: {query}")
    log("-" * 80)
    log("Answer:")
    log(answer)

    # Split answer into sentences
    answer_sentences = [s.strip() for s in answer.split("\n") if s.strip()]
    chunk_texts = [chunk["text"] for chunk in chunks]

    if not answer_sentences or not chunk_texts:
        log("[WARNING] Missing sentences or chunks. Skipping...")
        continue

    # Encode sentences and chunks
    log("[INFO] Encoding answer sentences and context chunks...")
    answer_embeddings = model.encode(answer_sentences, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)

    # Compare each answer sentence to all chunks
    for i, sent_emb in enumerate(answer_embeddings):
        sims = util.cos_sim(sent_emb, chunk_embeddings)
        max_sim = torch.max(sims).item()
        log(f"\nSentence {i+1}: \"{answer_sentences[i]}\"")
        log(f"  → Max Similarity to any chunk: {max_sim:.4f}")
        if max_sim < 0.6:
            log("  ⚠️  Possibly hallucinated or weakly grounded.")
        elif max_sim < 0.8:
            log("  ⚠️  Partially grounded.")
        else:
            log("  ✅ Clearly grounded in context.")

# === Write log to file ===
with open(output_log_path, "w") as f:
    f.write("\n".join(log_lines))

log(f"\n[INFO] Analysis complete. Log written to: {output_log_path}")
