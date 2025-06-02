import argparse
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Load and Embed ===
def load_chunks(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def embed_query(query, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode([query])[0]

def search_faiss(index_path, query_vector, top_k):
    index = faiss.read_index(index_path)
    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, top_k)
    return indices[0], distances[0]

def retrieve_top_chunks(query, jsonl_path, index_path, top_k=3):
    chunks = load_chunks(jsonl_path)
    query_vector = embed_query(query)
    top_indices, top_scores = search_faiss(index_path, query_vector, top_k)
    return [chunks[i] for i in top_indices if i < len(chunks)]

# === Prompt & Generation ===
def format_prompt(chunks, query):
    context = "\n".join([f"{i+1}. {chunk['text']}" for i, chunk in enumerate(chunks)])
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}
Answer:"""
    return prompt

def load_llm(model_name="google/flan-t5-base", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def generate_answer(prompt, tokenizer, model, max_tokens=200, device="cpu"):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# === Main RAG Pipeline ===
def run_rag(query, chunk_path, index_path, top_k=3, model_name="google/flan-t5-base", device="cpu"):
    chunks = retrieve_top_chunks(query, chunk_path, index_path, top_k)
    prompt = format_prompt(chunks, query)
    tokenizer, model = load_llm(model_name, device=device)
    answer = generate_answer(prompt, tokenizer, model, device=device)
    return {
        "query": query,
        "chunks": chunks,
        "answer": answer
    }

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline using FAISS + FLAN-T5")
    parser.add_argument("--query", required=True, help="Your natural language question")
    parser.add_argument("--jsonl_path", required=True, help="Path to merged .jsonl file")
    parser.add_argument("--index_path", required=True, help="Path to FAISS index file")
    parser.add_argument("--top_k", type=int, default=3, help="How many chunks to retrieve")
    parser.add_argument("--device", default="cpu", help="Device to use: 'cpu' or 'cuda'")
    args = parser.parse_args()

    result = run_rag(
        query=args.query,
        chunk_path=args.jsonl_path,
        index_path=args.index_path,
        top_k=args.top_k,
        device=args.device
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
