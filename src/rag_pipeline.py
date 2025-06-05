import argparse
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import openai
import os
import datetime
from rank_bm25 import BM25Okapi

# === Caching Models ===
@lru_cache(maxsize=1)
def get_embedder(model_name):
    """Cache and return embedding model."""
    return SentenceTransformer(model_name)

# === Load Data ===
def load_chunks(jsonl_path):
    """Load chunks from a .jsonl file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def embed_query(query, model_name):
    """Convert query to dense embedding."""
    embedder = get_embedder(model_name)
    return embedder.encode([query])[0]

def search_faiss(index_path, query_vector, top_k):
    """Search the FAISS index for top-k similar vectors."""
    index = faiss.read_index(index_path)
    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, top_k)
    return indices[0], distances[0]

def retrieve_top_chunks(query, jsonl_path, index_path, top_k=3, model_name=None):
    """Retrieve top-k text chunks most relevant to the query."""
    chunks = load_chunks(jsonl_path)
    query_vector = embed_query(query, model_name)
    top_indices, _ = search_faiss(index_path, query_vector, top_k)
    return [chunks[i] for i in top_indices ]



def retrieve_top_chunks(query, jsonl_path, index_path, alpha, top_k=3, model_name=None):
    """Retrieve top-k text chunks using hybrid search: dense (FAISS) + sparse (BM25)."""
    chunks = load_chunks(jsonl_path)  # list of dicts with 'text' field
    texts = [chunk["text"] for chunk in chunks]

    # ----- Dense Search (FAISS) -----
    query_vector = embed_query(query, model_name)
    top_indices_dense, dense_scores = search_faiss(index_path, query_vector, len(texts))  # get all scores

    # ----- Sparse Search (BM25) -----
    tokenized_corpus = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    sparse_scores = bm25.get_scores(query.split())  # shape: (len(chunks),)

    # ----- Normalize + Combine -----
    dense_scores_norm = dense_scores / np.linalg.norm(dense_scores) if np.linalg.norm(dense_scores) > 0 else dense_scores
    sparse_scores_norm = sparse_scores / np.linalg.norm(sparse_scores) if np.linalg.norm(sparse_scores) > 0 else sparse_scores

    hybrid_scores = alpha * dense_scores_norm + (1 - alpha) * sparse_scores_norm
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices if i < len(chunks)]


# === Prompting ===
def format_prompt(chunks, query):
    """Prepare the prompt using retrieved chunks and the query."""
    context = "\n".join([f"{i+1}. {chunk['text']}" for i, chunk in enumerate(chunks)])
    prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    return prompt

def generate_rag_prompt(query, retrieved_chunks):
    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    prompt = f"""Use the following chunks to try answering the question. If they contain partially useful or indirectly related information, make reasonable inferences. Say "No relevant information found" only if truly nothing helps.

### Context:
{context}

### Question:
{query}

### Answer:"""
    return prompt

def generate_answer_openai(prompt, model="gpt-4o", max_tokens=512):
    """Generate answer using OpenAI GPT-4o and print token usage (OpenAI v1.x)."""
    client = openai.OpenAI()  # Uses env var OPENAI_API_KEY
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens
    )
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    print(f"    [OpenAI] input_tokens: {input_tokens}, output_tokens: {output_tokens}")
    return response.choices[0].message.content, input_tokens, output_tokens

# === RAG Pipeline ===
def run_rag(query, chunk_path, index_path, top_k=3, model_name=None):
    chunks = retrieve_top_chunks(query, chunk_path, index_path, 0.7, top_k, model_name)
    prompt = generate_rag_prompt(query, chunks)
    answer, input_tokens, output_tokens = generate_answer_openai(prompt)
    enriched_chunks = [
        {"text": c["text"], "page": c.get("page"), "doc_title": c.get("doc_title")} for c in chunks
    ]
    return {
        "query": query,
        "chunks": enriched_chunks,
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch RAG using OpenAI GPT-4o")
    parser.add_argument("--csv_path", required=True, help="Path to input CSV with a 'query' column")
    parser.add_argument("--jsonl_path", required=True, help="Path to merged .jsonl file")
    parser.add_argument("--index_path", required=True, help="Path to FAISS index file")
    parser.add_argument("--top_k", type=int, default=3, help="How many chunks to retrieve")
    parser.add_argument("--output_json", help="Path to output JSON file (default: results<top_k>.json)")
    parser.add_argument("--model_name", required=True, help="SentenceTransformer model name for embedding")
    args = parser.parse_args()

    if not args.output_json:
        args.output_json = f"results{args.top_k}.json"

    # Read queries from the specified CSV path
    df = pd.read_csv(args.csv_path)

    # Load existing results if output_json exists
    if os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except Exception:
                existing_results = []
    else:
        existing_results = []

    results = existing_results
    total_input_tokens = 0
    total_output_tokens = 0

    for i, row in df.iterrows():
        query = row["query"]
        print(f"[INFO] Processing query {i+1}/{len(df)}: {query}")
        result = run_rag(
            query=query,
            chunk_path=args.jsonl_path,
            index_path=args.index_path,
            top_k=args.top_k,
            model_name=args.model_name
        )
        total_input_tokens += result.get("input_tokens", 0)
        total_output_tokens += result.get("output_tokens", 0)
        print(f"    [Cumulative] input_tokens: {total_input_tokens}, output_tokens: {total_output_tokens}")
        results.append(result)

    # Write results to the specified output location
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} results to {args.output_json}")

    # === Log OpenAI usage ===
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"{now} | input_tokens: {total_input_tokens}, output_tokens: {total_output_tokens}, "
        f"queries: {len(df)}, output_json: {args.output_json}\n"
    )
    with open("openai_usage.txt", "a", encoding="utf-8") as logf:
        logf.write(log_entry)
    print(f"📝 Usage logged to openai_usage.txt")
