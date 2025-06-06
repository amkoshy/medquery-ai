# src/medical_explainer_agent.py

import os
import json
import argparse
import datetime
import openai
import pandas as pd
import numpy as np
from typing import TypedDict, Optional, List
from functools import lru_cache
from langgraph.graph import StateGraph, END
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util as st_util
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# Load once, re-use globally
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------
# Embedding and Retrieval
# -------------------------------

@lru_cache(maxsize=1)
def get_embedder(model_name):
    return SentenceTransformer(model_name)

def load_chunks(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def embed_query(query, model_name):
    embedder = get_embedder(model_name)
    return embedder.encode([query])[0]

def embed_text(text, model_name):
    embedder = get_embedder(model_name)
    return embedder.encode([text])[0]

def search_faiss(index_path, query_vector, top_k):
    index = faiss.read_index(index_path)
    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, top_k)
    return indices[0], distances[0]

def retrieve_top_chunks(query, jsonl_path, index_path, alpha, top_k, model_name):
    chunks = load_chunks(jsonl_path)
    texts = [chunk["text"] for chunk in chunks]

    query_vector = embed_query(query, model_name)
    top_indices_dense, dense_scores = search_faiss(index_path, query_vector, len(texts))

    tokenized_corpus = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    sparse_scores = bm25.get_scores(query.split())

    dense_scores_norm = dense_scores / np.linalg.norm(dense_scores) if np.linalg.norm(dense_scores) > 0 else dense_scores
    sparse_scores_norm = sparse_scores / np.linalg.norm(sparse_scores) if np.linalg.norm(sparse_scores) > 0 else sparse_scores

    hybrid_scores = alpha * dense_scores_norm + (1 - alpha) * sparse_scores_norm
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices if i < len(chunks)], query_vector

# -------------------------------
# Prompting and LLM Answering
# -------------------------------

def generate_rag_prompt(query, chunks):
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    return f"""Use the following chunks to answer the question. If partially useful, make reasonable inferences. Say \"No relevant information found\" only if truly nothing helps.

### Context:
{context}

### Question:
{query}

### Answer:"""

def generate_answer_openai(prompt, model="gpt-4o", max_tokens=512):
    client = openai.OpenAI()
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
    return response.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens

# -------------------------------
# RAG Pipeline
# -------------------------------

def compute_similarity(query_vec, chunk_texts, answer, model_name):
    chunk_vecs = [embed_text(text, model_name) for text in chunk_texts]
    answer_vec = embed_text(answer, model_name)
    cos_sim_chunks = [float(cosine_similarity([query_vec], [v])[0][0]) for v in chunk_vecs]
    semantic_sim = float(cosine_similarity([query_vec], [answer_vec])[0][0])
    return cos_sim_chunks, semantic_sim

def run_rag(query, chunk_path, index_path, top_k, model_name):
    chunks, query_vector = retrieve_top_chunks(query, chunk_path, index_path, 0.7, top_k, model_name)
    prompt = generate_rag_prompt(query, chunks)
    answer, input_tokens, output_tokens = generate_answer_openai(prompt)

    chunk_texts = [chunk["text"] for chunk in chunks]
    cos_sim_chunks, semantic_sim = compute_similarity(query_vector, chunk_texts, answer, model_name)


    return {
        "query": query,
        "chunks": chunks,
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cosine_similarities": cos_sim_chunks,
        "semantic_similarity": semantic_sim
    }

# -------------------------------
# LangGraph Agent Definition
# -------------------------------

class AgentState(TypedDict):
    query: str
    answer: Optional[str]
    citations: Optional[str]
    rag_output: Optional[dict]
    confidence: Optional[float]
    reranked_chunks: Optional[List[dict]]

def build_app(chunk_path, index_path, top_k, model_name):
    def node_rag_query(state: AgentState) -> AgentState:
        result = run_rag(state["query"], chunk_path, index_path, top_k, model_name)
        result["model_name"] = model_name
        return {
            **state,
            "answer": result["answer"],
            "rag_output": result,
            "confidence": result["semantic_similarity"]
        }

    def node_explain_citations(state: AgentState) -> AgentState:
        query, answer = state["query"], state["answer"]
        prompt = f"""
You are an expert assistant reviewing how a RAG system used retrieved text chunks to answer a query.

Question: {query}
Answer: {answer}

Please explain, for each major sentence or point in the answer, which chunk(s) most likely contributed to it.
Use bullet points like:
- [Sentence]: came from Chunk X
"""
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return {**state, "citations": response.choices[0].message.content}

    def rerank_chunks_with_cross_encoder(state: AgentState) -> AgentState:
        query = state["query"]
        original_chunks = state["rag_output"]["chunks"]

        if not original_chunks:
            return state

        # Step 1: Re-rank with cross-encoder
        inputs = [(query, chunk["text"]) for chunk in original_chunks]
        scores = cross_encoder.predict(inputs)
        reranked_chunks = [x for _, x in sorted(zip(scores, original_chunks), reverse=True)]

        # Step 2: Generate a new answer using reranked chunks
        prompt = generate_rag_prompt(query, reranked_chunks)
        answer, input_tokens, output_tokens = generate_answer_openai(prompt)

        # Step 3: Recompute similarity
        chunk_texts = [chunk["text"] for chunk in reranked_chunks]
        query_vec = embed_query(query, model_name=state["rag_output"]["model_name"])
        cos_sim_chunks, semantic_sim = compute_similarity(query_vec, chunk_texts, answer, model_name=state["rag_output"]["model_name"])

        # Step 4: Return updated state
        return {
            **state,
            "answer": answer,
            "confidence": semantic_sim,
            "rag_output": {
                "query": query,
                "chunks": reranked_chunks,
                "answer": answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cosine_similarities": cos_sim_chunks,
                "semantic_similarity": semantic_sim,
                "model_name": state["rag_output"]["model_name"]
            }
        }


    def check_confidence(state: AgentState) -> str:
        rag_output = state.get("rag_output", {})
        chunk_sims = rag_output.get("cosine_similarities", [])
        citations = rag_output.get("citations", [])

        if not chunk_sims:
            print("[DEBUG] No chunk similarities found. Reranking...")
            return "low"

        max_sim = max(chunk_sims)
        avg_sim = sum(chunk_sims) / len(chunk_sims)
        
        print(f"[DEBUG] Max Chunk Similarity: {max_sim:.3f}")
        print(f"[DEBUG] Avg Chunk Similarity: {avg_sim:.3f}")
        
        # Add this block here
        if max_sim < 0.5:
            print("[DEBUG] Query too far from knowledge base. Returning fallback.")
            state["answer"] = "No relevant information found in the current knowledge base."
            state["citations"] = "No citations available."
            return "done"  # This must route to a terminal node


    def node_fallback(state: AgentState) -> AgentState:
        return state

    graph = StateGraph(AgentState)

    # Define all nodes
    graph.add_node("RAGQuery", node_rag_query)
    graph.add_node("ExplainCitations", node_explain_citations)
    graph.add_node("RerankChunks", rerank_chunks_with_cross_encoder)

    # Add fallback node to safely exit when query is too far from knowledge base
    def node_fallback(state: AgentState) -> AgentState:
        return state
    graph.add_node("Fallback", node_fallback)

    # Set entry point
    graph.set_entry_point("RAGQuery")

    # Conditional routing based on chunk similarity and presence of citations
    graph.add_conditional_edges("RAGQuery", check_confidence, {
        "good": "ExplainCitations",
        "low": "RerankChunks",
        "done": "Fallback"
    })

    # Fixed transitions
    graph.add_edge("RerankChunks", "ExplainCitations")
    graph.add_edge("ExplainCitations", END)
    graph.add_edge("Fallback", END)

    return graph.compile()


# -------------------------------
# CLI Mode
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    rag_agent = build_app(args.jsonl_path, args.index_path, args.top_k, args.model_name)
    df = pd.read_csv(args.csv_path)
    output_path = args.output_json or "rag_results.json"

    results = []
    for i, row in df.iterrows():
        query = row["query"]
        print(f"[INFO] Processing query {i+1}/{len(df)}: {query}")
        output = rag_agent.invoke({"query": query})
        result = {
            "query": query,
            "answer": output["answer"],
            "citations": output.get("citations", "No citations available."),
            **output.get("rag_output", {})
        }
        results.append(result)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved to {output_path}")
