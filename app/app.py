# app/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import json
from src.medical_explainer_agent import build_app

st.set_page_config(page_title="Medical RAG QA", layout="wide")
st.title("🩺 Medical RAG Agent")

# --- Parameters ---
chunk_path = "data/chunks/FRCA1_chunks.jsonl"
index_path = "data/index/FRCA1.index"
model_name = "BAAI/bge-base-en-v1.5"
top_k = 5

rag_agent = build_app(chunk_path, index_path, top_k, model_name)

# --- Upload CSV ---
st.markdown("### Upload CSV file with queries")
uploaded_file = st.file_uploader("CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    if st.button("Run RAG Agent"):
        results = []
        with st.spinner("Processing queries..."):
            for i, row in df.iterrows():
                query = row["query"]
                st.markdown(f"#### Query {i+1}: {query}")
                output = rag_agent.invoke({"query": query})
                result = {
                    "query": query,
                    "answer": output["answer"],
                    "citations": output["citations"],
                    **output["rag_output"]
                }
                results.append(result)

        st.success("🎉 Done!")
        for r in results:
            st.markdown(f"### Answer: {r['query']}")
            st.write(r["answer"])
            st.markdown("**Citations:**")
            st.code(r["citations"])
            st.markdown("**Top Chunks:**")
            for c in r["chunks"]:
                st.code(c["text"][:600])
            st.markdown(f"**Semantic Similarity:** {r['semantic_similarity']:.3f}")
            st.markdown("---")

        st.download_button("📥 Download Results", json.dumps(results, indent=2), file_name="rag_results.json")

