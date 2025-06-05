# medquery-ai





🧠 MedQuery-AI: Retrieval-Augmented Generation (RAG) for Medical QA

This project implements a Retrieval-Augmented Generation (RAG) system to answer domain-specific medical queries using proprietary datasets like Millers Anaesthesia and FRCA exam prep material.

The pipeline performs:

PDF ingestion and text extraction
Chunking with metadata tagging
Embedding with BAAI/BGE
FAISS indexing for dense retrieval
Hybrid retrieval using dense + sparse methods
Answer generation via GPT-4o (OpenAI API)




🔧 Key Commands

🗂️ Step 1: Ingest PDFs and Chunk
python -m scripts.ingest_pdf true 8 4
Parses raw PDFs and splits them into chunks
Uses parallel processing (8 processes, 4 threads per process)


📈 Step 2: Build Optimized FAISS Index
python -m scripts.build_index_optimized \
  --input_chunks data/chunks/Millers_anaesthesia-10th_Edition_chunks.jsonl \
  --faiss_index data/index/Millers_anaesthesia-10th_Edition_bge.index \
  --batch_size 64
Embeds chunks using BAAI/bge-base-en-v1.5
Stores dense vectors in FAISS for fast retrieval


🤖 Step 3: Run RAG Pipeline (Dense Only)
python -m src.rag_pipeline \
  --csv_path queries_answers/queries.csv \
  --jsonl_path data/chunks/FRCA1_chunks.jsonl \
  --index_path data/index/FRCA1.index \
  --top_k 5 \
  --output_json queries_answers/results_frca_2.json \
  --model_name BAAI/bge-base-en-v1.5
Retrieves top-5 dense chunks and passes them to GPT-4o for answer generation


🔀 Step 4: Run Hybrid RAG (Dense + Sparse)
python -m src.rag_pipeline \
  --csv_path queries_answers/queries.csv \
  --jsonl_path data/chunks/FRCA1_chunks.jsonl \
  --index_path data/index/FRCA1.index \
  --top_k 5 \
  --output_json queries_answers/results_hybrid_frca_2.json \
  --model_name BAAI/bge-base-en-v1.5
Uses a hybrid scorer combining FAISS + BM25
Improves recall and accuracy on keyword-heavy queries

📦 Dependencies

Install via:

pip install -r requirements.txt
📁 Project Structure

scripts/        # Ingestion, indexing, and evaluation scripts
src/            # Main RAG pipeline logic
data/           # Chunks, raw PDFs, indexes (excluded in .gitignore)
utils/          # PDF parsing and metadata tagging tools
queries_answers/ # Query datasets and generated outputs


by Abraham Mathew Koshy