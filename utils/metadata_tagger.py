# utils/metadata_tagger.py

import json
import uuid
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def detect_section_heading(line):
    return line.isupper() and len(line.split()) < 10



def tag_chunks_with_metadata(input_json_path, output_jsonl_path, doc_title):
    with open(input_json_path, "r", encoding="utf-8") as f:
        pages = json.load(f)  # assume format: {"page_1": "text...", "page_2": "text...", ...}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    with open(output_jsonl_path, "w", encoding="utf-8") as out_file:
        for page in pages:
            text = page.get("text", "")
            page_num = page.get("page", None)
            if not text or len(text.strip()) == 0:
                continue

            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "doc_title": doc_title,
                    "page": page_num,
                    "chunk_index": i,
                    "text": chunk
                }
                out_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
