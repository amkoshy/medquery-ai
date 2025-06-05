# utils/metadata_tagger.py
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter



def detect_section_heading(line):
    return line.isupper() and len(line.split()) < 10

def tag_chunks_with_metadata(page_json, doc_title=""):
    text = page_json.get("text", "")
    page_num = page_json.get("page", None)

    if not text.strip():
        return []

    # Detect section headings from text
    lines = text.splitlines()
    section_heading = None
    annotated_text = ""

    for line in lines:
        if detect_section_heading(line.strip()):
            section_heading = line.strip()
        annotated_text += line + "\n"

    # Prepare text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_text(annotated_text)
    chunk_data = []

    for i, chunk in enumerate(chunks):
        chunk_with_heading = f"[SECTION: {section_heading}]\n{chunk}" if section_heading else chunk
        metadata = {
            "text": chunk_with_heading.strip(),
            "doc_title": doc_title,
            "page": page_num,
            "chunk_index": i,
            "section_heading": section_heading,
            "text_length": len(chunk),
            "chunk_id": f"{doc_title}_p{page_num}_c{i}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        chunk_data.append(metadata)

    return chunk_data
