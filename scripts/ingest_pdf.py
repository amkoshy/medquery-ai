import json
import os
from utils.pdf_extractor import save_extracted_text
from utils.metadata_tagger import tag_chunks_with_metadata

def main(input_list_path, flag_rewrite=False):
    with open(input_list_path, "r", encoding="utf-8") as f:
        pdf_dict = json.load(f)

    for entry in pdf_dict:
        pdf_path = entry["file"]
        title = entry.get("title", "")

        if not os.path.exists(pdf_path):
            print(f"[✗] File not found: {pdf_path}")
            continue

        base_name = title or os.path.splitext(os.path.basename(pdf_path))[0]
        os.makedirs("data/chunks", exist_ok=True)
        output_path = f"data/chunks/{base_name}_chunks.jsonl"

        if os.path.exists(output_path) and not flag_rewrite:
            print(f"[!] Skipping chunking (already exists): {output_path}")
            continue

        print(f"[→] Processing: {pdf_path}  Title: {title}")

        try:
            # Step 1: Extract raw text per page
            json_path = save_extracted_text(pdf_path, "data/extracted_texts")

            # Step 2: Chunk + tag metadata
            tag_chunks_with_metadata(
                input_json_path=json_path,
                output_jsonl_path=output_path,
                doc_title=base_name
            )

            print(f"[✓] Done: {output_path}")
        except Exception as e:
            print(f"[✗] Failed to process {pdf_path}: {str(e)}")

if __name__ == "__main__":
    import sys
    # Usage: python -m scripts.ingest_pdf [flag_rewrite]
    flag_rewrite = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "true":
        flag_rewrite = True
    main("data/input_list.json", flag_rewrite)
