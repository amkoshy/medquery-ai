import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from utils.pdf_extractor import save_extracted_text
from utils.metadata_tagger import tag_chunks_with_metadata

def process_batch(batch, base_name):
    import os
    print(f"[DEBUG] [PID {os.getpid()}] Processing batch with {len(batch)} page(s)")
    result = []
    for page_json in batch:
        print(f"[DEBUG] [PID {os.getpid()}] Tagging page {page_json['page']}")
        chunks = tag_chunks_with_metadata(page_json, doc_title=base_name)
        result.append((chunks, page_json["page"]))
    return result

def main(input_list_path, flag_rewrite=False, max_workers=8, batch_size=4):
    print(f"[INFO] Detected {multiprocessing.cpu_count()} CPU cores on this machine.")
    print(f"[INFO] Using {max_workers} worker processes for parallel batch processing.")
    print(f"[INFO] Batch size per process: {batch_size}")

    with open(input_list_path, "r", encoding="utf-8") as f:
        pdf_dict = json.load(f)
    print(f"[DEBUG] Loaded {len(pdf_dict)} PDF entries from {input_list_path}")

    for entry_idx, entry in enumerate(pdf_dict):
        pdf_path = entry["file"]
        title = entry.get("title", "")
        total_pages = entry.get("total_pages", None)

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
            with ProcessPoolExecutor(max_workers=max_workers) as executor, \
                 open(output_path, "w", encoding="utf-8") as out_f:

                futures = deque()
                page_count = 0
                completed = 0
                max_queue = max_workers * 2  # Limit outstanding futures

                batch = []
                for pj in save_extracted_text(pdf_path, "data/extracted_texts", stream=True):
                    page_count += 1
                    print(f"[DEBUG] Extracted page {pj['page']} (page_count={page_count})")
                    batch.append(pj)
                    if len(batch) == batch_size:
                        print(f"[DEBUG] Submitting batch of {batch_size} page(s) to process pool (queue size before submit: {len(futures)})")
                        futures.append(executor.submit(process_batch, list(batch), base_name))
                        batch.clear()
                    # If too many futures, wait for one to finish
                    if len(futures) >= max_queue:
                        print(f"[DEBUG] Too many outstanding futures ({len(futures)}), waiting for one to finish...")
                        done_future = futures.popleft()
                        try:
                            results = done_future.result()
                            for chunks, current_page in results:
                                for chunk in chunks:
                                    out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                                completed += 1
                                print(f"[INFO] Finished page {current_page} ({completed}/{page_count})")
                        except Exception as e:
                            print(f"[ERROR] Exception in process_batch: {e}")

                # Submit any remaining pages in the last batch
                if batch:
                    print(f"[DEBUG] Submitting final batch of {len(batch)} page(s) to process pool")
                    futures.append(executor.submit(process_batch, list(batch), base_name))

                # Finish remaining futures
                print(f"[DEBUG] Waiting for {len(futures)} remaining futures to finish...")
                while futures:
                    results = futures.popleft().result()
                    for chunks, current_page in results:
                        for chunk in chunks:
                            out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        completed += 1
                        print(f"[INFO] Finished page {current_page} ({completed}/{page_count})")

            print(f"[✓] Done: {output_path}")
        except Exception as e:
            print(f"[✗] Failed to process {pdf_path}: {str(e)}")

if __name__ == "__main__":
    import sys
    # Usage: python -m scripts.ingest_pdf [flag_rewrite] [max_workers] [batch_size]
    flag_rewrite = False
    max_workers = 8
    batch_size = 4
    if len(sys.argv) > 1 and sys.argv[1].lower() == "true":
        flag_rewrite = True
    if len(sys.argv) > 2:
        max_workers = int(sys.argv[2])
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])
    print(f"[DEBUG] Starting main with flag_rewrite={flag_rewrite}, max_workers={max_workers}, batch_size={batch_size}")
    main("data/input_list.json", flag_rewrite, max_workers, batch_size)
