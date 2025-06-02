# utils/pdf_extractor.py

import pdfplumber
import os
import json

def extract_text_from_pdf(pdf_path):
    extracted_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                extracted_pages.append({
                    "page": page_num,
                    "text": text.strip()
                })
    return extracted_pages

def save_extracted_text(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.json")

    pages = extract_text_from_pdf(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    
    print(f"[✓] Extracted text saved to {output_path}")
    return output_path
