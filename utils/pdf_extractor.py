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

def save_extracted_text(pdf_path, output_dir, stream=True):
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            yield {"page": page_num, "text": text if text else ""}
