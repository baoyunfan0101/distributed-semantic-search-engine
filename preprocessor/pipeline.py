# preprocessor/pipeline.py

from preprocessor.html_cleaner import clean_html
from preprocessor.text_normalizer import normalize_text


def preprocess_html(html: str):
    # Clean HTML
    cleaned = clean_html(html)

    # Normalize text
    normalized = normalize_text(cleaned)

    return normalized
