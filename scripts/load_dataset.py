# scripts/load_dataset.py

from typing import Iterable

import os
import gzip
import requests
from warcio.archiveiterator import ArchiveIterator

from preprocessor.pipeline import preprocess_html
from utils.logger import log

LOCAL_WARC_PATH = "CC-MAIN-20180420081400-20180420101400-00000.warc.gz"

WARC_URL = (
    "https://data.commoncrawl.org/"
    "crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/"
    "warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"
)

MAX_DOCS = 5000
MAX_TEXT_CHARS = 20000


# Download WARC if missing
def download_if_missing(
        warc_path: str = LOCAL_WARC_PATH,
        warc_url: str = WARC_URL,
):
    if os.path.exists(warc_path):
        return warc_path

    log("WARC file not found, downloading...", "INFO")
    resp = requests.get(warc_url, stream=True)

    with open(warc_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Download complete: {warc_path}", "INFO")
    return warc_path


def warc_document_texts(warc_path: str, max_text_chars: int = MAX_TEXT_CHARS) -> Iterable[str]:
    with gzip.open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue

            try:
                payload = record.content_stream().read()
                html = payload.decode("utf-8", errors="ignore")
            except:
                continue

            text = preprocess_html(html)
            yield text[:max_text_chars]


def load_warc(
        warc_path: str = LOCAL_WARC_PATH,
        warc_url: str = WARC_URL,
        max_docs: int = MAX_DOCS
) -> list[str]:
    warc_path = download_if_missing(warc_path, warc_url)

    # Parse WARC to text
    log("Parsing dataset...", "INFO")
    documents = []

    for i, text in enumerate(warc_document_texts(warc_path)):
        if i >= max_docs:
            break
        documents.append(text)

    return documents
