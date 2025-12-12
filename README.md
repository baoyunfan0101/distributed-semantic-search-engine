# **distributed-semantic-search-engine** 

## Overview

A distributed semantic search engine pipeline exploring probabilistic data structures (Bloom Filter, Count-Min Sketch, and BERT-based LSH) for scalable crawling, indexing, and searching.

## Author

Yunfan Bao

## Installation

```bash
git clone https://github.com/baoyunfan0101/distributed-semantic-search-engine.git
cd distributed-semantic-search-engine
pip install -r requirements.txt
```

## Project Tree
```
distributed-semantic-search-engine/
│
├── crawler/
│   ├── __init__.py
│   ├── distributed_worker.py   # crawler worker
│   ├── redis_bloom.py          # Redis Bloom filter
│   ├── redis_frontier.py       # Redix frontier
│   └── handlers/               # HTML handlers
│       ├── __init__.py
│       └── save_html.py
│
├── indexer/
│   ├── __init__.py
│   ├── cms_indexer.py          # count-min sketch counter
│   └── local_indexer.py        # local counter
│
├── preprocessor/
│   ├── __init__.py
│   ├── html_cleaner.py
│   ├── pipeline.py             # preprocess pipeline
│   └── text_nomalizer.py
│
├── scripts/
│   ├── __init__.py
│   ├── bert_lsh_experiments.py # experiment 3.1
│   ├── bf_experiments.py       # experiment 1
│   ├── build_bert_lsh.py
│   ├── build_ngram_lsh.py
│   ├── cms_experiments.py      # experiment 2
│   ├── load_dataset.py
│   ├── lsh_contrast_experiments.py # experiment 3.2
│   └── lsh_demo_app.py
│
├── searcher/
│   ├── __init__.py
│   └── lsh.py                  # Locality-Sensitive Hashing
│
├── tokenizer/
│   ├── __init__.py
│   ├── filters.py
│   ├── ngram_tokenizer.py
│   ├── regex_tokenizer.py
│   └── stopwords.txt
│
├── utils/
│   ├── __init__.py
│   └── logger.py
│
├── vectorizer/
│   ├── __init__.py
│   ├── bert_embedder.py        # BERT embedding
│   ├── hyperplane_hasher.py    # Hyperplane hashing
│   └── minhasher.py            # MinHash
│
├── main.py                     # experiments main entry
│
└── README.md
```

## Entry Points

```bash
python main.py
```

## Disclaimer

This project is intended for research and educational purposes only.
It is not designed for clinical use.
