# utils/logger.py

import sys


def log(msg: str, tag: str = "INFO", file=sys.stdout):
    print(f"[{tag}] {msg}", file=file)


def log_section(title: str):
    print(f"\n====== {title} ======")
