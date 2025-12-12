# preprocessor/html_cleaner.py

from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    # HTML parser
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content sections
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()

    # Extract text
    text = soup.get_text()

    # Collapse whitespace
    text = " ".join(text.split())

    return text
