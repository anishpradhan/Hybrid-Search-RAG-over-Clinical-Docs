"""Download the public-domain CDC MMWR guideline documents used by the eval.
"""
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.cdc.gov/mmwr/volumes/{}.htm"
DOCS = {
    "hepatitis_b_screening_2023": "72/rr/rr7201a1",
    "hepatitis_c_screening_2020": "69/rr/rr6902a1",
    "dengue_vaccine_2021": "70/rr/rr7006a1",
    "organ_donor_screening_2020": "69/rr/rr6904a1",
}
OUT = Path(__file__).parent / "corpus"


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main") or soup.body
    for tag in main(["script", "style", "nav", "footer", "header", "form", "button", "figure"]):
        tag.decompose()
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
        if el.name == "li" and el.find("p"):
            continue  # paragraph inside already captured
        if el.name == "tr":
            text = " | ".join(c.get_text(" ", strip=True) for c in el.find_all(["td", "th"]))
        else:
            text = el.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 2:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append(f"\n{text}\n")
        else:
            lines.append(text)
    text = "\n".join(lines)
    # drop the reference list: everything from a final "References" heading
    idx = text.rfind("\nReferences\n")
    if idx > len(text) * 0.5:
        text = text[:idx]
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    for name, path in DOCS.items():
        r = requests.get(BASE.format(path), timeout=60, headers={"User-Agent": "rag-eval-corpus-fetch"})
        r.raise_for_status()
        r.encoding = "utf-8"
        text = html_to_text(r.text)
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"{name}: {len(text):,} chars", file=sys.stderr)


if __name__ == "__main__":
    main()
