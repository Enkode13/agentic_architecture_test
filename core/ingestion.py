import os
from pathlib import Path
import pymupdf


class ingest:
    def __init__(self):
        # Path to the directory where ingestion.py lives: core/
        self.core_dir = Path(__file__).resolve().parent

        # Build paths dynamically relative to core/
        self.path = self.core_dir / "raw_docs"
        self.source = self.path / "Schrödinger_1926_eng.pdf"

    def pdf_load_single(self):
        # Convert Path object to string if library requires a string path
        doc = pymupdf.open(str(self.source))
        full_text = ""

        file_name_only = os.path.basename(doc.name)

        for page in doc:
            # Use "blocks" to maintain a logical reading order
            blocks = page.get_text("blocks")
            for b in blocks:
                # b[4] is the text content
                full_text += b[4] + "\n\n"
        return full_text, file_name_only

if __name__ == "__main__":
    ingesting = ingest()
    print(ingesting.pdf_load_single())