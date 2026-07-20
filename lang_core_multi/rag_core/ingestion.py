import pymupdf, os

class ingest:
    def __init__(self):
        self.source = "C:/Projects/agentic_architecture_test/core/raw_docs/Schrödinger_1926_eng.pdf"
        self.path="C:/Projects/agentic_architecture_test/core/raw_docs/"

    def pdf_load_single(self):
        doc = pymupdf.open(self.source)
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