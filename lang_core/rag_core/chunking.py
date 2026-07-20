from ingestion import ingest
import semchunk
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")

class Chunking:
    def __init__(self):
        self.document = ingest()
    
    def chunk(self):
        chunker = semchunk.chunkerify(
            encoder.tokenizer,
            chunk_size=290, 
        )

        full_text, file_name = self.document.pdf_load_single()
        chunks = chunker(full_text)
        return chunks, file_name
    
if __name__ == "__main__":
    chunk = Chunking()

    print(len(chunk.chunk()))