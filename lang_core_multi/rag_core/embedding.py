from chunking import Chunking
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

class embed:
    def __init__(self):
        self.chunk = Chunking()

    def embeddings(self):
        embedded_chunks = []
        documents, doc_name = self.chunk.chunk()
        all_vectors = model.encode(documents, convert_to_numpy=True).tolist()  # convert to list for ChromaDB
        for i, (chunk, vector) in enumerate(zip(documents, all_vectors)):
            embedded_chunks.append({
                "text": chunk,              # Matches your schema VARCHAR field
                "dense_vector": vector,    # Matches your schema FLOAT_VECTOR field
                "doc_name": doc_name,
            })
        # print(embedded_chunks[0]["chunk"])
        return embedded_chunks

if __name__ == "__main__":
    embeding = embed()
    print(embeding.embeddings())