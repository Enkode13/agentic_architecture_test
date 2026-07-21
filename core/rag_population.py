from core.vectordb import vector_store
from core.embedding import embed

vs = vector_store()
vs.drop_collection("agent_RAG")

vs = vector_store()
embedding_instance = embed()

chunks = embedding_instance.embeddings()

vs.add_or_update_collection(chunks)