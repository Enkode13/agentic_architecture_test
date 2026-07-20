from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

from lang_core_multi.rag_core.vectordb import vector_store


class Retrieval:
    """Hybrid (dense + sparse BM25) retrieval over the Milvus vector store."""

    def __init__(
        self,
        collection_name: str = "agent_RAG",
        dense_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.store = vector_store()
        self.collection_name = collection_name
        self.dense_ef = SentenceTransformerEmbeddingFunction(
            model_name=dense_model,
            device="cpu",
            query_instruction="Represent this sentence for searching relevant passages:",
        )
        self.sparse_ef = BM25EmbeddingFunction()

    def hybrid_search(self, query: str, top_k: int = 5) -> list[str]:
        dense_vec = self.dense_ef.encode_queries([query])

        res_dense = AnnSearchRequest(
            data=dense_vec,
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=top_k,
        )
        res_sparse = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=top_k,
        )

        results = self.store.client.hybrid_search(
            collection_name=self.store.collection,
            reqs=[res_dense, res_sparse],
            ranker=RRFRanker(k=60.0),
            limit=top_k,
            output_fields=["text"],
        )

        return [hit.get("entity").get("text") for hits in results for hit in hits]


if __name__ == "__main__":
    query = "What is the goal of this paper?"
    retrieve = Retrieval()
    context = retrieve.hybrid_search(query, 3)
    print(context)
