from pymilvus import MilvusClient, db, DataType, Function, FunctionType

class vector_store:
    def __init__(self, db_name="agent_memory", collection_name="agent_RAG", dim=768):
        self.uri = "http://localhost:19530"
        self.token = "root:Milvus"

        self.ensure_db(db_name)
        self.client = MilvusClient(uri = self.uri, token=self.token, db_name=db_name)

        self.ensure_collection(collection_name, dim)
        self.collection = collection_name
        
    def ensure_db(self, db_name):
        admin_client = MilvusClient(uri=self.uri, token=self.token)
        
        # FIX: Use admin_client instead of the 'db' module
        if db_name not in admin_client.list_databases():
            print(f"Creating DB: {db_name}")
            admin_client.create_database(db_name)
        else:
            print(f"Database '{db_name}' already exists.")
            
        admin_client.close()

    def ensure_collection(self, collection_name, dense_dim):
        if not self.client.has_collection(collection_name):
            # 1. Create a Schema
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            
            # 2. Add Fields
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
            # The text field
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
            # The Dense Vector (Semantic)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
            # The Sparse Vector (BM25/Keywords)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            
            bm25_function = Function(
                name="text_bm25_emb",
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)

            # 3. Create Index (Required for Sparse Vectors)
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
            index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
            
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
                auto_id=True
            )
        else:
            print(f"Collection {collection_name} already exits!")

    
    def add_or_update_collection(self, data):
        return self.client.insert(collection_name=self.collection, data=data)
    
    def drop_collection(self, collection_name):
        """
        Permanently deletes a collection and all its data.
        """
        if self.client.has_collection(collection_name):
            print(f"Dropping collection: {collection_name}...")
            self.client.drop_collection(collection_name)
            return True
        else:
            print(f"Collection '{collection_name}' does not exist. Nothing to drop.")
            return False