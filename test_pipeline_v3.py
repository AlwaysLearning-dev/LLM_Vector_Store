from llama_index import VectorStoreIndex
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

class Pipeline:
    def __init__(self):
        # Initialize Qdrant client and SentenceTransformer
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "sigma_rules"
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def run(self, query):
        """
        Process the query, search Qdrant for relevant results, and return them.
        """
        print(f"Received query: {query}")

        # Convert the query into a vector
        query_vector = self.embedder.encode(query)
        print(f"Query vector generated: {query_vector}")

        # Define search filter
        search_filter = Filter(
            must=[FieldCondition(key="tags", match=MatchValue(value="sigma"))]
        )

        # Search Qdrant
        response = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=5,
            query_filter=search_filter
        )

        # Extract and return results
        if response:
            results = [res.payload for res in response]
            print(f"Found results: {results}")
            return results
        else:
            print("No matching Sigma rules found.")
            return ["No matching Sigma rules found."]

# Test the script locally
if __name__ == "__main__":
    pipeline = Pipeline()
    test_query = "Detect lateral movement with SMB"
    results = pipeline.run(test_query)
    for result in results:
        print(result)
