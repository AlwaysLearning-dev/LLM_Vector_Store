from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

class Pipeline:
    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "sigma_rules"

        # Initialize SentenceTransformer for query embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def run(self, query):
        """
        Process the query, search Qdrant for relevant results, and return them.
        Args:
            query (str): The input query to process
        Returns:
            list: List of relevant Sigma rules found in Qdrant
        """
        try:
            print(f"Processing query: {query}")

            # Embed the query using SentenceTransformer
            query_vector = self.embedder.encode(query).tolist()  # Convert to list for JSON serialization

            # Search Qdrant with a filter to restrict results to Sigma rules
            search_filter = Filter(
                must=[FieldCondition(key="tags", match=MatchValue(value="sigma"))]
            )

            # Perform the search
            response = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=5,
                query_filter=search_filter
            )

            # Extract and format results
            if response:
                results = [res.payload for res in response]
                print(f"Search results: {results}")
                return results
            else:
                return ["No matching Sigma rules found."]

        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return [f"Error processing query: {str(e)}"]

# For local testing
if __name__ == "__main__":
    pipeline = Pipeline()
    test_query = "Detect lateral movement with SMB"
    results = pipeline.run(test_query)
    for result in results:
        print(result)
