try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Required packages not found. Please install them using:\n"
        "pip install qdrant-client sentence-transformers"
    )

class Pipeline:
    def __init__(self):
        """
        Initialize the pipeline with Qdrant and SentenceTransformer.
        Raises:
            ConnectionError: If cannot connect to Qdrant
            ImportError: If required packages are not installed
        """
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(host="qdrant", port=6333)
            # Test connection
            self.qdrant_client.get_collections()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")

        try:
            # Initialize SentenceTransformer for query embeddings
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise ImportError(f"Failed to initialize SentenceTransformer: {str(e)}")

        self.collection_name = "sigma_rules"

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
            query_vector = self.embedder.encode(query).tolist()

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
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return [error_msg]

# For local testing
if __name__ == "__main__":
    try:
        pipeline = Pipeline()
        test_query = "Detect lateral movement with SMB"
        results = pipeline.run(test_query)
        for result in results:
            print(result)
    except Exception as e:
        print(f"Pipeline initialization failed: {str(e)}")
