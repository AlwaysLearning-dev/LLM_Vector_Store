import logging
import sys
from typing import List, Dict, Any, Generator, Union

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer
    logger.info("Successfully imported required packages")
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    raise

class Pipeline:
    def __init__(self) -> None:
        """Initialize the pipeline with required components."""
        logger.info("Initializing Pipeline")
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host="qdrant", 
                port=6333,
                timeout=10.0
            )
            logger.info("Qdrant client initialized")
            
            self.collection_name = "sigma_rules"
            
            # Initialize embedding model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def pipe(self, prompt: str) -> Generator[str, None, None]:
        """
        OpenWebUI compatible pipe method that yields results as a stream.
        
        Args:
            prompt (str): The input query/prompt
            
        Yields:
            str: Results as a stream of text
        """
        logger.info(f"Pipe received prompt: {prompt}")
        try:
            # Generate embeddings
            query_vector = self.embedder.encode(prompt).tolist()
            logger.debug("Generated query embeddings")

            # Create search filter
            search_filter = Filter(
                must=[FieldCondition(key="tags", match=MatchValue(value="sigma"))]
            )

            # Perform search
            response = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=5,
                query_filter=search_filter
            )
            logger.debug(f"Received {len(response)} results from Qdrant")

            # Stream results
            if response:
                yield "Here are the relevant Sigma rules:\n\n"
                for idx, res in enumerate(response, 1):
                    result_text = f"Result {idx}:\n{res.payload}\n\n"
                    yield result_text
            else:
                yield "No matching Sigma rules found."

        except Exception as e:
            logger.error(f"Error in pipe method: {e}", exc_info=True)
            yield f"Error processing query: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Legacy method for non-streaming results.
        """
        logger.info(f"Run method called with prompt: {prompt}")
        try:
            results = list(self.pipe(prompt))
            return [{"text": "".join(results)}]
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)
            return [{"error": str(e)}]

# For local testing
if __name__ == "__main__":
    try:
        pipeline = Pipeline()
        test_query = "Detect lateral movement with SMB"
        
        print("\nTesting pipe method (streaming):")
        for chunk in pipeline.pipe(test_query):
            print(chunk, end="", flush=True)
            
        print("\nTesting run method (non-streaming):")
        results = pipeline.run(test_query)
        for result in results:
            print(result)
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
