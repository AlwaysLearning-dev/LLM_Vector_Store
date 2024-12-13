import logging
import sys
from typing import List, Dict, Any, Generator

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
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection('sigma_rules')
            logger.info(f"Connected to sigma_rules collection")
            
            # Initialize embedding model that outputs 3072-dimensional vectors
            self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')
            logger.info(f"SentenceTransformer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """
        OpenWebUI compatible pipe method that yields results as a stream.
        
        Args:
            prompt (str, optional): The input query/prompt
            **kwargs: Additional keyword arguments from OpenWebUI
                user_message (str): The user's message
                
        Yields:
            str: Results as a stream of text
        """
        try:
            # Get the query from either prompt or user_message
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided in either prompt or user_message")
                
            logger.info(f"Pipe processing query: {query}")

            # Generate embeddings
            query_vector = self.embedder.encode(query).tolist()
            logger.debug(f"Generated query embeddings with size {len(query_vector)}")

            # Perform search with named vector parameter
            response = self.qdrant_client.search(
                collection_name="sigma_rules",
                query_vector={
                    "default": query_vector  # Use 'default' as the vector name
                },
                limit=5
            )
            logger.debug(f"Received {len(response)} results from Qdrant")

            # Stream results
            if response:
                yield "Based on the search, here are the relevant Sigma rules:\n\n"
                for idx, res in enumerate(response, 1):
                    # Format the payload nicely
                    if isinstance(res.payload, dict):
                        result_text = f"Result {idx}:\n"
                        for key, value in res.payload.items():
                            result_text += f"{key}: {value}\n"
                        result_text += "\n"
                    else:
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
            results = list(self.pipe(prompt=prompt, **kwargs))
            return [{"text": "".join(results)}]
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)
            return [{"error": str(e)}]

# For local testing
if __name__ == "__main__":
    try:
        pipeline = Pipeline()
        test_query = "Detect lateral movement with SMB"
        
        print("\nTesting pipe method with prompt:")
        for chunk in pipeline.pipe(prompt=test_query):
            print(chunk, end="", flush=True)
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
