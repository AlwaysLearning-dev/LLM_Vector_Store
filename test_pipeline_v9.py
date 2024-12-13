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
            # Initialize Qdrant client with new host
            self.qdrant_client = QdrantClient(
                host="qdrant", 
                port=6333,
                timeout=10.0
            )
            logger.info("Qdrant client initialized")
            
            self.collection_name = "sigma_rules"
            
            # Initialize embedding model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_size = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer initialized with vector size: {self.vector_size}")
            
            # Ensure collection exists with proper configuration
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def _ensure_collection(self) -> None:
        """Ensure the collection exists with proper vector configuration."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name}")
                self.qdrant_client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "size": self.vector_size,
                        "distance": "Cosine"
                    }
                )
            else:
                # Get collection info to verify configuration
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"Collection info: {collection_info}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}", exc_info=True)
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
