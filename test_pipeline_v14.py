import logging
import sys
from typing import List, Dict, Any, Generator
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from transformers import AutoTokenizer, AutoModel
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
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Initialize the model and tokenizer
            model_path = "/home/bob/llama/"  # Use the same model path as in your ingestion script
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True
            ).to(self.device)
            
            # Set padding token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model and tokenizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using the same method as ingestion."""
        try:
            tokens = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(**tokens).last_hidden_state.mean(dim=1).squeeze()
            
            return embedding.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """
        OpenWebUI compatible pipe method that yields results as a stream.
        
        Args:
            prompt (str, optional): The input query/prompt
            **kwargs: Additional keyword arguments from OpenWebUI
                user_message (str): The user's message
        """
        try:
            # Get the query from either prompt or user_message
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided in either prompt or user_message")
                
            logger.info(f"Pipe processing query: {query}")

            # Generate embeddings using the same method as ingestion
            query_vector = self.generate_embedding(query)
            logger.debug(f"Generated query embeddings with size {len(query_vector)}")

            # Perform search using the same format as ingestion
            response = self.qdrant_client.search(
                collection_name="sigma_rules",
                query_vector=("default", query_vector),
                limit=5
            )
            logger.debug(f"Received {len(response)} results from Qdrant")

            # Stream results
            if response:
                yield "Based on the search, here are the relevant Sigma rules:\n\n"
                for idx, res in enumerate(response, 1):
                    if isinstance(res.payload, dict):
                        result_text = f"Result {idx}:\n"
                        # Prioritize important fields
                        priority_fields = ['title', 'description', 'detection', 'author', 'tags']
                        for field in priority_fields:
                            if field in res.payload and res.payload[field]:
                                result_text += f"{field}: {res.payload[field]}\n"
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
        """Legacy method for non-streaming results."""
        logger.info(f"Run method called with prompt: {prompt}")
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            return [{"text": "".join(results)}]
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)
            return [{"error": str(e)}]
