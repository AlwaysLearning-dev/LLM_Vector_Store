import logging
import sys
import os
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
    from qdrant_client.http.models import NamedVector
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
            
            # List model files
            model_path = "/models"
            files = os.listdir(model_path)
            logger.info(f"Files in model directory: {files}")
            
            # Initialize the model and tokenizer
            logger.info(f"Loading model and tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            ).to(self.device)
            
            logger.info("Model and tokenizer initialized")
            
            # Test embedding generation
            test_embedding = self.generate_embedding("test")
            logger.info(f"Test embedding size: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using the same method as the ingestion script."""
        tokens = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(**tokens)
            embedding = output.last_hidden_state.mean(dim=1).squeeze()
            
        return embedding.cpu().numpy().tolist()

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

            # Create a named vector
            named_vector = NamedVector(
                name="default",
                vector=query_vector
            )

            # Perform search
            response = self.qdrant_client.search(
                collection_name="sigma_rules",
                query_vector=named_vector,
                limit=5
            )
            logger.debug(f"Received {len(response)} results from Qdrant")

            # Stream results
            if response:
                yield "Based on the search, here are the relevant Sigma rules:\n\n"
                for idx, res in enumerate(response, 1):
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
        """Legacy method for non-streaming results."""
        logger.info(f"Run method called with prompt: {prompt}")
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            return [{"text": "".join(results)}]
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)
            return [{"error": str(e)}]
