import logging
import sys
import os
import json
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
            
            # Initialize the model and tokenizer
            model_path = "/models"
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
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using the same method as the ingestion script."""
        # Format the query text as JSON string, similar to ingestion
        query_text = json.dumps({"query": text}, indent=2)
        
        tokens = self.tokenizer(
            query_text, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(**tokens)
            embedding = output.last_hidden_state.mean(dim=1).squeeze()
            
        return embedding.cpu().numpy().tolist()

    def format_rule_output(self, payload: Dict) -> str:
        """Format a Sigma rule payload for output."""
        important_fields = ['title', 'description', 'detection', 'tags', 'level']
        result = []
        
        for field in important_fields:
            if field in payload and payload[field]:
                if field == 'detection':
                    # Format detection rules more readably
                    result.append(f"Detection Rules:")
                    detection = payload[field]
                    if isinstance(detection, dict):
                        for k, v in detection.items():
                            result.append(f"  {k}: {v}")
                elif field == 'tags':
                    if payload[field]:
                        result.append(f"Tags: {', '.join(payload[field])}")
                else:
                    result.append(f"{field.capitalize()}: {payload[field]}")
        
        return '\n'.join(result)

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """OpenWebUI compatible pipe method that yields results as a stream."""
        try:
            # Get the query from either prompt or user_message
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided in either prompt or user_message")
                
            logger.info(f"Pipe processing query: {query}")

            # Generate embeddings
            query_vector = self.generate_embedding(query)
            logger.debug(f"Generated query embeddings with size {len(query_vector)}")

            # Create a named vector
            named_vector = NamedVector(
                name="default",
                vector=query_vector
            )

            # Perform search with higher limit and score threshold
            response = self.qdrant_client.search(
                collection_name="sigma_rules",
                query_vector=named_vector,
                limit=5,
                score_threshold=0.5  # Add minimum similarity threshold
            )
            logger.debug(f"Received {len(response)} results from Qdrant")

            # Stream results
            if response:
                yield f"Found {len(response)} relevant Sigma rules for '{query}':\n\n"
                for idx, res in enumerate(response, 1):
                    if isinstance(res.payload, dict):
                        result_text = f"Result {idx} (Similarity: {res.score:.2f}):\n"
                        result_text += self.format_rule_output(res.payload)
                        result_text += "\n\n"
                        yield result_text
                    else:
                        yield f"Result {idx}: {res.payload}\n\n"
            else:
                yield f"No Sigma rules found matching the query: {query}"

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
