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
    from qdrant_client.http.models import (
        NamedVector, 
        Filter, 
        FieldCondition, 
        MatchText,
        MatchValue
    )
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
            
            # Verify collection exists and get info
            collection_info = self.qdrant_client.get_collection('sigma_rules')
            logger.info(f"Collection info: {collection_info}")
            
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

    def search_rules(self, query: str) -> List[Dict]:
        """Search for rules using direct scroll method."""
        logger.debug(f"Searching for: {query}")
        matches = []
        offset = None
        
        # Get first batch of points
        scroll_result = self.qdrant_client.scroll(
            collection_name="sigma_rules",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        logger.debug(f"Retrieved {len(scroll_result[0])} points")
        
        for point in scroll_result[0]:
            payload = point.payload
            logger.debug(f"Checking point with title: {payload.get('title')}")
            
            # Convert all values to string and search
            found = False
            for key, value in payload.items():
                if value and isinstance(value, (str, list, dict)):
                    value_str = str(value).lower()
                    if query.lower() in value_str:
                        logger.debug(f"Found match in field {key}")
                        found = True
                        break
            
            if found:
                logger.debug(f"Adding match: {payload.get('title')}")
                matches.append(payload)
        
        logger.debug(f"Found total {len(matches)} matches")
        return matches

    def format_rule_output(self, payload: Dict) -> str:
        """Format a Sigma rule payload for output."""
        result = []
        
        # Add title and description
        if payload.get('title'):
            result.append(f"Title: {payload['title']}")
        if payload.get('description'):
            result.append(f"Description: {payload['description']}")
            
        # Add detection rules
        if payload.get('detection'):
            result.append("Detection Rules:")
            result.append(json.dumps(payload['detection'], indent=2))
            
        # Add filename
        if payload.get('filename'):
            result.append(f"Filename: {payload['filename']}")
            
        return '\n'.join(result)

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """OpenWebUI compatible pipe method that yields results as a stream."""
        try:
            # Get the query from either prompt or user_message
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided in either prompt or user_message")
                
            logger.info(f"Processing query: {query}")
            
            # Extract search term (look for crontab specifically)
            search_term = "crontab"  # Default to searching for crontab
            if "cron" in query.lower():
                search_term = "cron"
            
            logger.debug(f"Using search term: {search_term}")
            
            # Search for matches
            matches = self.search_rules(search_term)
            logger.info(f"Found {len(matches)} matches for '{search_term}'")

            # Stream results
            if matches:
                yield f"Found {len(matches)} Sigma rules related to {search_term}:\n\n"
                for idx, match in enumerate(matches, 1):
                    result_text = f"Match {idx}:\n"
                    result_text += self.format_rule_output(match)
                    result_text += "\n\n" + "-"*80 + "\n\n"
                    yield result_text
            else:
                yield f"No Sigma rules found. Tried searching for '{search_term}'"

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
