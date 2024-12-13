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
        """Search for rules using scroll method."""
        matches = []
        offset = None
        
        while True:
            scroll_result = self.qdrant_client.scroll(
                collection_name="sigma_rules",
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result[0]:
                break
                
            for point in scroll_result[0]:
                payload = point.payload
                # Search through all fields
                if any(str(query).lower() in str(v).lower() for v in payload.values() if v):
                    matches.append(payload)
            
            offset = scroll_result[1]
            if not offset:
                break
                
        return matches

    def format_rule_output(self, payload: Dict) -> str:
        """Format a Sigma rule payload for output."""
        result = []
        
        # Always include title and description first
        if payload.get('title'):
            result.append(f"Title: {payload['title']}")
        if payload.get('description'):
            result.append(f"Description: {payload['description']}")
            
        # Add detection rules if present
        if payload.get('detection'):
            result.append("Detection Rules:")
            result.append(json.dumps(payload['detection'], indent=2))
            
        # Add other relevant fields
        if payload.get('tags'):
            result.append(f"Tags: {', '.join(payload['tags'])}")
        if payload.get('level'):
            result.append(f"Level: {payload['level']}")
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
                
            logger.info(f"Pipe processing query: {query}")

            # Extract search terms (look for keywords after certain phrases)
            search_terms = [query.lower()]
            if "deal with" in query.lower():
                search_terms = [term.strip() for term in query.lower().split("deal with")[1].split()]
            elif "related to" in query.lower():
                search_terms = [term.strip() for term in query.lower().split("related to")[1].split()]
                
            # Search for matches
            all_matches = set()
            for term in search_terms:
                matches = self.search_rules(term)
                all_matches.update((match['title'], json.dumps(match)) for match in matches)

            # Convert back to list of unique matches
            unique_matches = [json.loads(match[1]) for match in all_matches]
            
            logger.debug(f"Found {len(unique_matches)} unique matches")

            # Stream results
            if unique_matches:
                yield f"Found {len(unique_matches)} Sigma rules matching your query:\n\n"
                for idx, match in enumerate(unique_matches, 1):
                    result_text = f"Match {idx}:\n"
                    result_text += self.format_rule_output(match)
                    result_text += "\n\n" + "-"*80 + "\n\n"
                    yield result_text
            else:
                yield f"No Sigma rules found matching your query. Try searching for specific terms like 'crontab', 'scheduled task', or 'systemd timer'."

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
