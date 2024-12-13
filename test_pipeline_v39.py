import logging
import sys
import json
import re
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
            
            # Initialize small, efficient model on GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            self.model.to(self.device)
            logger.info(f"Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract only phrases enclosed in quotes."""
        phrases = re.findall(r'"([^"]*)"', query)
        logger.debug(f"Extracted quoted phrases: {phrases}")
        return phrases

    def search_rules(self, search_terms: List[str]) -> List[Dict]:
        """Search for rules using exact phrase matching."""
        logger.debug(f"Searching for terms: {search_terms}")
        matches = set()
        
        # Get points
        scroll_result = self.qdrant_client.scroll(
            collection_name="sigma_rules",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        for point in scroll_result[0]:
            payload = point.payload
            
            # Check each search term
            for term in search_terms:
                term_lower = term.lower()
                found = False
                
                # Search in title and description first
                if payload.get('title') and term_lower in payload['title'].lower():
                    found = True
                elif payload.get('description') and term_lower in payload['description'].lower():
                    found = True
                # Only search other fields if not found in title/description
                elif payload.get('detection') and term_lower in json.dumps(payload['detection']).lower():
                    found = True
                    
                if found:
                    matches.add((payload.get('title', ''), json.dumps(payload, sort_keys=True)))
                    break
        
        unique_matches = [json.loads(match[1]) for match in matches]
        logger.debug(f"Found {len(unique_matches)} unique matches")
        return unique_matches

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
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided")

            logger.info(f"Processing query: {query}")

            # Extract quoted search terms
            search_terms = self.extract_search_terms(query)
            if search_terms:
                matches = self.search_rules(search_terms)
                logger.info(f"Found {len(matches)} matches")

                if matches:
                    yield f"Found {len(matches)} Sigma rules matching your query:\n\n"
                    for idx, match in enumerate(matches, 1):
                        result_text = f"Match {idx}:\n"
                        result_text += self.format_rule_output(match)
                        result_text += "\n\n" + "-" * 80 + "\n\n"
                        yield result_text
                else:
                    yield "No matches found for the quoted terms.\n"
            else:
                # Fallback to regular LLM response
                yield "No quoted terms found; switching to default LLM response...\n"
                # Integrate LLM generation logic here using OpenWebUI and Ollama
                
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
