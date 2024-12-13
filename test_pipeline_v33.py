import logging
import sys
import json
from typing import List, Dict, Any, Generator

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    logger.info("Successfully imported Qdrant client")
except ImportError as e:
    logger.error(f"Failed to import Qdrant client: {e}")
    raise

class Pipeline:
    def __init__(self) -> None:
        try:
            self.qdrant_client = QdrantClient(host="qdrant", port=6333)
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def search_sigma_rules(self, term: str) -> List[Dict]:
        """Search for Sigma rules in Qdrant."""
        logger.info(f"Searching for term: {term}")
        
        try:
            # Get all points from Qdrant
            result = self.qdrant_client.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            matches = []
            for point in result[0]:
                payload = point.payload
                # Search in relevant fields
                if any(term.lower() in str(v).lower() 
                      for v in [payload.get('title', ''),
                              payload.get('description', ''),
                              json.dumps(payload.get('detection', ''))]
                     ):
                    matches.append(payload)
            
            logger.info(f"Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

    def format_rule(self, rule: Dict) -> str:
        """Format a single Sigma rule for output."""
        lines = []
        
        if rule.get('title'):
            lines.append(f"Title: {rule['title']}")
        if rule.get('description'):
            lines.append(f"Description: {rule['description']}")
        if rule.get('detection'):
            lines.append("Detection Rules:")
            lines.append(json.dumps(rule['detection'], indent=2))
        if rule.get('level'):
            lines.append(f"Level: {rule['level']}")
        if rule.get('tags'):
            lines.append(f"Tags: {', '.join(rule['tags'])}")
        
        return '\n'.join(lines)

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        try:
            query = prompt or kwargs.get('user_message', '')
            if not query:
                yield ""
                return
            
            # Check if this is a direct quoted search
            if query.count('"') >= 2:
                # Extract the term between quotes
                start = query.find('"') + 1
                end = query.find('"', start)
                if start > 0 and end > start:
                    search_term = query[start:end]
                    logger.info(f"Extracted search term: {search_term}")
                    
                    # Search Qdrant
                    matches = self.search_sigma_rules(search_term)
                    
                    if matches:
                        yield f"Found {len(matches)} Sigma rules matching '{search_term}':\n\n"
                        for idx, rule in enumerate(matches, 1):
                            yield f"Rule {idx}:\n"
                            yield self.format_rule(rule)
                            yield "\n" + "-"*80 + "\n\n"
                    else:
                        yield f"No Sigma rules found matching '{search_term}'\n"
                    return
            
            # If not a search query, yield empty string to pass to LLM
            yield ""
            
        except Exception as e:
            logger.error(f"Error in pipe: {e}")
            yield f"Error processing query: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        results = list(self.pipe(prompt=prompt, **kwargs))
        
        # Filter out empty strings
        results = [r for r in results if r.strip()]
        
        if not results:
            # Return empty list to signal using normal LLM
            return []
        
        # Return Qdrant search results
        return [{"text": "".join(results)}]
