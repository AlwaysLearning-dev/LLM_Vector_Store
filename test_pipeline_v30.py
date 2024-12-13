import logging
import sys
import json
import re
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
        self.qdrant_client = QdrantClient(host="qdrant", port=6333)
        logger.info("Pipeline initialized")

    def search_qdrant(self, term: str) -> List[Dict]:
        """Search for Sigma rules in Qdrant."""
        matches = []
        result = self.qdrant_client.scroll(
            collection_name="sigma_rules",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        for point in result[0]:
            payload = point.payload
            if (term.lower() in str(payload.get('title', '')).lower() or 
                term.lower() in str(payload.get('description', '')).lower() or
                term.lower() in str(payload.get('detection', '')).lower()):
                matches.append(payload)
        
        return matches

    def format_rule(self, rule: Dict) -> str:
        """Format a Sigma rule for output."""
        output = []
        if rule.get('title'): 
            output.append(f"Title: {rule['title']}")
        if rule.get('description'): 
            output.append(f"Description: {rule['description']}")
        if rule.get('detection'):
            output.append("Detection:")
            output.append(json.dumps(rule['detection'], indent=2))
        return '\n'.join(output)

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        # Get query from either source
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return
            
        # Look for terms in quotes
        search_terms = re.findall(r'"([^"]*)"', query)
        
        # If we found quoted terms, search Qdrant
        if search_terms:
            for term in search_terms:
                rules = self.search_qdrant(term)
                if rules:
                    yield f"\nFound {len(rules)} Sigma rules for '{term}':\n\n"
                    for i, rule in enumerate(rules, 1):
                        yield f"Rule {i}:\n{self.format_rule(rule)}\n\n"
                else:
                    yield f"\nNo Sigma rules found for '{term}'\n"
        else:
            # For non-search queries, pass through to regular LLM
            return

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        results = list(self.pipe(prompt=prompt, **kwargs))
        if results:  # If we have Qdrant results
            return [{"text": "".join(results)}]
        return []  # Empty list signals to use normal LLM processing
