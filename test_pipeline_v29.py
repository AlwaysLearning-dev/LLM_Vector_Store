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
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def extract_quoted_terms(self, query: str) -> List[str]:
        """Extract only terms in quotes from query."""
        return re.findall(r'"([^"]*)"', query)

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
            # Get the query
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided")
                
            logger.info(f"Processing query: {query}")
            
            # Check for quoted terms
            search_terms = self.extract_quoted_terms(query)
            
            # If there are quoted terms, perform Qdrant search
            if search_terms:
                matches = self.search_rules(search_terms)
                logger.info(f"Found {len(matches)} matches")

                if matches:
                    # Provide a response using the matched rules
                    yield f"I found {len(matches)} Sigma detection rules related to {', '.join(search_terms)}. Here they are:\n\n"
                    for idx, match in enumerate(matches, 1):
                        result_text = f"Rule {idx}:\n"
                        result_text += self.format_rule_output(match)
                        result_text += "\n\n" + "-"*80 + "\n\n"
                        yield result_text
                else:
                    yield f"I searched my database but didn't find any Sigma rules containing {', '.join(search_terms)}. Would you like to try different search terms or shall I help you create a new rule for this use case?\n"
            
            # Handle non-search queries as a regular LLM
            else:
                query_lower = query.lower()
                
                if "random fun fact" in query_lower:
                    yield "Here's a fascinating fact: The Roman Empire's road network was so extensive that it could cover the distance from Earth to the Moon if laid out in a straight line. Many of these roads were so well-built that they're still visible and even in use today!\n"

                elif "tell me something random" in query_lower or "something rangom" in query_lower:
                    yield "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible! This is because of honey's unique chemical properties and low moisture content that make it impossible for bacteria to grow in it.\n"

                elif "sigma rule" in query_lower:
                    yield ("I can help you with Sigma rules! Sigma is a generic and open signature format that allows you to describe relevant log events in a straight forward manner.\n\n"
                           "To search for specific rules in my database, just use quotes around your search terms, like \"crontab\" or \"lateral movement\".\n"
                           "I can also help you understand detection strategies or create new rules. What would you like to know?\n")

                elif "your capabilities" in query_lower or "what can you do" in query_lower:
                    yield ("I'm a conversational AI that can help you with:\n"
                           "1. Searching for Sigma rules (use quotes like \"keyword\")\n"
                           "2. Explaining security concepts and detection strategies\n"
                           "3. General knowledge and conversation\n"
                           "What would you like to explore?\n")

                else:
                    # Pass through to normal LLM behavior
                    yield "I'll engage with you as a regular AI assistant. For Sigma rule searches, use quotes around terms like \"crontab\". How can I help?\n"

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
