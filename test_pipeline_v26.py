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
            
            # Verify collection exists and get info
            collection_info = self.qdrant_client.get_collection('sigma_rules')
            logger.info(f"Collection info: {collection_info}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def extract_keywords(self, query: str) -> List[str]:
        """Extract search keywords from the query."""
        # Keywords to ignore in search
        ignore_words = {'a', 'the', 'in', 'on', 'at', 'and', 'or', 'for', 'to', 'of', 'with', 'about', 'like', 'show', 'me', 'find', 'search', 'get', 'related', 'rules', 'sigma'}
        
        # Extract words and remove ignored ones
        words = query.lower().split()
        keywords = [word for word in words if word not in ignore_words]
        
        logger.debug(f"Extracted keywords: {keywords}")
        return keywords

    def search_rules(self, keywords: List[str]) -> List[Dict]:
        """Search for rules using the provided keywords."""
        logger.debug(f"Searching for keywords: {keywords}")
        matches = set()
        
        # Get all points
        scroll_result = self.qdrant_client.scroll(
            collection_name="sigma_rules",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        logger.debug(f"Retrieved {len(scroll_result[0])} points")
        
        for point in scroll_result[0]:
            payload = point.payload
            
            # Search through all fields for any keyword
            for keyword in keywords:
                found = False
                for key, value in payload.items():
                    if value and isinstance(value, (str, list, dict)):
                        value_str = str(value).lower()
                        if keyword in value_str:
                            # Use tuple of title and json string as hash-friendly key
                            matches.add((payload.get('title', ''), json.dumps(payload, sort_keys=True)))
                            found = True
                            break
                if found:
                    break
        
        # Convert back to list of unique matches
        unique_matches = [json.loads(match[1]) for match in matches]
        logger.debug(f"Found total {len(unique_matches)} unique matches")
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
            
        # Add tags if present
        if payload.get('tags'):
            result.append(f"Tags: {', '.join(payload['tags'])}")
        
        # Add level if present
        if payload.get('level'):
            result.append(f"Level: {payload['level']}")
            
        # Add filename
        if payload.get('filename'):
            result.append(f"Filename: {payload['filename']}")
            
        return '\n'.join(result)

    def analyze_query(self, query: str) -> Dict:
        """Analyze the query to determine the type of response needed."""
        query_lower = query.lower()
        
        # Check for different types of queries
        is_search = any(word in query_lower for word in ['show', 'find', 'search', 'get', 'related', 'about', 'what'])
        is_generation = any(word in query_lower for word in ['create', 'generate', 'write', 'make'])
        is_explanation = any(word in query_lower for word in ['explain', 'how', 'why', 'what is'])
        
        return {
            'is_search': is_search,
            'is_generation': is_generation,
            'is_explanation': is_explanation,
            'keywords': self.extract_keywords(query)
        }

    def generate_sigma_rule(self, query: str, similar_rules: List[Dict]) -> str:
        """Generate a new Sigma rule based on the query and similar rules."""
        # Extract relevant information from similar rules
        detection_patterns = []
        tags = set()
        levels = []
        
        for rule in similar_rules:
            if rule.get('detection'):
                detection_patterns.append(rule['detection'])
            if rule.get('tags'):
                tags.update(rule['tags'])
            if rule.get('level'):
                levels.append(rule['level'])
        
        # Generate new rule template
        new_rule = {
            "title": f"Custom rule for: {query}",
            "description": f"This rule was generated based on the query: {query}",
            "references": ["Generated by OpenWebUI Pipeline"],
            "tags": list(tags) if tags else ["attack.execution"],
            "level": max(set(levels), key=levels.count) if levels else "medium",
            "similar_rules": [rule.get('title') for rule in similar_rules if rule.get('title')]
        }
        
        return (
            f"Based on your query and similar rules, here's a suggested Sigma rule template:\n\n"
            f"Title: {new_rule['title']}\n"
            f"Description: {new_rule['description']}\n"
            f"Tags: {', '.join(new_rule['tags'])}\n"
            f"Level: {new_rule['level']}\n\n"
            f"Similar rules that were used as reference:\n"
            + '\n'.join(f"- {title}" for title in new_rule['similar_rules'])
        )

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """OpenWebUI compatible pipe method that yields results as a stream."""
        try:
            # Get the query from either prompt or user_message
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided in either prompt or user_message")
                
            logger.info(f"Processing query: {query}")
            
            # Analyze the query
            analysis = self.analyze_query(query)
            logger.debug(f"Query analysis: {analysis}")
            
            # Search for relevant rules
            matches = self.search_rules(analysis['keywords'])
            logger.info(f"Found {len(matches)} matching rules")

            # Generate appropriate response
            if analysis['is_search']:
                if matches:
                    yield f"Found {len(matches)} relevant Sigma rules:\n\n"
                    for idx, match in enumerate(matches, 1):
                        result_text = f"Match {idx}:\n"
                        result_text += self.format_rule_output(match)
                        result_text += "\n\n" + "-"*80 + "\n\n"
                        yield result_text
                else:
                    yield f"No Sigma rules found matching your query: {query}\n"

            elif analysis['is_generation']:
                if matches:
                    yield "I'll help you create a new Sigma rule based on similar existing rules.\n\n"
                    yield self.generate_sigma_rule(query, matches)
                else:
                    yield f"I can help you create a new Sigma rule, but I didn't find any similar rules for reference.\n"
                    
            elif analysis['is_explanation']:
                if matches:
                    yield f"I'll explain based on the {len(matches)} relevant Sigma rules I found:\n\n"
                    # Add explanations based on found rules
                    for match in matches:
                        yield f"Looking at {match.get('title')}:\n"
                        yield f"{match.get('description', 'No description available')}\n\n"
                else:
                    yield f"I can explain, but I don't have any specific Sigma rules about this topic.\n"
            
            else:
                # General conversation
                if matches:
                    yield f"I found {len(matches)} relevant Sigma rules that might help answer your question.\n\n"
                    for match in matches:
                        yield f"Based on {match.get('title')}:\n"
                        yield f"{match.get('description', 'No description available')}\n\n"
                else:
                    yield "I can help you with Sigma rules, searching for specific rules, creating new ones, or explaining detection strategies. What would you like to know?\n"

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
