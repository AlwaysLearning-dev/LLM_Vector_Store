from typing import List, Dict, Any, Generator
import logging
import json
import re
import os
import requests
from qdrant_client import QdrantClient

class Pipeline:
    def __init__(self):
        # Initialize clients
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        self.llm_url = "http://ollama:11434"
        self.model = os.getenv("LLAMA_MODEL_NAME", "llama3.2")

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract all potential search terms from query."""
        # Find quoted phrases
        phrases = re.findall(r'"([^"]*)"', query)
        
        # If no quotes, treat the whole query as terms
        if not phrases:
            terms = [term.strip() for term in query.split() if term.strip()]
            return terms
            
        return phrases

    def search_qdrant(self, terms: List[str]) -> List[Dict]:
        """Search for rules matching any of the terms."""
        try:
            # Get all records
            result = self.qdrant.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            matches = set()  # Use set to avoid duplicates
            if result and result[0]:
                for point in result[0]:
                    payload = point.payload
                    
                    # Search through all relevant fields
                    searchable_text = ' '.join(str(v) for v in [
                        payload.get('title', ''),
                        payload.get('description', ''),
                        json.dumps(payload.get('detection', {}))
                    ]).lower()
                    
                    # Check each term
                    for term in terms:
                        if term.lower() in searchable_text:
                            # Use title as key to ensure uniqueness
                            matches.add((payload.get('title', ''), json.dumps(payload)))
                            break

            # Convert back to list of unique matches
            return [json.loads(match[1]) for match in matches]

        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    def format_rule(self, rule: Dict) -> str:
        """Format a rule for output."""
        output = []
        if rule.get('title'):
            output.append(f"Title: {rule['title']}")
        if rule.get('description'):
            output.append(f"Description: {rule['description']}")
        if rule.get('detection'):
            output.append("Detection:")
            output.append(json.dumps(rule['detection'], indent=2))
        return '\n'.join(output)

    def looks_like_search(self, query: str) -> bool:
        """Determine if this is likely a search query."""
        # Check for quoted terms
        if '"' in query:
            return True
            
        # Check for search-related words
        search_words = ['search', 'find', 'show', 'list', 'get']
        query_words = query.lower().split()
        for word in search_words:
            if word in query_words:
                return True
                
        # If it's just 1-3 words, treat as search
        if len(query_words) <= 3:
            return True
            
        return False

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and return results."""
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            # Check if this looks like a search query
            if self.looks_like_search(query):
                # Extract terms and search
                terms = self.extract_search_terms(query)
                if terms:
                    matches = self.search_qdrant(terms)
                    
                    if matches:
                        yield f"Found {len(matches)} Sigma rules matching your query:\n\n"
                        for idx, rule in enumerate(matches, 1):
                            result = f"Rule {idx}:\n"
                            result += self.format_rule(rule)
                            result += "\n" + "-"*80 + "\n\n"
                            yield result
                        return
                    else:
                        yield f"No Sigma rules found matching: {', '.join(terms)}\n"
                        return

            # If not a search or no results, use LLM
            response = requests.post(
                url=f"{self.llm_url}/api/generate",
                json={"model": self.model, "prompt": query},
                stream=True
            )
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get("response", "")
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            yield f"Error: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run pipeline and return results."""
        results = list(self.pipe(prompt=prompt, **kwargs))
        if not results:
            return []
        return [{"text": "".join(results)}]
