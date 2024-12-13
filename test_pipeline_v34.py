import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient

class Pipeline:
    def __init__(self):
        self.client = QdrantClient(host="qdrant", port=6333)

    def search_rules(self, term: str) -> str:
        """Search for rules and return formatted string result."""
        result = self.client.scroll(
            collection_name="sigma_rules",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        matches = []
        for point in result[0]:
            payload = point.payload
            # Search in title and description
            if term.lower() in str(payload.get('title', '')).lower() or \
               term.lower() in str(payload.get('description', '')).lower():
                matches.append(payload)

        if not matches:
            return f"No Sigma rules found matching '{term}'"

        # Format results
        output = [f"Found {len(matches)} Sigma rules:"]
        for i, rule in enumerate(matches, 1):
            output.append(f"\nRule {i}:")
            if rule.get('title'):
                output.append(f"Title: {rule['title']}")
            if rule.get('description'):
                output.append(f"Description: {rule['description']}")
            output.append("-" * 40)

        return "\n".join(output)

    def run(self, prompt: str = None, **kwargs) -> List[Dict[str, Any]]:
        """Process input and return results."""
        query = prompt or kwargs.get('user_message', '')
        
        # Check for quoted search term
        if '"' in query:
            start = query.find('"') + 1
            end = query.find('"', start)
            if start > 0 and end > start:
                term = query[start:end]
                result = self.search_rules(term)
                return [{"text": result}]
        
        # If no quotes or no results, return empty list to use normal LLM
        return []
