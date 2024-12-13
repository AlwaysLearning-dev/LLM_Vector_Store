from typing import Generator
from qdrant_client import QdrantClient

class Pipeline:
    def __init__(self):
        self.client = QdrantClient(host="qdrant", port=6333)

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Required generator method for OpenWebUI."""
        query = prompt or kwargs.get('user_message', '')
        
        # Only proceed if we find a quoted term
        if '"' not in query:
            return
            
        # Extract search term
        start = query.find('"') + 1
        end = query.find('"', start)
        if start <= 0 or end <= start:
            return
            
        search_term = query[start:end]
        
        # Get all rules
        result = self.client.scroll(
            collection_name="sigma_rules",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        # Track if we found anything
        found_any = False
        
        # Search through rules
        for point in result[0]:
            payload = point.payload
            if search_term.lower() in str(payload.get('title', '')).lower() or \
               search_term.lower() in str(payload.get('description', '')).lower():
                found_any = True
                yield f"\nTitle: {payload.get('title', 'No title')}"
                yield f"\nDescription: {payload.get('description', 'No description')}"
                yield "\n---\n"
        
        if not found_any:
            yield f"\nNo Sigma rules found for '{search_term}'\n"

    def run(self, prompt: str = None, **kwargs):
        # Collect all generated output
        results = list(self.pipe(prompt=prompt, **kwargs))
        
        # If we got results, return them
        if results:
            return [{"text": "".join(results)}]
            
        # Otherwise return empty list to use LLM
        return []
