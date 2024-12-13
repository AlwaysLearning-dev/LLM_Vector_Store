from qdrant_client import QdrantClient

class Pipeline:
    def __init__(self):
        # Initialize connection once
        self.client = QdrantClient(host="qdrant", port=6333)

    def pipe(self, prompt=None, **kwargs):
        # Get the query
        query = prompt or kwargs.get('user_message', '')
        
        # If no quotes, return immediately to use LLM
        if '"' not in query:
            return ""
        
        # Extract search term
        start = query.find('"') + 1
        end = query.find('"', start)
        if start <= 0 or end <= start:
            return ""
            
        term = query[start:end]
        
        # Do a single search
        result = self.client.scroll(
            collection_name="sigma_rules",
            limit=5,  # Just get first 5
            with_payload=True,
            with_vectors=False
        )
        
        # Build response string
        output = []
        for point in result[0]:
            payload = point.payload
            if term.lower() in str(payload.get('title', '')).lower() or \
               term.lower() in str(payload.get('description', '')).lower():
                output.append(f"Title: {payload.get('title', 'No title')}")
                output.append(f"Description: {payload.get('description', 'No description')}\n")
        
        if output:
            return "\n".join(output)
        return "No matching rules found."

    def run(self, prompt=None, **kwargs):
        # Get response
        response = self.pipe(prompt=prompt, **kwargs)
        
        # If empty, use LLM
        if not response:
            return []
            
        # Otherwise return our search results
        return [{"text": response}]
