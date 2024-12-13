from qdrant_client import QdrantClient

class Pipeline:
    def __init__(self):
        self.client = QdrantClient(host="qdrant", port=6333)

    def process_input(self, text: str) -> str:
        """Process the input and return a string response."""
        # Check for quoted terms
        if '"' in text:
            start = text.find('"') + 1
            end = text.find('"', start)
            if start > 0 and end > start:
                search_term = text[start:end]
                
                # Search Qdrant
                result = self.client.scroll(
                    collection_name="sigma_rules",
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )
                
                matches = []
                for point in result[0]:
                    payload = point.payload
                    if search_term.lower() in str(payload.get('title', '')).lower() or \
                       search_term.lower() in str(payload.get('description', '')).lower():
                        matches.append(payload)
                
                if matches:
                    output = [f"Found {len(matches)} Sigma rules matching '{search_term}':"]
                    for i, rule in enumerate(matches, 1):
                        output.append(f"\nRule {i}:")
                        if rule.get('title'):
                            output.append(f"Title: {rule['title']}")
                        if rule.get('description'):
                            output.append(f"Description: {rule['description']}")
                        if rule.get('detection'):
                            output.append("Detection:")
                            output.append(str(rule['detection']))
                        output.append("-" * 40)
                    return "\n".join(output)
                else:
                    return f"No Sigma rules found matching '{search_term}'"
        
        # For non-search queries, return empty string to use LLM
        return ""

    def run(self, prompt: str = None, **kwargs):
        query = prompt or kwargs.get('user_message', '')
        result = self.process_input(query)
        
        if result:  # If we got search results
            return [{"text": result}]
        
        # Otherwise, return empty list to use LLM
        return []
