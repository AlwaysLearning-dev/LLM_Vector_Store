from typing import List, Dict, Any, Generator
import logging
import json
import re
import os
import requests
from qdrant_client import QdrantClient

class Pipeline:
    def __init__(self):
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        self.llm_url = "http://ollama:11434"
        self.model = os.getenv("LLAMA_MODEL_NAME", "llama3.2")

    def search_qdrant(self, search_term: str) -> List[Dict]:
        """Search Qdrant for matches."""
        try:
            # Get records
            result = self.qdrant.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            matches = []
            if result and result[0]:
                # Search through records
                for point in result[0]:
                    payload = point.payload
                    if (search_term.lower() in payload.get('title', '').lower() or
                        search_term.lower() in payload.get('description', '').lower()):
                        matches.append(payload)

            return matches
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and return results."""
        # Get query
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            # Look for quoted term
            match = re.search(r'"([^"]+)"', query)
            if match:
                search_term = match.group(1)
                
                # Search Qdrant
                matches = self.search_qdrant(search_term)
                
                if matches:
                    yield f"Found {len(matches)} Sigma rules matching '{search_term}':\n\n"
                    for idx, rule in enumerate(matches, 1):
                        result = f"Rule {idx}:\n"
                        if rule.get('title'):
                            result += f"Title: {rule['title']}\n"
                        if rule.get('description'):
                            result += f"Description: {rule['description']}\n"
                        if rule.get('detection'):
                            result += f"Detection:\n{json.dumps(rule['detection'], indent=2)}\n"
                        result += "\n" + "-"*80 + "\n\n"
                        yield result
                else:
                    yield f"No Sigma rules found matching '{search_term}'"
                return

            # If no quoted term, use LLM
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
