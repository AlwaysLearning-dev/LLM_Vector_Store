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

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract all potential search terms from query."""
        phrases = re.findall(r'"([^"]*)"', query)
        if not phrases:
            terms = [term.strip() for term in query.split() if term.strip()]
            return terms
        return phrases

    def search_qdrant(self, terms: List[str]) -> List[Dict]:
        """Search for rules matching any of the terms."""
        try:
            result = self.qdrant.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            matches = set()
            if result and result[0]:
                for point in result[0]:
                    payload = point.payload
                    searchable_text = ' '.join(str(v) for v in [
                        payload.get('title', ''),
                        payload.get('description', ''),
                        json.dumps(payload.get('detection', {}))
                    ]).lower()
                    
                    for term in terms:
                        if term.lower() in searchable_text:
                            matches.add((payload.get('title', ''), json.dumps(payload)))
                            break

            return [json.loads(match[1]) for match in matches]

        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    def format_rule(self, rule: Dict) -> str:
        """Format a rule in Sigma YAML format."""
        yaml_output = []
        
        # Format in standard Sigma YAML order
        fields = [
            'title', 'id', 'status', 'description', 'references', 'author',
            'date', 'modified', 'tags', 'logsource', 'detection',
            'falsepositives', 'level', 'filename'
        ]
        
        for field in fields:
            value = rule.get(field)
            if value is not None:
                if field == 'description':
                    # Use YAML block style for description
                    yaml_output.append(f"description: |")
                    for line in str(value).split('\n'):
                        yaml_output.append(f"    {line}")
                elif isinstance(value, (list, dict)):
                    # Format lists and dicts
                    yaml_output.append(f"{field}:")
                    if isinstance(value, list):
                        for item in value:
                            yaml_output.append(f"    - {item}")
                    else:
                        yaml_lines = json.dumps(value, indent=4).split('\n')
                        for line in yaml_lines[1:-1]:  # Skip first and last brackets
                            yaml_output.append(f"    {line}")
                else:
                    yaml_output.append(f"{field}: {value}")
        
        return '\n'.join(yaml_output)

    def looks_like_search(self, query: str) -> bool:
        """Determine if this is likely a search query."""
        if '"' in query:
            return True
        search_words = ['search', 'find', 'show', 'list', 'get']
        query_words = query.lower().split()
        if any(word in query_words for word in search_words):
            return True
        return len(query_words) <= 3

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and return results."""
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            if self.looks_like_search(query):
                terms = self.extract_search_terms(query)
                if terms:
                    matches = self.search_qdrant(terms)
                    
                    if matches:
                        yield f'<div style="font-size: 10pt;">\n'
                        yield f"Found {len(matches)} matching Sigma rules:\n\n"
                        for idx, rule in enumerate(matches, 1):
                            yield f"### Rule {idx}\n"
                            yield "```yaml\n"
                            yield self.format_rule(rule)
                            yield "\n```\n\n"
                        yield "</div>"
                        return
                    else:
                        yield f"No Sigma rules found matching: {', '.join(terms)}\n"
                        return

            # Use LLM for non-search queries
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
