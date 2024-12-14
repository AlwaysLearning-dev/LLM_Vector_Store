from typing import List, Dict, Any, Generator
from pydantic import BaseModel
import logging
import sys
import json
import re
import os
import requests
from qdrant_client import QdrantClient

def write_debug(msg: str):
    """Write debug message to file."""
    try:
        with open('/tmp/pipeline_debug.log', 'a', encoding='utf-8') as f:
            f.write(f"{msg}\n")
            f.flush()
    except Exception as e:
        print(f"Error writing to log: {e}")

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        LLAMA_MODEL_NAME: str
        LLAMA_BASE_URL: str

    def __init__(self) -> None:
        try:
            # Load configuration
            self.valves = self.Valves(
                **{
                    "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                    "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                    "LLAMA_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                    "LLAMA_BASE_URL": os.getenv("LLAMA_BASE_URL", "http://ollama:11434"),
                }
            )
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=self.valves.QDRANT_HOST, 
                port=self.valves.QDRANT_PORT,
                timeout=10.0
            )
            
        except Exception as e:
            write_debug(f"Error initializing pipeline: {e}")
            raise

    def get_search_term(self, query: str) -> str:
        """Extract search term from query text."""
        # Clean query
        query = query.strip()
        
        # Handle both single and double quotes
        if query.startswith('"') and query.endswith('"'):
            return query[1:-1].strip()
        if query.startswith("'") and query.endswith("'"):
            return query[1:-1].strip()
            
        # Look for quoted terms within query
        match = re.search(r'["\']([^"\']+)["\']', query)
        if match:
            return match.group(1).strip()
            
        return ""

    def search_rules(self, term: str) -> List[Dict]:
        """Search Sigma rules for given term."""
        try:
            write_debug(f"Searching for: {term}")
            
            # Get rules from Qdrant
            result = self.qdrant_client.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            # Search for matches
            matches = []
            for point in result[0]:
                payload = point.payload
                
                # Check title and description
                if (term.lower() in str(payload.get('title', '')).lower() or 
                    term.lower() in str(payload.get('description', '')).lower()):
                    matches.append(payload)
                    write_debug(f"Found matching rule: {payload.get('title')}")
            
            write_debug(f"Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            write_debug(f"Search error: {e}")
            return []

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and generate response."""
        try:
            # Get query
            query = prompt or kwargs.get('user_message', '')
            if not query:
                return
                
            write_debug(f"Processing query: {query}")
            
            # Extract search term
            term = self.get_search_term(query)
            
            # If we have a search term, search Qdrant
            if term:
                write_debug(f"Searching for term: {term}")
                matches = self.search_rules(term)
                
                if matches:
                    yield f"Found {len(matches)} Sigma rules matching '{term}':\n\n"
                    for idx, match in enumerate(matches, 1):
                        result = f"Rule {idx}:\n"
                        if match.get('title'):
                            result += f"Title: {match['title']}\n"
                        if match.get('description'):
                            result += f"Description: {match['description']}\n"
                        if match.get('detection'):
                            result += f"Detection:\n{json.dumps(match['detection'], indent=2)}\n"
                        result += "\n" + "-"*80 + "\n\n"
                        yield result
                else:
                    yield f"No Sigma rules found matching '{term}'\n"
                return
            
            # If no search term, use LLM
            write_debug("Using LLM")
            response = requests.post(
                url=f"{self.valves.LLAMA_BASE_URL}/api/generate",
                json={"model": self.valves.LLAMA_MODEL_NAME, "prompt": query},
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get("response", "")
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            write_debug(f"Error in pipe: {e}")
            yield f"Error: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run the pipeline."""
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            if not results:
                return []
            return [{"text": "".join(results)}]
        except Exception as e:
            write_debug(f"Error in run: {e}")
            return [{"error": str(e)}]
