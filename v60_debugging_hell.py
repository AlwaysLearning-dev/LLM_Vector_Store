from typing import List, Dict, Any, Generator
from pydantic import BaseModel
import logging
import sys
import json
import re
import os
import requests
from datetime import datetime
from qdrant_client import QdrantClient

def write_debug(msg: str):
    """Write debug message directly to file with timestamp."""
    try:
        with open('/tmp/pipeline_direct.log', 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            f.write(f"{timestamp} - {msg}\n")
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
            self.valves = self.Valves(
                **{
                    "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                    "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                    "LLAMA_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                    "LLAMA_BASE_URL": os.getenv("LLAMA_BASE_URL", "http://ollama:11434"),
                }
            )
            write_debug("Configuration loaded")

            self.qdrant_client = QdrantClient(
                host=self.valves.QDRANT_HOST, 
                port=self.valves.QDRANT_PORT,
                timeout=10.0
            )
            write_debug("Qdrant client initialized")
            
        except Exception as e:
            write_debug(f"Pipeline initialization failed: {str(e)}")
            raise

    def get_search_term(self, query: str) -> str:
        """Extract the search term from a query with quotes."""
        write_debug(f"Getting search term from query: {query}")
        
        # Look for text between quotes
        match = re.search(r'"([^"]+)"', query)
        if match:
            term = match.group(1).strip()
            write_debug(f"Found search term: {term}")
            return term
        return ""

    def search_qdrant(self, term: str) -> List[Dict]:
        """Search Qdrant for exact term match."""
        write_debug(f"Searching Qdrant for term: {term}")
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            matches = []
            for point in scroll_result[0]:
                payload = point.payload
                
                # Look for exact term in title and description
                if (term.lower() in str(payload.get('title', '')).lower() or 
                    term.lower() in str(payload.get('description', '')).lower()):
                    write_debug(f"Found match: {payload.get('title')}")
                    matches.append(payload)
            
            write_debug(f"Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            write_debug(f"Error searching Qdrant: {str(e)}")
            return []

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        try:
            query = prompt or kwargs.get('user_message', '')
            write_debug(f"Processing query: {query}")
            
            # First check for search term
            search_term = self.get_search_term(query)
            if search_term:
                write_debug("Found search term, performing Qdrant search")
                matches = self.search_qdrant(search_term)
                
                if matches:
                    yield f"Found {len(matches)} Sigma rules matching '{search_term}':\n\n"
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
                    yield f"No Sigma rules found matching '{search_term}'\n"
                return
            
            # If no search term, use LLM
            write_debug("No search term found, using LLM")
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
            error_msg = f"Error in pipe method: {str(e)}"
            write_debug(error_msg)
            yield error_msg

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            if not results:
                write_debug("No results generated")
                return []
            
            write_debug("Results generated successfully")
            return [{"text": "".join(results)}]
            
        except Exception as e:
            error_msg = f"Error in run method: {str(e)}"
            write_debug(error_msg)
            return [{"error": error_msg}]
