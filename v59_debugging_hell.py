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
        write_debug("=== Pipeline Initialization Starting ===")
        try:
            self.valves = self.Valves(
                **{
                    "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                    "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                    "LLAMA_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                    "LLAMA_BASE_URL": os.getenv("LLAMA_BASE_URL", "http://ollama:11434"),
                }
            )

            self.qdrant_client = QdrantClient(
                host=self.valves.QDRANT_HOST, 
                port=self.valves.QDRANT_PORT,
                timeout=10.0
            )
            
            # Verify Qdrant connection
            collection_info = self.qdrant_client.get_collection('sigma_rules')
            write_debug(f"Connected to Qdrant collection: sigma_rules")

        except Exception as e:
            write_debug(f"Pipeline initialization failed: {str(e)}")
            raise

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract only the exact text between quotes."""
        write_debug(f"Extracting search terms from: {query}")
        terms = []
        
        # Find all text between quotes
        matches = re.finditer(r'"([^"]*)"', query)
        for match in matches:
            term = match.group(1).strip()
            if term:  # Only add non-empty terms
                terms.append(term)
                write_debug(f"Found search term: {term}")
        
        return terms

    def search_qdrant(self, term: str) -> List[Dict]:
        write_debug(f"=== Starting Qdrant search for term: {term} ===")
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
                
                # Search in all relevant fields
                searchable_text = ' '.join(str(v) for v in [
                    payload.get('title', ''),
                    payload.get('description', ''),
                    json.dumps(payload.get('detection', {}))
                ]).lower()
                
                if term.lower() in searchable_text:
                    write_debug(f"Found match: {payload.get('title')}")
                    matches.append(payload)
            
            write_debug(f"Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            write_debug(f"Error searching Qdrant: {str(e)}")
            return []

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        write_debug("=== Starting pipe method ===")
        try:
            query = prompt or kwargs.get('user_message', '')
            write_debug(f"Processing query: {query}")
            
            if '"' in query:
                # Extract exact terms from quotes
                search_terms = self.extract_search_terms(query)
                write_debug(f"Extracted terms: {search_terms}")
                
                if search_terms:
                    all_matches = []
                    for term in search_terms:
                        matches = self.search_qdrant(term)
                        all_matches.extend(matches)
                    
                    if all_matches:
                        yield f"Found {len(all_matches)} Sigma rules matching your search:\n\n"
                        for idx, match in enumerate(all_matches, 1):
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
                        yield f"No Sigma rules found matching: {', '.join(search_terms)}\n"
                    return
            
            # No quotes found, use LLM
            write_debug("Using LLM for response")
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
        write_debug("=== Starting run method ===")
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
