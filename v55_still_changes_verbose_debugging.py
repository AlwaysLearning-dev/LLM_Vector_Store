from typing import List, Dict, Any, Generator, Union
from pydantic import BaseModel
import logging
import sys
import json
import re
import os
import requests
from qdrant_client import QdrantClient

# Set up verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        LLAMA_MODEL_NAME: str
        LLAMA_BASE_URL: str

    def __init__(self) -> None:
        logger.info("=== Pipeline Initialization Starting ===")
        try:
            # Valves Configuration
            self.valves = self.Valves(
                **{
                    "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                    "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                    "LLAMA_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                    "LLAMA_BASE_URL": os.getenv("LLAMA_BASE_URL", "http://ollama:11434"),
                }
            )
            logger.info(f"Configuration loaded: {self.valves}")

            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=self.valves.QDRANT_HOST, 
                port=self.valves.QDRANT_PORT,
                timeout=10.0
            )
            
            # Test Qdrant connection
            collection_info = self.qdrant_client.get_collection('sigma_rules')
            logger.info(f"Qdrant connection successful. Collection info: {collection_info}")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            raise

    def search_qdrant(self, term: str) -> List[Dict]:
        logger.info(f"=== Starting Qdrant search for term: {term} ===")
        try:
            # Get records from Qdrant
            scroll_result = self.qdrant_client.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result or not scroll_result[0]:
                logger.warning("No records returned from Qdrant")
                return []
            
            logger.info(f"Retrieved {len(scroll_result[0])} records from Qdrant")
            
            # Search for matches
            matches = []
            for point in scroll_result[0]:
                payload = point.payload
                logger.debug(f"Checking rule: {payload.get('title', 'No title')}")
                
                # Search in all relevant fields
                searchable_text = ' '.join(str(v) for v in [
                    payload.get('title', ''),
                    payload.get('description', ''),
                    json.dumps(payload.get('detection', {}))
                ]).lower()
                
                if term.lower() in searchable_text:
                    logger.info(f"Found match: {payload.get('title')}")
                    matches.append(payload)
            
            logger.info(f"Found {len(matches)} matches for term: {term}")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}", exc_info=True)
            return []

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        logger.info("=== Starting pipe method ===")
        try:
            # Get query
            query = prompt or kwargs.get('user_message', '')
            logger.info(f"Processing query: {query}")
            
            # Look for quoted terms
            if '"' in query:
                search_terms = re.findall(r'"([^"]*)"', query)
                logger.info(f"Found search terms: {search_terms}")
                
                if search_terms:
                    # Do Qdrant search
                    logger.info("Starting Qdrant search")
                    all_matches = []
                    for term in search_terms:
                        matches = self.search_qdrant(term)
                        all_matches.extend(matches)
                    
                    if all_matches:
                        yield f"Found {len(all_matches)} Sigma rules matching your search:\n\n"
                        for idx, match in enumerate(all_matches, 1):
                            yield f"Rule {idx}:\n"
                            if match.get('title'):
                                yield f"Title: {match['title']}\n"
                            if match.get('description'):
                                yield f"Description: {match['description']}\n"
                            if match.get('detection'):
                                yield "Detection:\n"
                                yield f"{json.dumps(match['detection'], indent=2)}\n"
                            yield "\n" + "-"*80 + "\n\n"
                    else:
                        yield f"No Sigma rules found matching: {', '.join(search_terms)}\n"
                    return
            
            # If no quotes or no matches, use LLM
            logger.info("Using LLM for response")
            try:
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
                logger.error(f"Error with LLM: {e}")
                yield "Error generating response from LLM."
                
        except Exception as e:
            logger.error(f"Error in pipe method: {e}", exc_info=True)
            yield f"Error processing request: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        logger.info("=== Starting run method ===")
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            if not results:
                logger.warning("No results generated")
                return []
            
            logger.info("Results generated successfully")
            return [{"text": "".join(results)}]
            
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)
            return [{"error": str(e)}]
